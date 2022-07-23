import torch
import pytorch_lightning as pl
from torchmetrics import Accuracy
from weaver.models import get_classifier
from weaver.optimizers import get_optim
from weaver.optimizers.utils import exclude_wd
from weaver.schedulers import get_sched

__all__ = ['FixMatchClassifier', 'FlexMatchClassifier']


class AveragedModelWithBuffers(torch.optim.swa_utils.AveragedModel):
    def update_parameters(self, model):
        super().update_parameters(model)
        for a, b in zip(self.module.buffers(), model.buffers()):
            a.copy_(b.to(a.device))


class FixMatchCrossEntropy(torch.nn.Module):
    def __init__(self, temperature=1.0, threshold=0.95, reduction='mean'):
        super().__init__()
        self.threshold = threshold
        self.temperature = temperature
        self.reduction = reduction
        self.𝜇ₘₐₛₖ = None

    def forward(self, logits_s, logits_w):
        probs = torch.softmax(logits_w / self.temperature, dim=-1)
        max_probs, targets = probs.max(dim=-1)
        masks = (max_probs > self.threshold).float()

        loss = torch.nn.functional.cross_entropy(
            logits_s, targets, reduction='none') * masks
        self.𝜇ₘₐₛₖ = masks.mean().detach()

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class FlexMatchCrossEntropy(FixMatchCrossEntropy):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes

    def forward(self, logits_s, logits_w):
        probs = torch.softmax(logits_w / self.temperature, dim=-1)
        max_probs, targets = probs.max(dim=-1)

        β = targets.bincount(max_probs > self.threshold, self.num_classes)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(β)
        β /= max(β.max(), len(targets) - β.sum())
        β /= 2 - β
        masks = (max_probs > self.threshold * β[targets]).float()

        loss = torch.nn.functional.cross_entropy(
            logits_s, targets, reduction='none') * masks
        self.𝜇ₘₐₛₖ = masks.mean().detach()

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def change_bn(model, momentum):
    if isinstance(model, torch.nn.BatchNorm2d):
        model.momentum = 1 - momentum
    else:
        for children in model.children():
            change_bn(children, momentum)


def replace_relu(model):
    for child_name, child in model.named_children():
        if isinstance(child, torch.nn.ReLU):
            setattr(model, child_name, torch.nn.LeakyReLU(0.1, inplace=True))
        else:
            replace_relu(child)


class FixMatchClassifier(pl.LightningModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model = get_classifier(**self.hparams.model['backbone'])
        change_bn(self.model, self.hparams.model['momentum'])
        replace_relu(self.model)
        self.criterionₗ = torch.nn.CrossEntropyLoss()
        self.criterionᵤ = FixMatchCrossEntropy(**self.hparams.model['loss_u'])
        self.train_acc = Accuracy()
        self.valid_cur_acc = Accuracy()
        self.valid_ema_acc = Accuracy()

        def avg_fn(averaged_model_parameter, model_parameter, num_averaged):
            α = self.hparams.model['momentum']
            return α * averaged_model_parameter + (1 - α) * model_parameter
        self.ema = AveragedModelWithBuffers(self.model, avg_fn=avg_fn)

    def training_step(self, batch, batch_idx):
        xₗ, yₗ = batch['labeled']
        (ˢxᵤ, ʷxᵤ), _ = batch['unlabeled']

        z = self.model(torch.cat((xₗ, ˢxᵤ, ʷxᵤ)))
        zₗ = z[:xₗ.shape[0]]
        ˢzᵤ, ʷzᵤ = z[xₗ.shape[0]:].chunk(2)
        del z

        lossₗ = self.criterionₗ(zₗ, yₗ)
        lossᵤ = self.criterionᵤ(ˢzᵤ, ʷzᵤ.detach())
        loss = lossₗ + lossᵤ

        self.train_acc.update(zₗ.softmax(dim=1), yₗ)
        return {'loss': loss,
                'detail': {'loss_l': lossₗ.detach(),
                           'loss_u': lossᵤ.detach(),
                           'mask': self.criterionᵤ.𝜇ₘₐₛₖ}}

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        self.ema.update_parameters(self.model)

    def training_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('trn/loss', loss, sync_dist=True)
        loss = torch.stack([x['detail']['mask'] for x in outputs]).mean()
        self.log('detail/mask', loss, sync_dist=True)
        loss = torch.stack([x['detail']['loss_l'] for x in outputs]).mean()
        self.log('detail/loss_l', loss, sync_dist=True)
        loss = torch.stack([x['detail']['loss_u'] for x in outputs]).mean()
        self.log('detail/loss_u', loss, sync_dist=True)

        acc = self.train_acc.compute()
        self.log('trn/acc', acc, rank_zero_only=True)
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        cur_z = self.model(x)
        cur_loss = self.criterionₗ(cur_z, y)
        self.valid_cur_acc.update(cur_z.softmax(dim=1), y)
        ema_z = self.ema(x)
        ema_loss = self.criterionₗ(ema_z, y)
        self.valid_ema_acc.update(ema_z.softmax(dim=1), y)
        return {'loss/cur': cur_loss, 'loss/ema': ema_loss}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['loss/cur'] for x in outputs]).mean()
        self.log('val/loss/cur', loss, sync_dist=True)
        loss = torch.stack([x['loss/ema'] for x in outputs]).mean()
        self.log('val/loss/ema', loss, sync_dist=True)

        acc = self.valid_cur_acc.compute()
        self.log('val/acc/cur', acc, rank_zero_only=True)
        self.valid_cur_acc.reset()
        acc = self.valid_ema_acc.compute()
        self.log('val/acc/ema', acc, rank_zero_only=True)
        self.valid_ema_acc.reset()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    def configure_optimizers(self):
        params = exclude_wd(self.model)
        optim = get_optim(params, **self.hparams.optimizer)
        sched = get_sched(optim, **self.hparams.scheduler)
        return {'optimizer': optim, 'lr_scheduler': {'scheduler': sched}}


class FlexMatchClassifier(FixMatchClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.criterionᵤ = FlexMatchCrossEntropy(
            num_classes=self.hparams.model['backbone']['num_classes'],
            **self.hparams.model['loss_u']
        )
