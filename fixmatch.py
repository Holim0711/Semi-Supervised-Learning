from math import cos, pi
import torch
import pytorch_lightning as pl
from torchmetrics import Accuracy
from holim_lightning.models import get_model
from holim_lightning.optimizers import get_optim, exclude_wd
from holim_lightning.schedulers import get_lr_dict


class AveragedModelWithBuffers(torch.optim.swa_utils.AveragedModel):
    def update_parameters(self, model):
        super().update_parameters(model)
        for a, b in zip(self.module.buffers(), model.buffers()):
            a.copy_(b.to(a.device))


class FixMatchCrossEntropy(torch.nn.Module):
    def __init__(self, temperature, threshold, reduction='mean'):
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

        self.model = get_model(**self.hparams.model['backbone'])
        change_bn(self.model, self.hparams.model['momentum'])
        replace_relu(self.model)
        self.criterionₗ = torch.nn.CrossEntropyLoss()
        self.criterionᵤ = FixMatchCrossEntropy(**self.hparams.model['loss_u'])
        self.train_acc = Accuracy()
        self.valid_acc = Accuracy()

        def avg_fn(averaged_model_parameter, model_parameter, num_averaged):
            α = self.hparams.model['momentum']
            return α * averaged_model_parameter + (1 - α) * model_parameter
        self.ema = AveragedModelWithBuffers(self.model, avg_fn=avg_fn)

    def training_step(self, batch, batch_idx):
        xₗ, yₗ = batch['clean']
        (ˢxᵤ, ʷxᵤ), _ = batch['noisy']

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
        z = self.ema(x)
        loss = self.criterionₗ(z, y)
        self.valid_acc.update(z.softmax(dim=1), y)
        return {'loss': loss}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('val/loss', loss, sync_dist=True)

        acc = self.valid_acc.compute()
        self.log('val/acc', acc, rank_zero_only=True)
        self.valid_acc.reset()

    def configure_optimizers(self):
        params = exclude_wd(self.model)
        optim = get_optim(params, **self.hparams.optimizer)
        if self.hparams.lr_dict['scheduler']['name'] == 'Cosine716':
            T = self.hparams.lr_dict['scheduler']['max_steps']
            sched = {'scheduler': torch.optim.lr_scheduler.LambdaLR(
                        optim, lambda t: cos((7 / 16) * (t / T) * pi)),
                     'interval': 'step'}
        else:
            sched = get_lr_dict(optim, **self.hparams.lr_dict)
        return {'optimizer': optim, 'lr_scheduler': sched}
