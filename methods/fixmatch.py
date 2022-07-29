import torch
import pytorch_lightning as pl
from torchmetrics import Accuracy
from weaver.models import get_classifier
from weaver.optimizers import get_optim
from weaver.optimizers.utils import exclude_wd
from weaver.schedulers import get_sched
from .utils import EMA, change_bn_momentum, replace_relu_to_lrelu

__all__ = ['FixMatchClassifier']


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


class FixMatchClassifier(pl.LightningModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model = get_classifier(**self.hparams.model['backbone'])
        if a := self.hparams.model.get('lrelu'):
            replace_relu_to_lrelu(self.model, a)
        self.criterionₗ = torch.nn.CrossEntropyLoss()
        self.criterionᵤ = FixMatchCrossEntropy(**self.hparams.model['loss_u'])
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

        if m := self.hparams.model.get('ema'):
            change_bn_momentum(self.model, m)
            self.ema = EMA(self.model, m)
            self.val_acc_ema = Accuracy()

    def training_step(self, batch, batch_idx):
        xₗ, yₗ = batch['labeled']
        iᵤ, ((ˢxᵤ, ʷxᵤ), _) = batch['unlabeled']

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
        self.log('train/loss', loss, sync_dist=True)
        acc = self.train_acc.compute()
        self.log('train/acc', acc, rank_zero_only=True)
        self.train_acc.reset()

        loss = torch.stack([x['detail']['mask'] for x in outputs]).mean()
        self.log('detail/mask', loss, sync_dist=True)
        loss = torch.stack([x['detail']['loss_l'] for x in outputs]).mean()
        self.log('detail/loss_l', loss, sync_dist=True)
        loss = torch.stack([x['detail']['loss_u'] for x in outputs]).mean()
        self.log('detail/loss_u', loss, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.model(x)
        loss = self.criterionₗ(z, y)
        self.val_acc.update(z.softmax(dim=1), y)
        results = {'loss': loss}

        if self.hparams.model.get('ema'):
            z = self.ema(x)
            loss = self.criterionₗ(z, y)
            self.val_acc_ema.update(z.softmax(dim=1), y)
            results['loss/ema'] = loss

        return results

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('val/loss', loss, sync_dist=True)
        acc = self.val_acc.compute()
        self.log('val/acc', acc, rank_zero_only=True)
        self.val_acc.reset()

        if self.hparams.model.get('ema'):
            loss = torch.stack([x['loss/ema'] for x in outputs]).mean()
            self.log('val/loss/ema', loss, sync_dist=True)
            acc = self.val_acc_ema.compute()
            self.log('val/acc/ema', acc, rank_zero_only=True)
            self.val_acc_ema.reset()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    def configure_optimizers(self):
        params = exclude_wd(self.model)
        optim = get_optim(params, **self.hparams.optimizer)
        sched = get_sched(optim, **self.hparams.scheduler)
        return {'optimizer': optim, 'lr_scheduler': {'scheduler': sched}}
