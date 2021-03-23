import torch
import pytorch_lightning as pl
from holim_lightning.models import get_model
from holim_lightning.optimizers import get_optim
from holim_lightning.schedulers import get_sched


class EMAModel(torch.optim.swa_utils.AveragedModel):

    def __init__(self, model, decay=0.9999, device=None):
        self.decay = decay
        def ema_fn(p_swa, p_model, n):
            return self.decay * p_swa + (1. - self.decay) * p_model
        super().__init__(model, device, ema_fn)

    def update_parameters(self, model):
        super().update_parameters(model)
        for b_swa, b_model in zip(self.module.buffers(), model.buffers()):
            device = b_swa.device
            b_model_ = b_model.detach().to(device)
            if self.n_averaged == 0:
                b_swa.detach().copy_(b_model_)
            else:
                b_swa.detach().copy_(self.avg_fn(b_swa.detach(), b_model_,
                                                 self.n_averaged.to(device)))


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
        self.𝜇ₘₐₛₖ = masks.mean().item()

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class FixMatchClassifier(pl.LightningModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model = get_model(
            self.hparams.model['backbone'],
            self.hparams.model['num_classes'],
            pretrained=self.hparams.model['pretrained'])
        self.ema = EMAModel(self.model, self.hparams.model['EMA']['decay'])
        self.CE = torch.nn.CrossEntropyLoss()
        self.FM_CE = FixMatchCrossEntropy(
            temperature=self.hparams.model['fixmatch']['temperature'],
            threshold=self.hparams.model['fixmatch']['threshold'])
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.ema_valid_acc = pl.metrics.Accuracy()

    def forward(self, x):
        return self.ema(x).softmax(dim=1)

    def training_step(self, batch, batch_idx):
        xₗ, yₗ = batch['labeled']
        (xᵤ, ʳxᵤ), _ = batch['unlabeled']

        ᵗz = self.model(torch.cat((xₗ, xᵤ, ʳxᵤ)))
        ᵗzₗ = ᵗz[:xₗ.shape[0]]
        ᵗzᵤ, ʳzᵤ = ᵗz[xₗ.shape[0]:].chunk(2)
        del ᵗz
 
        lossᵤ = self.FM_CE(ʳzᵤ, ᵗzᵤ.clone().detach())

        lossₗ = self.CE(ᵗzₗ, yₗ)
        self.train_acc.update(ᵗzₗ.softmax(dim=1), yₗ)

        self.log_dict({
            'detail/mask': self.FM_CE.𝜇ₘₐₛₖ,
            'step': self.global_step,
        })
        return {
            'loss': lossₗ + lossᵤ * self.hparams.model['fixmatch']['factor'],
            'loss_l': lossₗ,
            'loss_u': lossᵤ,
        }

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, *args, **kwargs):
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, *args, **kwargs)
        self.ema.update_parameters(self.model)

    def training_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        acc = self.train_acc.compute()
        lossₗ = torch.stack([x['loss_l'] for x in outputs]).mean()
        lossᵤ = torch.stack([x['loss_u'] for x in outputs]).mean()
        self.log_dict({
            'train/loss': loss,
            'train/acc': acc,
            'detail/loss_l': lossₗ,
            'detail/loss_u': lossᵤ,
            'step': self.current_epoch,
        })
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.model(x)
        loss = self.CE(z, y)
        self.valid_acc.update(z.softmax(dim=1), y)
        zₑₘₐ = self.ema(x)
        lossₑₘₐ = self.CE(zₑₘₐ, y)
        self.ema_valid_acc.update(zₑₘₐ.softmax(dim=1), y)
        return {'loss': loss, 'loss_ema': lossₑₘₐ}

    def validation_epoch_end(self, outputs):
        loss_raw = torch.stack([x['loss'] for x in outputs]).mean()
        acc_raw = self.valid_acc.compute()
        loss_ema = torch.stack([x['loss_ema'] for x in outputs]).mean()
        acc_ema = self.ema_valid_acc.compute()
        self.log_dict({
            'val/raw/loss': loss_raw,
            'val/raw/acc': acc_raw,
            'val/ema/loss': loss_ema,
            'val/ema/acc': acc_ema,
            'step': self.current_epoch,
        })
        self.valid_acc.reset()
        self.ema_valid_acc.reset()

    def configure_optimizers(self):
        no_decay = ['bias', 'bn']
        parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(
                nd in n for nd in no_decay)],
             'weight_decay': self.hparams.optim['optimizer']['weight_decay']},
            {'params': [p for n, p in self.model.named_parameters() if any(
                nd in n for nd in no_decay)],
             'weight_decay': 0.0},
        ]
        optim = get_optim(parameters, **self.hparams.optim['optimizer'])
        sched = get_sched(optim, **self.hparams.optim['scheduler'])
        return {
            'optimizer': optim,
            'lr_scheduler': {
                'name': "lr",
                'scheduler': sched,
                'interval': self.hparams.optim['interval'],
            },
        }
