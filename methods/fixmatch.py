import torch
from torchmetrics.classification import MulticlassAccuracy
from .base import BaseModule

__all__ = ['FixMatchModule']


class FixMatchCrossEntropy(torch.nn.Module):
    def __init__(self, temperature, threshold):
        super().__init__()
        self.temperature = temperature
        self.threshold = threshold
        self.𝜇 = 0.0

    def forward(self, logits_s, logits_w):
        c, ŷ = (logits_w / self.temperature).softmax(dim=-1).max(dim=-1)
        mask = (c > self.threshold)
        self.𝜇 = mask.float().mean()
        loss = torch.nn.functional.cross_entropy(logits_s, ŷ, reduction='none')
        return (loss * mask).mean()


class FixMatchModule(BaseModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.criterionˡ = torch.nn.CrossEntropyLoss()
        self.criterionᵘ = FixMatchCrossEntropy(
            self.hparams.method['temperature'],
            self.hparams.method['threshold'])
        self.train_accuracy = MulticlassAccuracy(
            self.hparams['dataset']['num_classes'])

    def training_step(self, batch, batch_idx):
        iˡ, (xˡ, yˡ) = batch['labeled']
        iᵘ, ((uʷ, uˢ), _) = batch['unlabeled']
        bˡ, bᵘ = len(iˡ), len(iᵘ)

        z = self.model(torch.cat((xˡ, uʷ, uˢ)))
        zˡ, zʷ, zˢ = z.split([bˡ, bᵘ, bᵘ])

        lossˡ = self.criterionˡ(zˡ, yˡ)
        lossᵘ = self.criterionᵘ(zˢ, zʷ.detach())
        loss = lossˡ + lossᵘ

        self.log('train/loss', loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=bˡ + bᵘ)
        self.log('train/loss_l', lossˡ, on_step=False, on_epoch=True, sync_dist=True, batch_size=bˡ)
        self.log('train/loss_u', lossᵘ, on_step=False, on_epoch=True, sync_dist=True, batch_size=bᵘ)
        self.log('train/mask', self.criterionᵘ.𝜇, on_step=False, on_epoch=True, sync_dist=True, batch_size=bᵘ)
        self.train_accuracy.update(zˡ, yˡ)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        self.log('train/acc', self.train_accuracy)
