import torch
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy
from weaver import get_classifier, get_optimizer, get_scheduler
from weaver.optimizers import exclude_wd, EMAModel


def change_bn_momentum(model, momentum):
    if isinstance(model, torch.nn.BatchNorm2d):
        model.momentum = 1 - momentum
    for child in model.children():
        change_bn_momentum(child, momentum)


def replace_relu_to_lrelu(model, a=0.1):
    for name, child in model.named_children():
        if isinstance(child, torch.nn.ReLU):
            setattr(model, name, torch.nn.LeakyReLU(a, inplace=True))
    for child in model.children():
        replace_relu_to_lrelu(child, a)


class BaseModule(pl.LightningModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model = get_classifier(**self.hparams.model)
        if a := self.hparams.get('lrelu'):
            replace_relu_to_lrelu(self.model, a)

        if m := self.hparams.get('ema'):
            change_bn_momentum(self.model, m)
            self.ema = EMAModel(self.model, m)

        self.val_criterion = torch.nn.CrossEntropyLoss()
        self.val_accuracy = MulticlassAccuracy(
            self.hparams['dataset']['num_classes'])

    def forward(self, x):
        if self.hparams.model.get('ema'):
            return self.ema(x)
        return self.model(x)

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.hparams.model.get('ema'):
            self.ema.update_parameters(self.model)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self(x)
        loss = self.val_criterion(z, y)
        self.val_accuracy.update(z, y)
        self.log('val/loss', loss, on_step=False, on_epoch=True, sync_dist=True)

    def validation_epoch_end(self, outputs):
        self.log('val/acc', self.val_accuracy)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    def configure_optimizers(self):
        params = exclude_wd(self.model)
        optim = get_optimizer(params, **self.hparams.optimizer)
        sched = get_scheduler(optim, **self.hparams.scheduler)
        return {'optimizer': optim, 'lr_scheduler': {'scheduler': sched}}
