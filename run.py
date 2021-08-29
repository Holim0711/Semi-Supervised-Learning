import os
import sys
import json
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from models import *
from datasets import select_datasets
from holim_lightning.transforms import get_trfms, NqTwinTransform


def train(hparams, tparams):
    transform_w = get_trfms(hparams['transform']['weak'])
    transform_s = get_trfms(hparams['transform']['strong'])
    transform_v = get_trfms(hparams['transform']['valid'])

    dm = select_datasets(**hparams['dataset'])
    dm.train_transformₗ = transform_w
    dm.train_transformᵤ = NqTwinTransform(transform_s, transform_w)
    dm.valid_transform = transform_v

    model = FixMatchClassifier(**hparams)

    logger = TensorBoardLogger("logs", hparams['name'])

    callbacks = [
        ModelCheckpoint(monitor='val/acc'),
        LearningRateMonitor(logging_interval='step'),
    ]

    trainer = pl.Trainer(logger=logger, callbacks=callbacks, **tparams)
    trainer.fit(model, dm)


def read_hparams():
    with open(sys.argv[2]) as file:
        hparams = json.load(file)
    hparams['name'] = os.path.splitext(os.path.split(sys.argv[2])[1])[0]
    hparams['script'] = ' '.join(sys.argv)
    return hparams


if __name__ == "__main__":
    if sys.argv[1] == 'train':
        train(read_hparams())
    elif sys.argv[1] == 'finetune':
        train(read_hparams(), ckpt_path=sys.argv[3])
