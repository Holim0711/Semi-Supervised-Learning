import json
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.utilities.seed import seed_everything

from models import *
from datasets import select_datasets
from holim_lightning.transforms import get_trfms, NqTwinTransform


def train(hparams, tparams, ckpt_path=None):
    transform_w = get_trfms(hparams['transform']['weak'])
    transform_s = get_trfms(hparams['transform']['strong'])
    transform_v = get_trfms(hparams['transform']['valid'])

    dm = select_datasets(
        transform={'clean': transform_w,
                   'noisy': NqTwinTransform(transform_s, transform_w),
                   'valid': transform_v},
        **hparams['dataset'])

    model = FixMatchClassifier(**hparams)

    logger = TensorBoardLogger(
        "lightning_logs", str(hparams['dataset']['num_clean']))

    callbacks = [
        ModelCheckpoint(monitor='val/acc'),
        LearningRateMonitor(),
    ]

    trainer = pl.Trainer(logger=logger, callbacks=callbacks, **tparams)
    trainer.fit(model, dm)


def test(hparams, tparams, ckpt_path=None):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'test'])
    parser.add_argument('--ckpt_path', type=str)

    group = parser.add_argument_group('configs')
    group.add_argument('config', type=str)
    group.add_argument('--random_seed', type=int)
    group.add_argument('--num_clean', type=int)

    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    for g in parser._action_groups:
        group_dict = {act.dest: getattr(args, act.dest, None)
                      for act in g._group_actions}
        if g.title == 'configs':
            configs = argparse.Namespace(**group_dict)
        elif g.title == 'pl.Trainer':
            tparams = group_dict

    with open(configs.config) as file:
        hparams = json.load(file)
    if configs.random_seed:
        hparams['random_seed'] = configs.random_seed
    if configs.num_clean:
        hparams['dataset']['num_clean'] = configs.num_clean
    if hparams.get('random_seed') is not None:
        seed_everything(hparams['random_seed'])

    del tparams['logger']
    train_limits = ['max_steps', 'max_epochs']
    if all(tparams[k] is None for k in train_limits):
        for k in train_limits:
            tparams[k] = hparams['lr_dict']['scheduler'].get(k)

    if args.mode == 'train':
        train(hparams, tparams, args.ckpt_path)
    else:
        test(hparams, tparams, args.ckpt_path)
