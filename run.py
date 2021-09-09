import os
import json
import argparse

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.utilities.seed import seed_everything

from fixmatch import FixMatchClassifier
from holim_lightning.transforms import get_trfms, NqTwinTransform
from noisy_cifar import NoisyCIFAR10, NoisyCIFAR100

DataModule = {
    'cifar10': NoisyCIFAR10,
    'cifar100': NoisyCIFAR100,
}


def train(hparams, dparams, tparams):
    logger = TensorBoardLogger(
        f"lightning_logs/{hparams['dataset']['name']}",
        str(dparams.num_clean)
    )
    callbacks = [
        ModelCheckpoint(save_top_k=1, monitor='val/acc', mode='max'),
        LearningRateMonitor(),
    ]
    trainer = Trainer.from_argparse_args(tparams, logger=logger, callbacks=callbacks)

    num_devices = trainer.num_nodes * max(trainer.num_processes, trainer.num_gpus, trainer.tpu_cores or 0)

    transform_w = get_trfms(hparams['transform']['weak'])
    transform_s = get_trfms(hparams['transform']['strong'])
    transform_v = get_trfms(hparams['transform']['valid'])

    dm = DataModule[hparams['dataset']['name']].from_argparse_args(
        os.path.join('data', hparams['dataset']['name']),
        dparams,
        batch_size_clean=hparams['dataset']['batch_size']['clean'] // num_devices,
        batch_size_noisy=hparams['dataset']['batch_size']['noisy'] // num_devices,
        batch_size_valid=hparams['dataset']['batch_size']['valid'] // num_devices,
        transform_clean=transform_w,
        transform_noisy=NqTwinTransform(transform_s, transform_w),
        transform_valid=transform_v,
    )

    model = FixMatchClassifier(**hparams)
    trainer.fit(model, dm)


def test(hparams, dparams, tparams, ckpt_path):
    trainer = Trainer.from_argparse_args(tparams)
    dm = DataModule[hparams['dataset']['name']].from_argparse_args(
        os.path.join('data', hparams['dataset']['name']),
        dparams,
        batch_size_valid=hparams['dataset']['batch_size']['valid'],
        transform_valid=get_trfms(hparams['transform']['valid']),
    )
    model = FixMatchClassifier.load_from_checkpoint(ckpt_path)
    trainer.test(model, dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'test'])
    parser.add_argument('config', type=str)
    args, _ = parser.parse_known_args()

    with open(args.config) as file:
        hparams = json.load(file)

    if args.mode == 'train':
        parser.add_argument('--random_seed', type=int)
    else:
        parser.add_argument('--ckpt_path', type=str, required=True)

    parser = DataModule[hparams['dataset']['name']].add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    for g in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in g._group_actions}
        if g.title in {'NoisyCIFAR10', 'NoisyCIFAR100'}:
            dparams = argparse.Namespace(**group_dict)
        elif g.title == 'pl.Trainer':
            tparams = argparse.Namespace(**group_dict)

    if args.random_seed:
        hparams['random_seed'] = args.random_seed

    if hparams.get('random_seed') is not None:
        seed_everything(hparams['random_seed'])

    if args.mode == 'train':
        train(hparams, dparams, tparams)
    else:
        test(hparams, dparams, tparams, args.ckpt_path)
