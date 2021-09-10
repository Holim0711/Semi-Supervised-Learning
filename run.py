import os
import json
import argparse

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.utilities.seed import seed_everything
from torchvision.datasets.cifar import CIFAR10
from torch.utils.data import DataLoader

from fixmatch import FixMatchClassifier
from holim_lightning.transforms import get_trfms, NqTwinTransform
from noisy_cifar import NoisyCIFAR10

DataModule = {
    'cifar10': NoisyCIFAR10,
}


def train(hparams, dparams, tparams):
    logger = TensorBoardLogger("lightning_logs", str(dparams.num_clean))
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


def test(hparams, tparams, ckpt_path):
    trainer = Trainer.from_argparse_args(tparams)
    dataset = CIFAR10(root=os.path.join('data', hparams['dataset']['name']),
                      train=False,
                      transform=get_trfms(hparams['transform']['valid']))
    dataloader = DataLoader(dataset, hparams['dataset']['batch_size']['valid'],
                            num_workers=os.cpu_count(), pin_memory=True)
    pl_module = FixMatchClassifier.load_from_checkpoint(ckpt_path)
    trainer.test(pl_module, dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'test'])
    parser.add_argument('config', type=str)
    args, _ = parser.parse_known_args()

    with open(args.config) as file:
        hparams = json.load(file)

    if args.mode == 'train':
        parser.add_argument('--random_seed', type=int)
        parser = DataModule[hparams['dataset']['name']].add_argparse_args(parser)
    else:
        parser.add_argument('--ckpt_path', type=str, required=True)

    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    for g in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in g._group_actions}
        if g.title in {'NoisyCIFAR10'}:
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
        test(hparams, tparams, args.ckpt_path)
