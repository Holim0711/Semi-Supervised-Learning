import os
import json
import argparse

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.utilities.seed import seed_everything

from fixmatch import *
from weaver.transforms import get_xform
from weaver.transforms.twin_transforms import NqTwinTransform
from dataset import SemiCIFAR10, SemiCIFAR100

DataModule = {
    'cifar10': SemiCIFAR10,
    'cifar100': SemiCIFAR100,
}


def train(config, args):
    config['dataset']['n'] = args.n
    config['dataset']['random_seed'] = args.random_seed
    seed_everything(args.random_seed)

    trainer = Trainer.from_argparse_args(
        args,
        logger=TensorBoardLogger('logs', config['dataset']['name']),
        callbacks=[
            ModelCheckpoint(save_top_k=1, monitor='val/acc', mode='max'),
            LearningRateMonitor(),
        ]
    )

    N = trainer.num_nodes * trainer.num_devices

    transform_w = get_xform('Compose', transforms=config['transform']['weak'])
    transform_s = get_xform('Compose', transforms=config['transform']['str'])
    transform_v = get_xform('Compose', transforms=config['transform']['val'])

    dm = DataModule[config['dataset']['name']](
        os.path.join('data', config['dataset']['name']),
        args.n,
        transforms={
            'labeled': transform_w,
            'unlabeled': NqTwinTransform(transform_s, transform_w),
            'val': transform_v
        },
        batch_sizes={
            'labeled': config['dataset']['batch_sizes']['labeled'] // N,
            'unlabeled': config['dataset']['batch_sizes']['unlabeled'] // N,
            'val': config['dataset']['batch_sizes']['val'],
        },
        random_seed=args.random_seed
    )

    model = FixMatchClassifier(**config)
    # model = FlexMatchClassifier(**config)
    trainer.fit(model, dm)


def test(config, args):
    trainer = Trainer.from_argparse_args(args, logger=False)
    dm = DataModule[config['dataset']['name']](
        os.path.join('data', config['dataset']['name']),
        n=0,
        transforms={'val': get_xform(config['transform']['val'])},
        batch_sizes={'val': config['dataset']['batch_sizes']['val']},
    )
    model = FixMatchClassifier.load_from_checkpoint(args.ckpt_path)
    trainer.test(model, dm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'test'])
    parser.add_argument('config', type=str)
    parser.add_argument('--n', type=int)
    parser.add_argument('--random_seed', type=int, default=1234)
    parser.add_argument('--ckpt_path', type=str)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    with open(args.config) as file:
        config = json.load(file)

    if args.mode == 'train':
        assert args.n > 0
        assert args.random_seed
        train(config, args)
    else:
        assert args.ckpt_path
        test(config, args)


if __name__ == "__main__":
    main()
