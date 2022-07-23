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


def train(args):
    config = args.config
    seed_everything(config['random_seed'])

    trainer = Trainer.from_argparse_args(
        args,
        logger=TensorBoardLogger('logs', config['dataset']['name']),
        callbacks=[
            ModelCheckpoint(save_top_k=1, monitor='val/acc/ema', mode='max'),
            LearningRateMonitor(),
        ]
    )

    N = trainer.num_nodes * trainer.num_devices

    transform_w = get_xform('Compose', transforms=config['transform']['weak'])
    transform_s = get_xform('Compose', transforms=config['transform']['str'])
    transform_v = get_xform('Compose', transforms=config['transform']['val'])

    dm = DataModule[config['dataset']['name']](
        os.path.join('data', config['dataset']['name']),
        config['dataset']['num_labeled'],
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
        random_seed=config['dataset']['random_seed']
    )

    model = FixMatchClassifier(**config)
    # model = FlexMatchClassifier(**config)
    trainer.fit(model, dm)


def read_json(path):
    return json.load(open(path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=read_json)
    parser.add_argument('--data.num_labeled', type=int)
    parser.add_argument('--data.random_seed', type=int)
    parser.add_argument('--random_seed', type=int)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    args.config['dataset'].setdefault('num_labeled',
                                      getattr(args, 'data.num_labeled'))
    args.config['dataset'].setdefault('random_seed',
                                      getattr(args, 'data.random_seed'))
    args.config.setdefault('random_seed', args.random_seed)
    train(args)
