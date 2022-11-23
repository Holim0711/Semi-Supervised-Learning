import os
import json
import argparse

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning_lite import seed_everything
from torchvision.transforms import Compose, Lambda
from weaver import get_transforms

from dataset import (
    SemiCIFAR10,
    SemiCIFAR100,
)
from methods import (
    FixMatchClassifier,
    FlexMatchClassifier,
)


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

    transform_w = Compose(get_transforms(config['transform']['weak']))
    transform_s = Compose(get_transforms(config['transform']['strong']))
    transform_v = Compose(get_transforms(config['transform']['val']))

    Dataset = {
        'CIFAR10': SemiCIFAR10,
        'CIFAR100': SemiCIFAR100,
    }[config['dataset']['name']]

    dm = Dataset(
        os.path.join('data', config['dataset']['name']),
        config['dataset']['num_labeled'],
        transforms={
            'labeled': transform_w,
            'unlabeled': Lambda(lambda x: (transform_s(x), transform_w(x))),
            'val': transform_v
        },
        batch_sizes={
            'labeled': config['dataset']['batch_sizes']['labeled'] // N,
            'unlabeled': config['dataset']['batch_sizes']['unlabeled'] // N,
            'val': config['dataset']['batch_sizes']['val'],
        },
        random_seed=config['dataset']['random_seed'],
        expand_labeled=True,
        enumerate_unlabeled=True,
    )

    if config['method'] == 'fixmatch':
        model = FixMatchClassifier(**config)
    elif config['method'] == 'flexmatch':
        model = FlexMatchClassifier(**config)

    trainer.fit(model, dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=lambda x: json.load(open(x)))
    parser.add_argument('--dataset.num_labeled', type=int)
    parser.add_argument('--dataset.random_seed', type=int)
    parser.add_argument('--random_seed', type=int)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    if (v := getattr(args, 'dataset.num_labeled')) is not None:
        args.config['dataset']['num_labeled'] = v
    if (v := getattr(args, 'dataset.random_seed')) is not None:
        args.config['dataset']['random_seed'] = v
    if (v := getattr(args, 'random_seed')) is not None:
        args.config['random_seed'] = v
    train(args)
