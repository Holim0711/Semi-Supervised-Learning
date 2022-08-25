import os
import json
import argparse

from pytorch_lightning import Trainer

from weaver.transforms import get_xform

from dataset import (
    SemiCIFAR10,
    SemiCIFAR100,
)
from methods import (
    FixMatchClassifier,
    FlexMatchClassifier,
)


def test(args):
    config = args.config

    trainer = Trainer.from_argparse_args(args, logger=False)

    transform_v = get_xform('Compose', transforms=config['transform']['val'])

    Dataset = {
        'CIFAR10': SemiCIFAR10,
        'CIFAR100': SemiCIFAR100,
    }[config['dataset']['name']]

    dm = Dataset(
        os.path.join('data', config['dataset']['name']),
        config['dataset']['num_labeled'],
        transforms={
            'val': transform_v
        },
        batch_sizes={
            'val': config['dataset']['batch_sizes']['val'],
        },
        random_seed=config['dataset']['random_seed'],
    )

    if config['method'] == 'fixmatch':
        model = FixMatchClassifier.load_from_checkpoint(args.checkpoint)
    elif config['method'] == 'flexmatch':
        model = FlexMatchClassifier.load_from_checkpoint(args.checkpoint)

    trainer.test(model, dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=lambda x: json.load(open(x)))
    parser.add_argument('checkpoint', type=str)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    test(args)
