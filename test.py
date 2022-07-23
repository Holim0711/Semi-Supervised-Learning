import os
import json
import argparse

from pytorch_lightning import Trainer

from fixmatch import *
from weaver.transforms import get_xform
from dataset import SemiCIFAR10, SemiCIFAR100

DataModule = {
    'cifar10': SemiCIFAR10,
    'cifar100': SemiCIFAR100,
}


def test(args):
    config = args.config
    trainer = Trainer.from_argparse_args(args, logger=False)
    dm = DataModule[config['dataset']['name']](
        os.path.join('data', config['dataset']['name']),
        n=0,
        transforms={'val': get_xform(config['transform']['val'])},
        batch_sizes={'val': config['dataset']['batch_sizes']['val']},
    )
    model = FixMatchClassifier.load_from_checkpoint(args.ckpt_path)
    trainer.test(model, dm)


def read_json(path):
    return json.load(open(path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=read_json)
    parser.add_argument('ckpt_path', type=str)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    test(args)
