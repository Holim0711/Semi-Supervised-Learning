import os
import yaml
import argparse

from pytorch_lightning import Trainer
from pytorch_lightning.utilities.argparse import parse_env_variables
from torchvision.transforms import Compose
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader
from weaver import get_transforms

from methods import FixMatchModule, FlexMatchModule


def test(config, checkpoint):
    trainer = Trainer(**vars(parse_env_variables(Trainer)), logger=False)

    t = Compose(get_transforms(config['transform']['val']))

    if config['dataset']['name'] == 'CIFAR10':
        dataset = CIFAR10(config['dataset']['root'], train=False, transform=t)
    elif config['dataset']['name'] == 'CIFAR100':
        dataset = CIFAR100(config['dataset']['root'], train=False, transform=t)

    dataloader = DataLoader(
        dataset, config['batch_size']['val'],
        num_workers=os.cpu_count(), pin_memory=True)

    if config['method']['name'] == 'FixMatch':
        model = FixMatchModule.load_from_checkpoint(checkpoint)
    elif config['method']['name'] == 'FlexMatch':
        model = FlexMatchModule.load_from_checkpoint(checkpoint)

    trainer.test(model, dataloader)


def read_yaml(path):
    return yaml.load(open(path), Loader=yaml.FullLoader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('hparams', type=read_yaml)
    parser.add_argument('checkpoint', type=str)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    test(args.hparams, args.checkpoint)
