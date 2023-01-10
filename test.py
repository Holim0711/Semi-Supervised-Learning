import os
import sys
import re
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.argparse import parse_env_variables
from torchvision.transforms import Compose
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader
from weaver import get_transforms

from methods import FixMatchModule, FlexMatchModule, FlexDashhModule


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
    elif config['method']['name'] == 'FlexDash':
        model = FlexDashhModule.load_from_checkpoint(checkpoint)

    trainer.test(model, dataloader)


if __name__ == "__main__":
    v = int(sys.argv[1])
    hparams = os.path.join('lightning_logs', f'version_{v}', 'hparams.yaml')
    hparams = yaml.load(open(hparams), Loader=yaml.FullLoader)

    ckptdir = os.path.join('lightning_logs', f'version_{v}', 'checkpoints')
    checkpoints = os.listdir(ckptdir)
    if len(checkpoints) == 0:
        raise Exception('there are no checkpoints')
    elif len(checkpoints) == 1:
        checkpoint = os.path.join(ckptdir, checkpoints[0])
    elif len(checkpoints) > 1:
        checkpoints.sort(key=lambda s: [int(t) if t.isdigit() else t
                                        for t in re.split(r'(\d+)', s)])
        ckpt = checkpoints[int(sys.argv[2])]
        print(f'loading {ckpt}..', file=sys.stderr)
        checkpoint = os.path.join(ckptdir, ckpt)

    test(hparams, checkpoint)
