import os
import warnings
from math import ceil
from typing import Callable, Optional
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, Subset, ConcatDataset, DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import CIFAR10, CIFAR100


def random_select(y, N, seed=None):
    """ Select a total of 'N' indices equally for each class """
    y = np.array(y)
    C = np.unique(y)
    n = N // len(C)
    random_state = np.random.RandomState(seed)

    random_I = []
    for c in C:
        Iᶜ = np.where(y == c)[0]
        random_Iᶜ = random_state.choice(Iᶜ, n, replace=False)
        random_I.extend(random_Iᶜ)
    return random_I


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return index, self.dataset[index]


class UnlabeledDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, _ = self.dataset[index]
        return x


class SemiDataModule(LightningDataModule):

    Dataset = None

    def __init__(
        self,
        root: str,
        num_labeled: int,
        random_seed: int = 0,
        transforms: dict[str, Callable] = {},
        batch_sizes: dict[str, int] = {},
        expand_labeled: bool = False,
        enumerate_unlabeled: bool = False,
    ):
        super().__init__()
        self.root = root
        self.num_labeled = num_labeled
        self.random_seed = random_seed

        splits = ['labeled', 'unlabeled', 'val']
        self.transforms = {k: transforms.get(k, ToTensor()) for k in splits}
        self.batch_sizes = {k: batch_sizes.get(k, 1) for k in splits}

        for k in transforms:
            if k not in splits:
                warnings.warn(f"'{k}' in transforms is ignored")
        for k in batch_sizes:
            if k not in splits:
                warnings.warn(f"'{k}' in batch_sizes is ignored")

        self.expand_labeled = expand_labeled
        self.enumerate_unlabeled = enumerate_unlabeled

    def prepare_data(self):
        self.Dataset(self.root, download=True)

    def get_dataset(self, split: str, transform: Optional[Callable] = None):
        if split == 'labeled':
            d = self.Dataset(self.root, transform=transform)
            i = random_select(d.targets, self.num_labeled, self.random_seed)
            return Subset(d, i)
        elif split == 'unlabeled':
            return self.Dataset(self.root, transform=transform)
        elif split == 'val':
            return self.Dataset(self.root, train=False, transform=transform)

        raise ValueError(f'Unknwon dataset split: {split}')

    def setup(self, stage=None):
        P = self.get_dataset('labeled', self.transforms['labeled'])
        U = self.get_dataset('unlabeled', self.transforms['unlabeled'])
        V = self.get_dataset('val', self.transforms['val'])

        if self.expand_labeled:
            m = ((len(U) * self.batch_sizes['labeled']) /
                 (len(P) * self.batch_sizes['unlabeled'] * 2))
            P = ConcatDataset([P] * max(ceil(m), 1))

        if self.enumerate_unlabeled:
            U = IndexedDataset(U)

        self._datasets = {'labeled': P, 'unlabeled': U, 'val': V}

    def get_dataloader(
        self,
        split: str,
        shuffle: bool = True,
        num_workers: int = os.cpu_count(),
        pin_memory: bool = True
    ):
        return DataLoader(
            self._datasets[split],
            self.batch_sizes[split],
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    def train_dataloader(self):
        return {'labeled': self.get_dataloader('labeled'),
                'unlabeled': self.get_dataloader('unlabeled')}

    def val_dataloader(self):
        return self.get_dataloader('val', shuffle=False)

    def test_dataloader(self):
        return self.val_dataloader()


class SemiCIFAR10(SemiDataModule):
    Dataset = CIFAR10


class SemiCIFAR100(SemiDataModule):
    Dataset = CIFAR100
