# FixMatch-Lightning

## Dependencies
- PyTorch
- PyTorch-Lightning
- Weaver (https://github.com/Holim0711/Weaver)

## Training

### CIFAR10
```
python train.py configs/fixmatch/cifar10.json --gpus 1 --max_epoch 9362 --random_seed 0 --deterministic
python train.py configs/fixmatch/cifar10.json --gpus 1 --max_epoch 9362 --random_seed 0 --deterministic --dataset.num_labeled 100
python train.py configs/flexmatch/cifar10.json --gpus 1 --max_epoch 9362 --random_seed 0 --deterministic
python train.py configs/flexmatch/cifar10.json --gpus 1 --max_epoch 9362 --random_seed 0 --deterministic --dataset.num_labeled 100
```

### CIFAR100
```
python train.py configs/fixmatch/cifar100.json --gpus 4 --strategy ddp --sync_batchnorm --max_epoch 9362 --random_seed 0
python train.py configs/fixmatch/cifar100.json --gpus 4 --strategy ddp --sync_batchnorm --max_epoch 9362 --random_seed 0 --dataset.num_labeled 1000
python train.py configs/flexmatch/cifar100.json --gpus 4 --strategy ddp --sync_batchnorm --max_epoch 9362 --random_seed 0
python train.py configs/flexmatch/cifar100.json --gpus 4 --strategy ddp --sync_batchnorm --max_epoch 9362 --random_seed 0 --dataset.num_labeled 1000
```

## Results

### CIFAR10
|           |   40  |  100  |
|   :---:   | :---: | :---: |
|  FixMatch |  WIP  |  WIP  |
| FlexMatch |  WIP  |  WIP  |

### CIFAR100
|           |  400  |  1000 |
|   :---:   | :---: | :---: |
|  FixMatch |  WIP  |  WIP  |
| FlexMatch |  WIP  |  WIP  |

## References
- [official - FixMatch](https://github.com/google-research/fixmatch)
- [kekmodel - FixMatch](https://github.com/kekmodel/FixMatch-pytorch)
