# Semi-Supervised Learning Algorithms

## Dependencies
- PyTorch
- PyTorch-Lightning
- Weaver (https://github.com/Holim0711/Weaver)

## List
- FixMatch
- FlexMatch

## Training

### CIFAR10
```
python train.py configs/fixmatch/cifar10.json --gpus 1 --max_epoch 9362 --random_seed 0 --deterministic
python train.py configs/flexmatch/cifar10.json --gpus 1 --max_epoch 9362 --random_seed 0 --deterministic
```

### CIFAR100
```
python train.py configs/fixmatch/cifar100.json --gpus 4 --strategy ddp --sync_batchnorm --max_epoch 9362 --random_seed 0 --deterministic
python train.py configs/flexmatch/cifar100.json --gpus 4 --strategy ddp --sync_batchnorm --max_epoch 9362 --random_seed 0 --deterministic
```

## Results

### CIFAR10
|           | Ours  |    Paper     |
|   :---:   | :---: |    :---:     |
|  FixMatch |  WIP  | 86.19 ± 3.37 |
| FlexMatch |  WIP  | 95.03 ± 0.06 |

### CIFAR100
|           | Ours  |    Paper     |
|   :---:   | :---: |    :---:     |
|  FixMatch |  WIP  | 51.15 ± 1.75 |
| FlexMatch | 58.48 | 60.06 ± 1.62 |
