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
|                   | 40           | 250          |
| :---:             | :---:        | :---:        |
| FixMatch (paper)  | 86.19 ± 3.37 | 94.93 ± 0.65 |
| FixMatch (this)   | 93.84        | -            |
| FlexMatch (paper) | 95.03 ± 0.06 | 95.02 ± 0.09 |
| FlexMatch (this)  | 95.36        | -            |

### CIFAR100
|                   | 400          | 2500         |
| :---:             | :---:        | :---:        |
| FixMatch (paper)  | 51.15 ± 1.75 | 71.71 ± 0.11 |
| FixMatch (this)   | -            | -            |
| FlexMatch (paper) | 60.06 ± 1.62 | 73.51 ± 0.20 |
| FlexMatch (this)  | 58.51 (↑)    | -            |
