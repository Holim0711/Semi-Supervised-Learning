# Semi-Supervised Learning Algorithms

## Dependencies
- PyTorch
- PyTorch-Lightning
- Weaver (`pip install weaver-pytorch-rnx0dvmdxk`)

## List
- FixMatch
- FlexMatch

## Training

### CIFAR10
```
python train.py configs/fixmatch/cifar10.json --accelerator gpu --devices 1 --max_epoch 9362 --random_seed 0 --dataset.num_labeled 100
python train.py configs/flexmatch/cifar10.json --accelerator gpu --devices 1 --max_epoch 9362 --random_seed 0 --dataset.num_labeled 100
```

### CIFAR100
```
python train.py configs/fixmatch/cifar100.json --accelerator gpu --devices 4 --strategy ddp --sync_batchnorm --max_epoch 9362 --random_seed 0 --dataset.num_labeled 1000
python train.py configs/flexmatch/cifar100.json --accelerator gpu --devices 4 --strategy ddp --sync_batchnorm --max_epoch 9362 --random_seed 0 --dataset.num_labeled 1000
```

## Results

### CIFAR10
|           | 40           | 100          | 250          |
| :---:     | :---:        | :---:        | :---:        |
| FixMatch  | -            | -            | -            |
| FlexMatch | -            | -            | -            |

### CIFAR100
|           | 400          | 1000         | 2500         |
| :---:     | :---:        | :---:        | :---:        |
| FixMatch  | -            | -            | -            |
| FlexMatch | -            | -            | -            |
