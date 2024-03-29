# Semi-Supervised Learning Algorithms

## Dependencies
- [PyTorch](https://pytorch.org/)
- [Lightning](https://www.pytorchlightning.ai/)
- [Hydra](https://hydra.cc/)
- [Weaver](https://github.com/Holim0711/Weaver)

## List of training algorithms
- FixMatch
- FlexMatch

## Setting Environment Variables
```bash
export PL_TRAINER_ACCELERATOR=gpu
export PL_TRAINER_DEVICES=1
export PL_TRAINER_DETERMINISTIC=True
export PL_TRAINER_MAX_EPOCHS=9362
```

## Training
```bash
python train.py --config-name={config name}
python train.py --config-name=fixmatch-cifar10
```

### Testing
```bash
python test.py {version number}
python test.py 0
```

## Results

### CIFAR10
|           | 40           | 100          | 250          |
| :---:     | :---:        | :---:        | :---:        |
| FixMatch  | 90.99        | -            | -            |
| FlexMatch | 95.43        | -            | -            |
| FlexDash  | 95.50        | -            | -            |

### CIFAR100
|           | 400          | 1000         | 2500         |
| :---:     | :---:        | :---:        | :---:        |
| FixMatch  | -            | -            | -            |
| FlexMatch | -            | -            | -            |
| FlexDash  | -            | -            | -            |
