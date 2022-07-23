# FixMatch-Lightning

## Dependencies
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install pytorch-lightning

# install weaver-pytorch
# install dual-cifar-lightning
```

## Training
```
python run.py train configs/cifar10.json --n 40 --gpus 1 --max_epochs 9362
python run.py train configs/cifar10.json --n 250 --gpus 1 --max_epochs 9362
python run.py train configs/cifar10.json --n 4000 --gpus 2 --sync_batchnorm --accelerator 'ddp' --max_epochs 9362
```

## Testing
```
python run.py test configs/cifar10.json --gpus 1 --ckpt_path {path_to_checkpoint.ckpt}
```

## Results

### CIFAR10
|     |   4   |   25  |  400  |
|:---:| :---: | :---: | :---: |
| Acc | 93.73 | 95.64 | 96.15 |

### CIFAR100

```
python train.py configs/fixmatch/cifar100.json --gpus 4 --strategy ddp --sync_batchnorm --max_epoch 10000 --random_seed 0
```

|     |   4   |   25  |  100  |
|:---:| :---: | :---: | :---: |
| Acc |  WIP  |  WIP  |  WIP  |


## References
- [official - FixMatch](https://github.com/google-research/fixmatch)
- [kekmodel - FixMatch](https://github.com/kekmodel/FixMatch-pytorch)
