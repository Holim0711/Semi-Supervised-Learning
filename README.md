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
python train.py configs/fixmatch/cifar10.json --gpus 1 --max_epoch 9362 --random_seed 0
python train.py configs/fixmatch/cifar10.json --gpus 1 --max_epoch 9362 --random_seed 0 --data.num_labeled 100
```

## Results

### CIFAR10
|     |   40  |  100  |
|:---:| :---: | :---: |
| Acc |  WIP  |  WIP  |

### CIFAR100

```
python train.py configs/fixmatch/cifar100.json --gpus 4 --strategy ddp --sync_batchnorm --max_epoch 10000 --random_seed 0
```

|     |  400  |  1000 |
|:---:| :---: | :---: |
| Acc |  WIP  |  WIP  |


## References
- [official - FixMatch](https://github.com/google-research/fixmatch)
- [kekmodel - FixMatch](https://github.com/kekmodel/FixMatch-pytorch)
