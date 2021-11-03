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
python run.py train configs/cifar10.json --n 4000 --gpus 2 --sync_batchnorm --accelerator 'ddp' --max_epochs 9362
```

## Results

### CIFAR10
|     |   4   |   25  |  400  |
|:---:| :---: | :---: | :---: |
| Acc | 93.82 | 95.48 | 95.95 |

### CIFAR100
|     |   4   |   25  |  100  |
|:---:| :---: | :---: | :---: |
| Acc |  WIP  |  WIP  |  WIP  |


## References
- [official - FixMatch](https://github.com/google-research/fixmatch)
- [kekmodel - FixMatch](https://github.com/kekmodel/FixMatch-pytorch)
