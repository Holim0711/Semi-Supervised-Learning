# FixMatch-Lightning

## Dependencies
```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
pip install pytorch-lightning
pip install lightning-bolts

holim-lightning
noisy-cifar
```

## Results

### Accuracy
| Dataset | 4 | 25 | 400 |
|:---:|:---:|:---:|:---:|
| CIFAR10 | 93.77 | 95.48 | 95.95 |
| CIFAR100 | WIP | WIP | WIP |


## References
- [official - FixMatch](https://github.com/google-research/fixmatch)
- [kekmodel - FixMatch](https://github.com/kekmodel/FixMatch-pytorch)
- [official - Wide Residual Networks](https://github.com/szagoruyko/wide-residual-networks)
- [ildoonet - RandAugment](https://github.com/ildoonet/pytorch-randaugment)
