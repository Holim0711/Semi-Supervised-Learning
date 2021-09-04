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

### CIFAR10
| #Labels | 40 | 250 | 4000 |
|:---:|:---:|:---:|:---:|
| Acc. | TODO | 95.47 | 95.95 |


## References
- [official - FixMatch](https://github.com/google-research/fixmatch)
- [kekmodel - FixMatch](https://github.com/kekmodel/FixMatch-pytorch)
- [official - Wide Residual Networks](https://github.com/szagoruyko/wide-residual-networks)
- [ildoonet - RandAugment](https://github.com/ildoonet/pytorch-randaugment)
