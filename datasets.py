def select_datasets(name, **kwargs):
    if name == 'cifar10':
        from noisy_cifar import NoisyCIFAR10
        return NoisyCIFAR10(**kwargs)
