import torch
from .flexmatch import FlexMatchModule
from math import exp, log, cos, pi

__all__ = ['FlexDashModule']


def cosine_annealing(max, min, t):
    return min + (max - min) * 0.5 * (1 + cos(t * pi))


class FlexDashCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, num_samples, temperature, threshold, warmup):
        super().__init__()
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.temperature = temperature
        self.threshold = threshold
        self.warmup = warmup
        self.𝜇 = 0.0
        self.register_buffer('Ŷ', torch.tensor([num_classes] * num_samples))
        self.iteration = 0

    def forward(self, logits_s, logits_w):
        c, ŷ = (logits_w / self.temperature).softmax(dim=-1).max(dim=-1)
        self.ŷ = torch.where(c > self.threshold, ŷ, -1)

        torch.use_deterministic_algorithms(False)
        β = self.Ŷ.bincount(minlength=self.num_classes + 1)
        torch.use_deterministic_algorithms(True)
        β[self.num_classes] = 1
        β = β / (2 * β.max() - β)

        if self.iteration < self.warmup:
            τ = exp(-cosine_annealing(log(self.num_classes), -log(self.threshold), self.iteration / self.warmup))
        else:
            τ = self.threshold
        mask = (c > τ * β[ŷ])
        self.𝜇 = mask.float().mean()

        loss = torch.nn.functional.cross_entropy(logits_s, ŷ, reduction='none')
        return (loss * mask).mean()


class FlexDashModule(FlexMatchModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.criterionᵘ = FlexDashCrossEntropy(
            self.hparams['dataset']['num_classes'],
            self.hparams['dataset']['num_samples'],
            self.hparams.method['temperature'],
            self.hparams.method['threshold'],
            self.hparams.method['warmup'])

    def on_train_epoch_start(self):
        self.criterionᵘ.iteration = self.current_epoch
