import torch
from .fixmatch import FixMatchModule

__all__ = ['FlexMatchModule']


class FlexMatchCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, num_samples, temperature, threshold):
        super().__init__()
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.temperature = temperature
        self.threshold = threshold
        self.𝜇 = 0.0
        self.register_buffer('Ŷ', torch.tensor([num_classes] * num_samples))

    def forward(self, logits_s, logits_w):
        c, ŷ = (logits_w / self.temperature).softmax(dim=-1).max(dim=-1)
        self.ŷ = torch.where(c > self.threshold, ŷ, -1)

        torch.use_deterministic_algorithms(False)
        β = self.Ŷ.bincount()
        torch.use_deterministic_algorithms(True)
        β = β / (2 * β.max() - β)

        mask = (c > self.threshold * β[ŷ])
        self.𝜇 = mask.float().mean()

        loss = torch.nn.functional.cross_entropy(logits_s, ŷ, reduction='none')
        return (loss * mask).mean()


class FlexMatchModule(FixMatchModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.criterionᵘ = FlexMatchCrossEntropy(
            self.hparams['dataset']['num_classes'],
            self.hparams['dataset']['num_samples'],
            self.hparams.method['temperature'],
            self.hparams.method['threshold'])

    def on_train_batch_end(self, outputs, batch, batch_idx):
        i = batch['unlabeled'][0]
        ŷ = self.criterionᵘ.ŷ
        if torch.distributed.is_initialized():
            i = self.all_gather(i).flatten(end_dim=1)
            ŷ = self.all_gather(ŷ).flatten(end_dim=1)
        self.criterionᵘ.Ŷ[i[ŷ != -1]] = ŷ[ŷ != -1]
