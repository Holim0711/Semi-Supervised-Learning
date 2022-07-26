import torch
from .fixmatch import FixMatchCrossEntropy, FixMatchClassifier

__all__ = ['FlexMatchClassifier']


class FlexMatchCrossEntropy(FixMatchCrossEntropy):
    def __init__(self, num_classes, num_samples, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.register_buffer('ŷ', torch.tensor([num_classes] * num_samples))

    def all_gather(self, x, world_size):
        x_list = [torch.zeros_like(x) for _ in range(world_size)]
        torch.distributed.all_gather(x_list, x)
        return torch.hstack(x_list)

    def forward(self, logits_s, logits_w, indices):
        probs = torch.softmax(logits_w / self.temperature, dim=-1)
        max_probs, targets = probs.max(dim=-1)

        β = self.ŷ.bincount()
        β = β / β.max()
        β = β / (2 - β)
        masks = (max_probs > self.threshold * β[targets]).float()

        ŷ = torch.where(max_probs > self.threshold, targets, -1)
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            ŷ = self.all_gather(ŷ, world_size)
            indices = self.all_gather(indices, world_size)
        self.ŷ[indices[ŷ != -1]] = ŷ[ŷ != -1]

        loss = torch.nn.functional.cross_entropy(
            logits_s, targets, reduction='none') * masks
        self.𝜇ₘₐₛₖ = masks.mean().detach()

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class FlexMatchClassifier(FixMatchClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.criterionᵤ = FlexMatchCrossEntropy(
            self.hparams.model['backbone']['num_classes'],
            {
                'CIFAR10': 50000,
                'CIFAR100': 50000,
            }[self.hparams.dataset['name']],
            **self.hparams.model['loss_u']
        )
