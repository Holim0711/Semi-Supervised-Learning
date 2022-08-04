import torch
from .fixmatch import FixMatchCrossEntropy, FixMatchClassifier

__all__ = ['FlexMatchClassifier']


class FlexMatchCrossEntropy(FixMatchCrossEntropy):
    def __init__(self, num_classes, num_samples, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.Ŷ = torch.tensor([num_classes] * num_samples)

    def forward(self, logits_s, logits_w):
        probs = torch.softmax(logits_w / self.temperature, dim=-1)
        max_probs, targets = probs.max(dim=-1)

        β = self.Ŷ.bincount()
        β = β / β.max()
        β = β / (2 - β)
        β = β.to(targets.device)
        masks = (max_probs > self.threshold * β[targets]).float()

        self.ŷ = torch.where(max_probs > self.threshold, targets, -1)

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

    def training_step(self, batch, batch_idx):
        result = super().training_step(batch, batch_idx)
        i = batch['unlabeled'][0].cpu()
        ŷ = self.criterionᵤ.ŷ.cpu()
        if torch.distributed.is_initialized():
            i = self.all_gather(i).flatten(end_dim=1)
            ŷ = self.all_gather(ŷ).flatten(end_dim=1)
        self.criterionᵤ.Ŷ[i[ŷ != -1]] = ŷ[ŷ != -1]
        return result
