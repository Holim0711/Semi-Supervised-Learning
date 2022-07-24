import torch
from .fixmatch import FixMatchCrossEntropy, FixMatchClassifier

__all__ = ['FlexMatchClassifier']


class FlexMatchCrossEntropy(FixMatchCrossEntropy):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes

    def forward(self, logits_s, logits_w):
        probs = torch.softmax(logits_w / self.temperature, dim=-1)
        max_probs, targets = probs.max(dim=-1)

        β = targets.bincount(max_probs > self.threshold, self.num_classes)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(β)
        β /= max(β.max(), len(targets) - β.sum())
        β /= 2 - β
        masks = (max_probs > self.threshold * β[targets]).float()

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
            num_classes=self.hparams.model['backbone']['num_classes'],
            **self.hparams.model['loss_u']
        )
