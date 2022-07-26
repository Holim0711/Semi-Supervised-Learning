import torch


class EMA(torch.optim.swa_utils.AveragedModel):
    def __init__(self, model, momentum):
        def avg_fn(avg_model, model, num_averaged):
            return momentum * avg_model + (1 - momentum) * model
        super().__init__(model, avg_fn=avg_fn)

    def update_parameters(self, model):
        super().update_parameters(model)
        for a, b in zip(self.module.buffers(), model.buffers()):
            a.copy_(b.to(a.device))


def change_bn_momentum(model, momentum):
    if isinstance(model, torch.nn.BatchNorm2d):
        model.momentum = 1 - momentum
    else:
        for children in model.children():
            change_bn_momentum(children, momentum)


def replace_relu_to_lrelu(model, a=0.1):
    for child_name, child in model.named_children():
        if isinstance(child, torch.nn.ReLU):
            setattr(model, child_name, torch.nn.LeakyReLU(a, inplace=True))
        else:
            replace_relu_to_lrelu(child, a)
