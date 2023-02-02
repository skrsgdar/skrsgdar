from torch.nn import Module
from torch.nn import Linear
from opacus.grad_sample import GradSampleModule

import torch


class LR(Module):
    def __init__(self, n_in, n_out, bias=False):
        super(LR, self).__init__()
        self.linear = GradSampleModule(Linear(n_in, n_out, bias=bias))

    def forward(self, x):
        return self.linear(x)


def get_model(type: str):
    if type.lower() == "lr":
        return LR
