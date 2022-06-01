import torch
import torch.nn as nn

from .base import NondeterministicLayer

__all__ = ['Linear']

class Linear(NondeterministicLayer):
    def __init__(self, in_features, out_features, bias=True, noise='Gaussian', variance=None):
        super().__init__(noise, variance)

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.mu_layer = nn.Linear(in_features, out_features, bias)
        if self.variance is None:
            self.logvar_layer = nn.Linear(in_features, out_features, bias)