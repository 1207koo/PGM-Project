import torch
import torch.nn as nn

from .base import NondeterministicLayer

__all__ = ['Linear']

class Linear(NondeterministicLayer):
    def __init__(self, in_features, out_features, bias=True, noise='Gaussian', variance=None, batchnorm=False):
        super().__init__(noise, variance, batchnorm)

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        if self.batchnorm:
            self.batchnorm_layer = nn.BatchNorm1d(in_features)
        self.mu_layer = nn.Linear(in_features, out_features, bias)
        if self.variance is None:
            self.logvar_layer = nn.Linear(in_features, out_features, bias)