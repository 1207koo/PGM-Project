import torch
import torch.nn as nn
import numpy as np

class NondeterministicLayer(nn.Module):
    def __init__(self, noise='Gaussian', variance=None):
        super(NondeterministicLayer, self).__init__()

        self.noise = noise # 'Gaussian', 'Uniform'
        self.variance = variance # None or float

        self.mu_layer = None
        self.logvar_layer = None
    
    def forward(self, x, return_mu=False):
        mu = self.mu_layer(x)
        if return_mu:
            return mu
        if self.variance is None:
            sigma = torch.exp(self.logvar_layer(x) / 2.0)
        else:
            sigma = np.sqrt(self.variance)
        
        if self.noise.lower() == 'gaussian':
            y = mu + sigma * torch.randn(mu.size(), device=x.device)
        elif self.noise.lower() == 'uniform':
            y = mu + np.sqrt(12.0) * sigma * (torch.rand(mu.size(), device=x.device) - 0.5)
        else:
            raise NotImplementedError
        return y