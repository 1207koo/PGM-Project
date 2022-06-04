import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('..')
import nd
from args import *

class Sampler2(nn.Module):
    def __init__(self, feature_maps, nondeterministic=False):
        super(Sampler2, self).__init__()        
        layers = []
        if nondeterministic:
            layers.append(nd.Linear(10 * feature_maps, 20 * feature_maps, bias=False, variance = args.variance, batchnorm=True))
        else:
            layers.append(nn.Linear(10 * feature_maps, 20 * feature_maps, bias=False))
        layers.append(nn.Linear(20 * feature_maps, 10 * feature_maps, bias=False))     
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = x
        for i in range(len(self.layers)):
            out = self.layers[i](out)
            if i < len(self.layers) - 1:
                out = F.leaky_relu(out, negative_slope = 0.1)
        return out

class Sampler3(nn.Module):
    def __init__(self, feature_maps, nondeterministic=False):
        super(Sampler3, self).__init__()        
        layers = []
        layers.append(nn.Linear(10 * feature_maps, 40 * feature_maps, bias=False))
        if nondeterministic:
            layers.append(nd.Linear(40 * feature_maps, 20 * feature_maps, bias=False, variance = args.variance, batchnorm=True))
        else:
            layers.append(nn.Linear(40 * feature_maps, 20 * feature_maps, bias=False))
        layers.append(nn.Linear(20 * feature_maps, 10 * feature_maps, bias=False))       
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = x
        for i in range(len(self.layers)):
            out = self.layers[i](out)
            if i < len(self.layers) - 1:
                out = F.leaky_relu(out, negative_slope = 0.1)
        return out

class DiscriminatorSingle2(nn.Module):
    def __init__(self, feature_maps):
        super(DiscriminatorSingle2, self).__init__()        
        layers = []
        layers.append(nn.Linear(10 * feature_maps, feature_maps, bias=False))
        layers.append(nn.Linear(feature_maps, 1, bias=False)) 
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = x
        for i in range(len(self.layers)):
            out = self.layers[i](out)
            if i < len(self.layers) - 1:
                out = F.leaky_relu(out, negative_slope = 0.1)
        return out

class DiscriminatorSingle3(nn.Module):
    def __init__(self, feature_maps):
        super(DiscriminatorSingle3, self).__init__()        
        layers = []
        layers.append(nn.Linear(10 * feature_maps, 5 * feature_maps, bias=False))
        layers.append(nn.Linear(5 * feature_maps, feature_maps, bias=False))
        layers.append(nn.Linear(feature_maps, 1, bias=False)) 
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = x
        for i in range(len(self.layers)):
            out = self.layers[i](out)
            if i < len(self.layers) - 1:
                out = F.leaky_relu(out, negative_slope = 0.1)
        return out


class DiscriminatorDouble3(nn.Module):
    def __init__(self, feature_maps):
        super(DiscriminatorDouble3, self).__init__()        
        layers = []
        layers.append(nn.Linear(20 * feature_maps, 5 * feature_maps, bias=False))
        layers.append(nn.Linear(5 * feature_maps, feature_maps, bias=False))
        layers.append(nn.Linear(feature_maps, 1, bias=False)) 
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = x
        for i in range(len(self.layers)):
            out = self.layers[i](out)
            if i < len(self.layers) - 1:
                out = F.leaky_relu(out, negative_slope = 0.1)
        return out

class DiscriminatorDouble4(nn.Module):
    def __init__(self, feature_maps):
        super(DiscriminatorDouble4, self).__init__()        
        layers = []
        layers.append(nn.Linear(20 * feature_maps, 10 * feature_maps, bias=False))
        layers.append(nn.Linear(10 * feature_maps, 5 * feature_maps, bias=False))
        layers.append(nn.Linear(5 * feature_maps, feature_maps, bias=False))
        layers.append(nn.Linear(feature_maps, 1, bias=False)) 
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = x
        for i in range(len(self.layers)):
            out = self.layers[i](out)
            if i < len(self.layers) - 1:
                out = F.leaky_relu(out, negative_slope = 0.1)
        return out