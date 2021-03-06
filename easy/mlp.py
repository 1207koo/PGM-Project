from utils import *
from args import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm

import sys
sys.path.append('..')
import nd

class MLP(nn.Module):
    def __init__(self, hidden_neurons, n_layers, input_shape, num_classes, rotations, few_shot, nondeterministic=False):
        super(MLP, self).__init__()
        self.layers = []
        last_size = input_shape[0] * input_shape[1] * input_shape[2]
        for i in range(n_layers):
            if nondeterministic and i % 2 == 1 and i < n_layers - 1:
                self.layers.append(nd.Linear(last_size, hidden_neurons))
            else:
                self.layers.append(nn.Linear(last_size, hidden_neurons))
            last_size = hidden_neurons
            self.layers.append(nn.ReLU())
        self.module_layers = nn.ModuleList(self.layers)
        self.last_layer = linear(last_size, num_classes)
        self.rotations = rotations
        self.linear_rot = linear(last_size, 4)

    def forward(self, x, run_type = 'forward'):
        if run_type == 'linear':
            return self.last_layer(x)
        elif run_type == 'linear_rot':
            return self.linear_rot(x)
        features = x.reshape(x.shape[0], -1)
        for i in range(len(self.layers)):
            features = self.module_layers[i](features)
            if args.dropout > 0:
                out = F.dropout(out, p=args.dropout, training=self.training, inplace=True)
        out = self.last_layer(features)
        if self.rotations:
            out_rot = self.linear_rot(features)
            return (out, out_rot), features
        return out, features
