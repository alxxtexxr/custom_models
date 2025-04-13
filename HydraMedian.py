from collections import OrderedDict
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tsai.imports import default_device
from tsai.models.layers import Flatten, rocket_nd_head

class HydraMedianBackbonePlus(nn.Module):

    def __init__(self, c_in, c_out, seq_len, k = 8, g = 64, max_c_in = 8, clip=True, device=default_device(), zero_init=True):

        super().__init__()

        self.k = k # num kernels per group
        self.g = g # num groups

        max_exponent = np.log2((seq_len - 1) / (9 - 1)) # kernel length = 9

        self.dilations = 2 ** torch.arange(int(max_exponent) + 1, device=device)
        self.num_dilations = len(self.dilations)

        self.paddings = torch.div((9 - 1) * self.dilations, 2, rounding_mode = "floor").int()

        # if g > 1, assign: half the groups to X, half the groups to diff(X)
        divisor = 2 if self.g > 1 else 1
        _g = g // divisor
        self._g = _g
        self.W = [self.normalize(torch.randn(divisor, k * _g, 1, 9)).to(device=device) for _ in range(self.num_dilations)]


        # combine c_in // 2 channels (2 < n < max_c_in)
        c_in_per = np.clip(c_in // 2, 2, max_c_in)
        self.I = [torch.randint(0, c_in, (divisor, _g, c_in_per), device=device) for _ in range(self.num_dilations)]

        # clip values
        self.clip = clip

        self.device = device
        self.num_features = min(2, self.g) * self._g * self.k * self.num_dilations * 3#2

    @staticmethod
    def normalize(W):
        W -= W.mean(-1, keepdims = True)
        W /= W.abs().sum(-1, keepdims = True)
        return W

    # transform in batches of *batch_size*
    def batch(self, X, split=None, batch_size=256):
        bs = X.shape[0]
        if bs <= batch_size:
            return self(X)
        elif split is None:
            Z = []
            for i in range(0, bs, batch_size):
                Z.append(self(X[i:i+batch_size]))
            return torch.cat(Z)
        else:
            Z = []
            batches = torch.as_tensor(split).split(batch_size)
            for i, batch in enumerate(batches):
                Z.append(self(X[batch]))
            return torch.cat(Z)

    def forward(self, X):

        bs = X.shape[0]

        if self.g > 1:
            diff_X = torch.diff(X)

        Z = []
        
        for dilation_index in range(self.num_dilations):

            d = self.dilations[dilation_index].item()
            p = self.paddings[dilation_index].item()

            # diff_index == 0 -> X
            # diff_index == 1 -> diff(X)
            for diff_index in range(min(2, self.g)):
                _Z = F.conv1d(X[:, self.I[dilation_index][diff_index]].sum(2) if diff_index == 0 else diff_X[:, self.I[dilation_index][diff_index]].sum(2),
                              self.W[dilation_index][diff_index], dilation = d, padding = p, groups = self._g).view(bs, self._g, self.k, -1)

                max_values, max_indices = _Z.max(2)
                count_max = torch.zeros(bs, self._g, self.k, device=self.device)

                min_values, min_indices = _Z.min(2)
                count_min = torch.zeros(bs, self._g, self.k, device=self.device)
                
                median_values, median_indices = _Z.median(2)
                count_median = torch.zeros(bs, self._g, self.k, device=self.device)

                count_max.scatter_add_(-1, max_indices, max_values)
                count_min.scatter_add_(-1, min_indices, torch.ones_like(min_values))
                count_median.scatter_add_(-1, median_indices, median_values)

                Z.append(count_max)
                Z.append(count_min)
                Z.append(count_median)
        
        Z = torch.cat(Z, 1).view(bs, -1)
        
        if self.clip:
            Z = F.relu(Z)
        
        return Z

class HydraMedianPlus(nn.Sequential):

    def __init__(self,
        c_in:int, # num of channels in input
        c_out:int, # num of channels in output
        seq_len:int, # sequence length
        d:tuple=None, # shape of the output (when ndim > 1)
        k:int=8, # number of kernels per group
        g:int=64, # number of groups
        max_c_in:int=8, # max number of channels per group
        clip:bool=True, # clip values >= 0
        use_bn:bool=True, # use batch norm
        fc_dropout:float=0., # dropout probability
        custom_head:Any=None, # optional custom head as a torch.nn.Module or Callable
        zero_init:bool=True, # set head weights and biases to zero
        use_diff:bool=True, # use diff(X) as input
        device:str=default_device(), # device to use
        ):

        # Backbone
        backbone = HydraMedianBackbonePlus(c_in, c_out, seq_len, k=k, g=g, max_c_in=max_c_in, clip=clip, device=device, zero_init=zero_init)
        num_features = backbone.num_features
        
        # Head
        self.head_nf = num_features
        if custom_head is not None:
            if isinstance(custom_head, nn.Module): head = custom_head
            else: head = custom_head(self.head_nf, c_out, 1)
        elif d is not None:
            head = rocket_nd_head(num_features, c_out, seq_len=None, d=d, use_bn=use_bn, fc_dropout=fc_dropout, zero_init=zero_init)
        else:
            layers = [Flatten()]
            if use_bn:
                layers += [nn.BatchNorm1d(num_features)]
            if fc_dropout:
                layers += [nn.Dropout(fc_dropout)]
            linear = nn.Linear(num_features, c_out)
            if zero_init:
                nn.init.constant_(linear.weight.data, 0)
                nn.init.constant_(linear.bias.data, 0)
            layers += [linear]
            head = nn.Sequential(*layers)

        super().__init__(OrderedDict([('backbone', backbone), ('head', head)]))

HydraMedian = HydraMedianPlus