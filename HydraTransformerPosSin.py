import os
import sys
import fire
import numpy as np
import torch

from tsai.tsai.imports import default_device
from tsai.tsai.models.RNNAttention import _TSTEncoder, get_activation_fn
from tsai.tsai.models.HydraPlus import HydraBackbonePlus
from tsai.tsai.models.layers import Flatten, SigmoidRange

noop = torch.nn.Sequential()

class SinusoidalPositionalEncoding(torch.nn.Module):
    def __init__(self, max_len, embed_size):
        super(SinusoidalPositionalEncoding, self).__init__()
        # Create the positional encodings once in log space
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) * (-torch.log(torch.tensor(10000.0)) / embed_size))
        pos_encodings = torch.zeros(max_len, embed_size)
        pos_encodings[:, 0::2] = torch.sin(position * div_term)
        pos_encodings[:, 1::2] = torch.cos(position * div_term)
        pos_encodings = pos_encodings.unsqueeze(0).permute(0, 2, 1)  # Shape: (1, embed_size, max_len)
        self.register_buffer('pos_encodings', pos_encodings)

    def forward(self, x):
        # x is expected to be of shape (batch_size, c_in, seq_len)
        seq_len = x.size(2)
        # Truncate or extend the positional encodings to match the sequence length
        position_embeddings = self.pos_encodings[:, :, :seq_len]  # Shape: (1, embed_size, seq_len)
        return x + position_embeddings

class HydraBackbonePlus(torch.nn.Module):

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
        self.num_features = min(2, self.g) * self._g * self.k * self.num_dilations * 2

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
                _Z = torch.nn.functional.conv1d(X[:, self.I[dilation_index][diff_index]].sum(2) if diff_index == 0 else diff_X[:, self.I[dilation_index][diff_index]].sum(2),
                                                self.W[dilation_index][diff_index], dilation = d, padding = p, groups = self._g)
                _Z = _Z.view(bs, self._g, self.k, -1)
                
                max_values, max_indices = _Z.max(2)
                count_max = torch.zeros(bs, self._g, self.k, device=self.device)

                min_values, min_indices = _Z.min(2)
                count_min = torch.zeros(bs, self._g, self.k, device=self.device)

                count_max.scatter_add_(-1, max_indices, max_values)
                count_min.scatter_add_(-1, min_indices, torch.ones_like(min_values))

                Z.append(count_max)
                Z.append(count_min)

        Z = torch.cat(Z, 1)#.view(bs, -1)
        
        if self.clip:
            Z = torch.nn.functional.relu(Z)

        return Z

class HydraTransformerPosSin(torch.nn.Module):
    def __init__(self,
                 c_in, c_out, seq_len,
                 # Transformer  
                 encoder_layers=3, n_heads:int=16, d_k=None, d_v=None, 
                 d_ff=256, encoder_dropout=0.1, act="gelu", fc_dropout:float=0.,
                 y_range=None,
                 # Hydra  
                 k = 8, g = 64, max_c_in = 8, clip=True, device=default_device(), zero_init=True):
        super().__init__()
        
        # Hydra
        self.hydra_backbone = HydraBackbonePlus(c_in, c_out, seq_len, k=k, g=g, max_c_in=max_c_in, clip=clip, device=device, zero_init=zero_init)
        
        # Transformer
        q_len = self.hydra_backbone.num_dilations * self.hydra_backbone.g * 2
        d_model = self.hydra_backbone.k
        self.encoder = _TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, dropout=encoder_dropout, activation=act, n_layers=encoder_layers)
        self.encoder_head_nf = q_len * d_model
        self.encoder_head = self.create_encoder_head(nf=self.encoder_head_nf, c_out=c_out, act=act, fc_dropout=fc_dropout, y_range=y_range)

        # Positional Encoding (using Sinusoidal)
        self.positional_encoding = SinusoidalPositionalEncoding(seq_len, c_in)
    
    def create_encoder_head(self, nf, c_out, act="gelu", fc_dropout=0., y_range=None):
        layers = [get_activation_fn(act), Flatten()]
        if fc_dropout: layers += [torch.nn.Dropout(fc_dropout)]
        layers += [torch.nn.Linear(nf, c_out)]
        if y_range: layers += [SigmoidRange(*y_range)]
        return torch.nn.Sequential(*layers)
    
    def forward(self, x):
        x_pos = self.positional_encoding(x) # Add positional encodings

        # Hydra
        hydra_output = self.hydra_backbone(x_pos) # [B, L, D]
        
        # Transformer
        z = self.encoder(hydra_output)    # z: [bs x q_len x d_model]
        z = z.transpose(2,1).contiguous() # z: [bs x d_model x q_len]
        z = self.encoder_head(z)
        return z