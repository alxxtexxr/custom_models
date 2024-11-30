import os
import sys

import torch
import torch.nn as nn

from tsai.tsai.models.layers import Module, ConvBlock, SqueezeExciteBlock, GAP1d, Concat, noop

# xLSTM
from torch import Tensor
from torch.optim import AdamW
from torch.optim import Optimizer

from typing import Any, Dict, List, Tuple, Callable, Iterable

from itertools import repeat
from einops import rearrange

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './x-lstm')))

from xlstm.lstm import sLSTM
from xlstm.lstm import mLSTM
from xlstm.utils import Hidden
from xlstm.utils import TokenizerWrapper

OptimizerCallable = Callable[[Iterable], Optimizer]

# mLSTM
from math import sqrt
from torch import exp
from torch import sigmoid
from einops import einsum, rearrange

from torch.nn.functional import silu

from xlstm.utils import enlarge_as
from xlstm.utils import CausalConv1d

class mLSTM(nn.Module):
    '''The matrix-Long Short Term Memory (mLSTM) module as
    originally introduced in Beck et al. (2024)] see:
    (https://arxiv.org/abs/2405.04517).
    
    This model is a variant of the standard LSTM model and
    offers superior memory due to its storing values in a
    matrix instead of a scalar. It is fully parallelizable
    and updates internal memory with the covariance rule.
    '''
    
    def __init__(
        self,
        inp_dim : int,
        head_num : int,
        head_dim : int,
        p_factor : int = 2,
        ker_size : int = 4,
    ) -> None:
        super().__init__()
        
        self.inp_dim = inp_dim
        self.head_num = head_num
        self.head_dim = head_dim

        hid_dim = head_num * head_dim
        
        self.inp_norm = nn.LayerNorm(inp_dim)
        self.hid_norm = nn.GroupNorm(head_num, hid_dim)
        
        # NOTE: The factor of two in the output dimension of the up_proj
        # is due to the fact that the output needs to branch into two
        self.up_l_proj = nn.Linear(inp_dim, int(p_factor * inp_dim))
        self.up_r_proj = nn.Linear(inp_dim, hid_dim)
        self.down_proj = nn.Linear(hid_dim, inp_dim)
        
        self.causal_conv = CausalConv1d(1, 1, kernel_size=ker_size)
        
        self.skip = nn.Conv1d(int(p_factor * inp_dim), hid_dim, kernel_size=1, bias=False)
        
        self.W_i = nn.Linear(int(p_factor * inp_dim), head_num)
        self.W_f = nn.Linear(int(p_factor * inp_dim), head_num)
        self.W_o = nn.Linear(int(p_factor * inp_dim), hid_dim)
        
        self.W_q = nn.Linear(int(p_factor * inp_dim), hid_dim)
        self.W_k = nn.Linear(int(p_factor * inp_dim), hid_dim)
        self.W_v = nn.Linear(int(p_factor * inp_dim), hid_dim)
        
    @property
    def device(self) -> str:
        return next(self.parameters()).device
    
    def init_hidden(self, bs : int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        c_0 = torch.zeros(bs, self.head_num, self.head_dim, self.head_dim, device=self.device)
        n_0 = torch.ones (bs, self.head_num, self.head_dim               , device=self.device)
        m_0 = torch.zeros(bs, self.head_num                              , device=self.device)
        
        return c_0, n_0, m_0
    
    def forward(
        self,
        seq: Tensor,
        hid: Tuple[Tensor, Tensor],
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        # Separate the hidden (previous) state into the cell state,
        # the normalizer state, the hidden state, and the stabilizer state.
        c_tm1, n_tm1, m_tm1 = hid
        
        x_n : Tensor = self.inp_norm(seq) # shape: b i
        
        x_t = self.up_l_proj(x_n) # shape: b (i * p_factor)
        r_t = self.up_r_proj(x_n) # shape: b (h d)
        
        # Compute the causal convolutional input (to be 
        # used for the query and key gates)
        x_c = self.causal_conv(x_t) # shape: b 1 (i * p_factor)
        x_c = silu(x_c).squeeze()   # shape: b (i * p_factor)
        
        q_t = rearrange(self.W_q(x_c), 'b (h d) -> b h d', h=self.head_num)
        k_t = rearrange(self.W_k(x_c), 'b (h d) -> b h d', h=self.head_num) / sqrt(self.head_dim)
        v_t = rearrange(self.W_v(x_t), 'b (h d) -> b h d', h=self.head_num)
        
        i_t: Tensor = self.W_i(x_c) # shape: b h
        f_t: Tensor = self.W_f(x_c) # shape: b h
        o_t: Tensor = self.W_o(x_t) # shape: b (h d)
        
        # Compute the gated outputs for the newly computed inputs
        m_t = torch.max(f_t + m_tm1, i_t)
        
        i_t = exp(i_t - m_t)         # Eq. (25) in ref. paper
        f_t = exp(f_t - m_t + m_tm1) # Eq. (26) in ref. paper
        o_t = sigmoid(o_t)           # Eq. (27) in ref. paper
        
        # Update the internal states of the model
        c_t = enlarge_as(f_t, c_tm1) * c_tm1 + enlarge_as(i_t, c_tm1) * einsum(v_t, k_t, 'b h d, b h p -> b h d p')
        n_t = enlarge_as(f_t, n_tm1) * n_tm1 + enlarge_as(i_t, k_t)   * k_t                    
        h_t = o_t * rearrange(
                einsum(c_t, q_t, 'b h d p, b h p -> b h d') /
                einsum(n_t, q_t, 'b h d, b h d -> b h').clamp(min=1).unsqueeze(-1),
                'b h d -> b (h d)'
            ) # Eq. (21) in ref. paper

        x_c = rearrange(x_c, 'b i -> b i 1')
        out = self.hid_norm(h_t) + self.skip(x_c).squeeze() # shape: b (h d)
        out = out * silu(r_t)                               # shape: b (h d)
        # out = self.down_proj(out)                         # shape: h i
        
        # Return output with the residual connection and the
        # newly updated hidden state.
        # return out + seq, (c_t, n_t, m_t)
        return out, (c_t, n_t, m_t)

class xLSTM(nn.Module):
    def __init__(
        self, 
        inp_dim : int,
        num_layers : int,
        signature : Tuple[int, int],
        head_dim : int,
        head_num : int,
        p_factor : Tuple[float, float] = (2, 4/3),
        ker_size : int = 4,
        optimizer : OptimizerCallable = AdamW,
        tokenizer: TokenizerWrapper | None = None,
        inference_kw: Dict[str, Any] = {}
    ) -> None:
        super().__init__()
        
        self.optimizer = optimizer
        self.inference_kw = inference_kw
        self.tokenizer = None if tokenizer is None else tokenizer.get_tokenizer()
        
        m_factor, s_factor = p_factor
        
        # mlstm_par = {
        #     'inp_dim' : inp_dim,
        #     'head_dim' : head_dim,
        #     'head_num' : head_num,
        #     'p_factor' : m_factor,
        #     'ker_size' : ker_size,
        # }
        
        slstm_par = {
            'inp_dim' : inp_dim,
            'head_dim' : head_dim,
            'head_num' : head_num,
            'p_factor' : s_factor,
            'ker_size' : ker_size,
        }
        
        m_num, s_num = signature
        which = [True] * m_num + [False] * s_num
        
        self.llm : List[mLSTM | sLSTM] = nn.ModuleList([
            mLSTM(
                inp_dim=(inp_dim if i == 0 else head_dim*head_num),
                head_dim=head_dim,
                head_num=head_num,
                p_factor=m_factor,
                ker_size=ker_size,
            ) if w else sLSTM(**slstm_par)
            for i, (w, _) in enumerate(zip(repeat(which), range(num_layers)))
        ])
        
    def forward(
        self,
        # tok: Tensor,
        seq: Tensor,
        hid: Hidden | None = None,
        batch_first : bool = False,
    ) -> Tuple[Tensor, Hidden]:
        if batch_first: seq = rearrange(seq, 'b s i -> s b i')
        if hid is None: hid = [l.init_hidden(seq.size(1)) for l in self.llm]
        
        # Pass the sequence through the mLSTM and sLSTM blocks
        out = []
        for inp in seq:
            # Compute model output and update the hidden states
            for i, lstm in enumerate(self.llm):
                inp, hid[i] = lstm(inp, hid[i])
            
            out.append(inp)
        
        out = torch.stack(out, dim=(1 if batch_first else 0))
        
        return out, hid

class MxLSTMFCN_v2(Module):
    def __init__(self, c_in, c_out, 
                 xlstm_signature= (1, 0), xlstm_head_size=8, xlstm_num_heads=4, xlstm_num_layers=1, xlstm_dropout=0.2,
                 fc_dropout=0., conv_layers=[128, 256, 128], kss=[7, 5, 3], se=16):
        
        self.xlstm_head_size = xlstm_head_size
        self.xlstm_num_heads = xlstm_num_heads
        self.conv_layers = conv_layers

        # xLSTM
        self.xlstm = xLSTM(
            inp_dim=c_in,
            signature=xlstm_signature,
            head_dim=xlstm_head_size,
            head_num=xlstm_num_heads,
            num_layers=xlstm_num_layers,
        )
        self.xlstm_dropout = nn.Dropout(xlstm_dropout) if xlstm_dropout else noop
        
        # FCN
        assert len(conv_layers) == len(kss)
        self.convblock1 = ConvBlock(c_in, conv_layers[0], kss[0])
        self.se1 = SqueezeExciteBlock(conv_layers[0], se) if se != 0 else noop
        self.convblock2 = ConvBlock(conv_layers[0], conv_layers[1], kss[1])
        self.se2 = SqueezeExciteBlock(conv_layers[1], se) if se != 0 else noop
        self.convblock3 = ConvBlock(conv_layers[1], conv_layers[2], kss[2])
        self.gap = GAP1d(1)
        
        # Common
        self.concat = Concat()
        self.fc_dropout = nn.Dropout(fc_dropout) if fc_dropout else noop
        # self.fc = nn.Linear(hidden_size * (1 + bidirectional) + conv_layers[-1], c_out)
        self.fc = nn.Linear((xlstm_head_size*xlstm_num_heads) + conv_layers[-1], c_out)
        

    def forward(self, x):  
        # xLSTM
        output, _ = self.xlstm(x)
        last_out = output[:, -1]
        last_out = self.xlstm_dropout(last_out)
        
        # FCN
        x = x.permute(0, 2, 1)
        x = self.convblock1(x)
        x = self.se1(x)
        x = self.convblock2(x)
        x = self.se2(x)
        x = self.convblock3(x)
        x = self.gap(x)

        # Concat
        x = self.concat([last_out, x])
        x = self.fc_dropout(x)
        x = self.fc(x)
        return x