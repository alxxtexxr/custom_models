import os
import sys

import torch
import torch.nn as nn

from tsai.models.layers import Module, ConvBlock, SqueezeExciteBlock, GAP1d, Concat, noop

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
        
        mlstm_par = {
            'inp_dim' : inp_dim,
            'head_dim' : head_dim,
            'head_num' : head_num,
            'p_factor' : m_factor,
            'ker_size' : ker_size,
        }
        
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
            mLSTM(**mlstm_par) if w else sLSTM(**slstm_par)
            for w, _ in zip(repeat(which), range(num_layers))
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

class MxLSTMFCN(Module):
    def __init__(self, c_in, c_out,
                 xlstm_signature= (1, 0), xlstm_head_size=8, xlstm_num_heads=4, xlstm_num_layers=1, xlstm_dropout=0.2,
                 fc_dropout=0., conv_layers=[128, 256, 128], kss=[7, 5, 3], se=12):
        
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
        self.fc = nn.Linear(c_in + conv_layers[-1], c_out)
        

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