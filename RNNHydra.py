import fire
import torch

from tsai.tsai.imports import default_device
from tsai.tsai.models.RNNAttention import _TSTEncoder, get_activation_fn
from tsai.tsai.models.HydraPlus import HydraBackbonePlus
from tsai.tsai.models.layers import Concat, Flatten, SigmoidRange

noop = torch.nn.Sequential()

class _RNNHydra_Base(torch.nn.Module):
    def __init__(self,
                 # RNN  
                 c_in, c_out, seq_len, hidden_size=128, rnn_layers=1, bias=True, cell_dropout=0, rnn_dropout=0.8, bidirectional=False, shuffle=True, 
                 fc_dropout:float=0.,
                 # Hydra  
                 k = 8, g = 64, max_c_in = 8, clip=True, device=default_device(), zero_init=True):
        super().__init__()
        
        # RNN
        self.rnn = self._cell(seq_len if shuffle else c_in, hidden_size, num_layers=rnn_layers, bias=bias, batch_first=True, 
                              dropout=cell_dropout, bidirectional=bidirectional)
        self.rnn_dropout = torch.nn.Dropout(rnn_dropout) if rnn_dropout else noop
        self.shuffle = Permute(0,2,1) if not shuffle else noop # You would normally permute x. Authors did the opposite.
        
        # Hydra
        self.hydra_backbone = HydraBackbonePlus(c_in, c_out, seq_len, k=k, g=g, max_c_in=max_c_in, clip=clip, device=device, zero_init=zero_init)
        
        # Head
        self.concat = Concat()
        self.fc_dropout = torch.nn.Dropout(fc_dropout) if fc_dropout else noop
        self.fc = torch.nn.Linear(hidden_size * (1 + bidirectional) + self.hydra_backbone.num_features, c_out)
    
    def forward(self, x):
        # RNN
        rnn_input = self.shuffle(x) # permute --> (batch_size, seq_len, n_vars) when batch_first=True
        output, _ = self.rnn(rnn_input)
        last_out = output[:, -1] # output of last sequence step (many-to-one)
        last_out = self.rnn_dropout(last_out)
        
        # Hydra
        x = self.hydra_backbone(x)
        
        # Head
        x = self.concat([last_out, x])
        x = self.fc_dropout(x)
        x = self.fc(x)
        return x

class RNNHydra(_RNNHydra_Base):
    _cell = torch.nn.RNN

class LSTMHydra(_RNNHydra_Base):
    _cell = torch.nn.LSTM