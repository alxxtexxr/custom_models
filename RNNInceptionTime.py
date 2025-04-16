import torch
import torch.nn as nn

from tsai.models.InceptionTime import InceptionBlock
from tsai.models.layers import Permute, Concat, GAP1d

noop = nn.Sequential()

def ifnone(a, b):
    return b if a is None else a

class _RNNInceptionTime_Base(nn.Module):
    def __init__(self,
                 c_in, c_out, seq_len,
                 # RNN params
                 hidden_size=128, rnn_layers=1, bias=True, cell_dropout=0, rnn_dropout=0.8, bidirectional=False, shuffle=True, 
                 fc_dropout:float=0.,
                 # InceptionTime params
                 nf=32, 
                 ):
        super().__init__()
        
        # RNN layers
        self.rnn = self._cell(seq_len if shuffle else c_in, hidden_size, num_layers=rnn_layers, bias=bias, batch_first=True, 
                              dropout=cell_dropout, bidirectional=bidirectional)
        self.rnn_dropout = nn.Dropout(rnn_dropout) if rnn_dropout else noop
        self.shuffle = Permute(0,2,1) if not shuffle else noop # You would normally permute x. Authors did the opposite.
        
        # InceptionTime layers
        self.inception_block = InceptionBlock(c_in, nf)
        inception_head_nf = nf * 4
        self.gap = GAP1d(1)
        
        # Head layers
        self.concat = Concat()
        self.fc_dropout = nn.Dropout(fc_dropout) if fc_dropout else noop
        self.fc = nn.Linear(hidden_size * (1 + bidirectional) + inception_head_nf, c_out)
    
    def forward(self, x):
        # RNN forward
        rnn_input = self.shuffle(x) # permute --> (batch_size, seq_len, n_vars) when batch_first=True
        output, _ = self.rnn(rnn_input)
        last_out = output[:, -1] # output of last sequence step (many-to-one)
        last_out = self.rnn_dropout(last_out)
        
        # InceptionTime forward
        x = self.inception_block(x)
        x = self.gap(x)
        
        # Head forward
        x = self.concat([last_out, x])
        x = self.fc_dropout(x)
        x = self.fc(x)
        return x

class RNNInceptionTime(_RNNInceptionTime_Base):
    _cell = nn.RNN
    
class GRUInceptionTime(_RNNInceptionTime_Base):
    _cell = nn.GRU

class LSTMInceptionTime(_RNNInceptionTime_Base):
    _cell = nn.LSTM
