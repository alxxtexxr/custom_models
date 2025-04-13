import torch

from tsai.models.InceptionTime import InceptionBlock
from tsai.models.layers import Permute, Concat, GAP1d

noop = torch.nn.Sequential()

def ifnone(a, b):
    return b if a is None else a

class _RNNInceptionTimePosX_Base(torch.nn.Module):
    def __init__(self,
                 c_in, c_out, seq_len,
                 # RNN  
                 hidden_size=128, rnn_layers=1, bias=True, cell_dropout=0, rnn_dropout=0.8, bidirectional=False, shuffle=True, 
                 fc_dropout:float=0.,
                 # InceptionTime
                 nf=32, 
                 ):
        super().__init__()
        
        # RNN
        self.rnn = self._cell(seq_len if shuffle else c_in, hidden_size, num_layers=rnn_layers, bias=bias, batch_first=True, 
                              dropout=cell_dropout, bidirectional=bidirectional)
        self.rnn_dropout = torch.nn.Dropout(rnn_dropout) if rnn_dropout else noop
        self.shuffle = Permute(0,2,1) if not shuffle else noop # You would normally permute x. Authors did the opposite.
        
        # InceptionTime
        self.inception_block = InceptionBlock(c_in, nf)
        inception_head_nf = nf * 4
        self.gap = GAP1d(1)
        
        # Head
        self.concat = Concat()
        self.fc_dropout = torch.nn.Dropout(fc_dropout) if fc_dropout else noop
        self.fc = torch.nn.Linear(hidden_size * (1 + bidirectional) + inception_head_nf, c_out)

    def forward(self, x, x_pos):
        # RNN
        rnn_input = self.shuffle(x_pos) # permute --> (batch_size, seq_len, n_vars) when batch_first=True
        output, _ = self.rnn(rnn_input)
        last_out = output[:, -1] # output of last sequence step (many-to-one)
        last_out = self.rnn_dropout(last_out)
        
        # InceptionTime
        x = self.inception_block(x)
        x = self.gap(x)
        
        # Head
        x = self.concat([last_out, x])
        x = self.fc_dropout(x)
        x = self.fc(x)
        return x

class RNNInceptionTimePosX(_RNNInceptionTimePosX_Base):
    _cell = torch.nn.RNN
    
class GRUInceptionTimePosX(_RNNInceptionTimePosX_Base):
    _cell = torch.nn.GRU

class LSTMInceptionTimePosX(_RNNInceptionTimePosX_Base):
    _cell = torch.nn.LSTM