import torch

from tsai.models.InceptionTime import InceptionBlock
from tsai.models.layers import Permute, Concat, GAP1d

noop = torch.nn.Sequential()

def ifnone(a, b):
    return b if a is None else a

class _InceptionTimeRNN_Base(torch.nn.Module):
    def __init__(self,
                 c_in, c_out, seq_len,
                 fc_dropout=0.0,
                 # InceptionTime parameters
                 nf=32, 
                 # RNN parameters
                 hidden_size=128, rnn_layers=1, bias=True, cell_dropout=0.0, bidirectional=False, shuffle=True, 
                 ):
        super().__init__()

        # InceptionTime layers
        self.inception_block = InceptionBlock(c_in, nf)
        
        # RNN layers
        self.rnn = self._cell(seq_len if shuffle else c_in, hidden_size, num_layers=rnn_layers, bias=bias, batch_first=True, 
                              dropout=cell_dropout, bidirectional=bidirectional)
        self.shuffle = Permute(0, 2, 1) if not shuffle else noop # You would normally permute x. Authors did the opposite.
        
        # FC layers
        self.fc_dropout = torch.nn.Dropout(fc_dropout) if fc_dropout else noop
        self.fc = torch.nn.Linear(hidden_size * (1 + bidirectional), c_out)
    
    def forward(self, x):
        # InceptionTime forward
        inception_out = self.inception_block(x)

        # RNN forward
        rnn_in = self.shuffle(inception_out) # permute --> (batch_size, seq_len, n_vars) when batch_first=True
        rnn_out, _ = self.rnn(rnn_in)
        rnn_out = rnn_out[:, -1] # output of last sequence step (many-to-one)

        # FC forward
        fc_out = self.fc(self.fc_dropout(rnn_out))
        return fc_out

class InceptionTimeRNN(_InceptionTimeRNN_Base):
    _cell = torch.nn.RNN
    
class InceptionTimeGRU(_InceptionTimeRNN_Base):
    _cell = torch.nn.GRU

class InceptionTimeLSTM(_InceptionTimeRNN_Base):
    _cell = torch.nn.LSTM
