import torch

from tsai.tsai.models.InceptionTime import InceptionBlock
from tsai.tsai.models.layers import Permute, Concat, GAP1d

noop = torch.nn.Sequential()

def ifnone(a, b):
    return b if a is None else a

class _RNNInceptionTimeSep_Base(torch.nn.Module):
    def __init__(self,
                 c_in, c_out, seq_len,
                 # RNN  
                 hidden_size=128, rnn_layers=1, bias=True, cell_dropout=0, rnn_dropout=0.8, bidirectional=False, shuffle=True, 
                 fc_dropout:float=0.,
                 # InceptionTime
                 nf=32,
                 # Separate Features
                #  sep=[[0, 1, 2, 3, 4, 5, 6, 7],                               # Left Foot
                #       [8, 9, 10, 11, 12, 13, 14, 15],                         # Right Foot
                #       [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]] # Both Feet
                sep=[[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15]]
                 ):
        super().__init__()

        self.sep = sep

        # RNN
        self.rnn_cell_list = []
        self.rnn_dropout_list = []
        self.rnn_shuffle_list = []

        # InceptionTime
        self.inception_block_list = []
        self.inception_gap_list = []

        for sep_i in sep:
            c_in_sep = len(sep_i)

            # RNN
            rnn_cell = self._cell(seq_len if shuffle else c_in_sep, hidden_size, num_layers=rnn_layers, bias=bias, batch_first=True, 
                             dropout=cell_dropout, bidirectional=bidirectional)
            rnn_dropout_ = torch.nn.Dropout(rnn_dropout) if rnn_dropout else noop
            rnn_shuffle = Permute(0, 2, 1) if not shuffle else noop # You would normally permute x. Authors did the opposite.
            self.rnn_cell_list.append(rnn_cell)
            self.rnn_dropout_list.append(rnn_dropout_)
            self.rnn_shuffle_list.append(rnn_shuffle)
            
            # InceptionTime
            inception_block = InceptionBlock(c_in_sep, nf)
            inception_gap = GAP1d(1)
            self.inception_block_list.append(inception_block)
            self.inception_gap_list.append(inception_gap)
        
        # Head
        self.concat = Concat()
        self.fc_dropout = torch.nn.Dropout(fc_dropout) if fc_dropout else noop
        inception_head_nf = nf * 4
        self.fc = torch.nn.Linear((hidden_size * (1 + bidirectional) + inception_head_nf) * len(sep), c_out)
    
    def forward(self, x):
        outputs = []

        for i, feature_idxs in enumerate(self.sep):
            x_ = x[:, feature_idxs, :]

            # RNN
            rnn_input = self.rnn_shuffle_list[i].to(x_.device)(x_) # permute --> (batch_size, seq_len, n_vars) when batch_first=True
            output, _ = self.rnn_cell_list[i].to(x_.device)(rnn_input)
            rnn_output = output[:, -1] # output of last sequence step (many-to-one)
            rnn_output = self.rnn_dropout_list[i].to(x_.device)(rnn_output)
            
            # Hydra
            inception_outupt = self.inception_block_list[i].to(x_.device)(x_)
            inception_outupt = self.inception_gap_list[i].to(x_.device)(inception_outupt)

            outputs += [rnn_output, inception_outupt]
        
        # Head
        x = self.concat(outputs)
        x = self.fc_dropout(x)
        x = self.fc(x)
        return x

class RNNInceptionTimeSep(_RNNInceptionTimeSep_Base):
    _cell = torch.nn.RNN
    
class GRUInceptionTimeSep(_RNNInceptionTimeSep_Base):
    _cell = torch.nn.GRU

class LSTMInceptionTimeSep(_RNNInceptionTimeSep_Base):
    _cell = torch.nn.LSTM

if __name__ == '__main__':
    x = torch.randn(1, 500, 16).permute(0, 2, 1)
    m = RNNInceptionTimeSep(c_in=16, c_out=4, seq_len=500)
    o = m(x)
    print(o.shape)
