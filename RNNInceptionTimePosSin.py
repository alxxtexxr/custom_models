import torch

from tsai.models.InceptionTime import InceptionBlock
from tsai.models.layers import Permute, Concat, GAP1d

noop = torch.nn.Sequential()

def ifnone(a, b):
    return b if a is None else a

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

class _RNNInceptionTimePosSin_Base(torch.nn.Module):
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

        # Positional Encoding (using Sinusoidal)
        self.positional_encoding = SinusoidalPositionalEncoding(seq_len, c_in)
    
    def forward(self, x):
        # RNN
        x_pos = self.positional_encoding(x) # Add positional encodings
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

class RNNInceptionTimePosSin(_RNNInceptionTimePosSin_Base):
    _cell = torch.nn.RNN
    
class GRUInceptionTimePosSin(_RNNInceptionTimePosSin_Base):
    _cell = torch.nn.GRU

class LSTMInceptionTimePosSin(_RNNInceptionTimePosSin_Base):
    _cell = torch.nn.LSTM