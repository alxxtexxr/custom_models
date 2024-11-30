import torch

from tsai.tsai.models.InceptionTime import InceptionBlock
from tsai.tsai.models.layers import Permute, Concat, GAP1d

noop = torch.nn.Sequential()

def ifnone(a, b):
    return b if a is None else a

class _RNNInceptionTimeBody2_Base(torch.nn.Module):
    def __init__(self,
                 c_in, c_out, seq_len,
                 # Body Metrics 
                 body_c_in, body_nf=128,
                 # RNN  
                 hidden_size=128, rnn_layers=1, bias=True, cell_dropout=0, rnn_dropout=0.8, bidirectional=False, shuffle=True, 
                 fc_dropout:float=0.,
                 # InceptionTime
                 nf=32, 
                 ):
        super().__init__()

        inception_head_nf = nf * 4
        assert  body_nf == (hidden_size * (1 + bidirectional)) == inception_head_nf
        
        # RNN
        self.rnn = self._cell(seq_len if shuffle else c_in, hidden_size, num_layers=rnn_layers, bias=bias, batch_first=True, 
                              dropout=cell_dropout, bidirectional=bidirectional)
        self.rnn_dropout = torch.nn.Dropout(rnn_dropout) if rnn_dropout else noop
        self.shuffle = Permute(0, 2, 1) if not shuffle else noop # You would normally permute x. Authors did the opposite.
        
        # InceptionTime
        self.inception_block = InceptionBlock(c_in, nf)
        
        self.gap = GAP1d(1)
        
        # Head
        self.concat = Concat()
        self.fc_dropout = torch.nn.Dropout(fc_dropout) if fc_dropout else noop
        self.fc = torch.nn.Linear((hidden_size * (1 + bidirectional)) + inception_head_nf, c_out)

        # Body Metrics
        self.body_block = torch.nn.Sequential(
            torch.nn.Linear(body_c_in, body_nf),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(body_nf, body_nf * 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(body_nf * 2, body_nf),
            torch.nn.ReLU(),
        )

    def forward(self, x, x_body):
        # Body
        body_out = self.body_block(x_body)

        # RNN
        rnn_in = self.shuffle(x) # permute --> (batch_size, seq_len, n_vars) when batch_first=True
        rnn_out, _ = self.rnn(rnn_in)
        rnn_out = rnn_out[:, -1] # output of last sequence step (many-to-one)
        rnn_out = self.rnn_dropout(rnn_out)
        rnn_out += body_out
        
        # InceptionTime
        inception_out = self.inception_block(x)
        inception_out = self.gap(inception_out)
        inception_out += body_out
        
        # Head
        x = self.concat([rnn_out, inception_out])
        x = self.fc_dropout(x)
        x = self.fc(x)
        return x

class RNNInceptionTimeBody2(_RNNInceptionTimeBody2_Base):
    _cell = torch.nn.RNN
    
class GRUInceptionTimeBody2(_RNNInceptionTimeBody2_Base):
    _cell = torch.nn.GRU

class LSTMInceptionTimeBody2(_RNNInceptionTimeBody2_Base):
    _cell = torch.nn.LSTM

if __name__ == '__main__':
    x = torch.randn(1, 500, 16).permute(0, 2, 1)
    x_body = torch.randn(1, 3)
    model = RNNInceptionTimeBody2(c_in=16, c_out=4, seq_len=500, body_c_in=3)
    print(model(x, x_body).shape)
    