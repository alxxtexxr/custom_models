import torch

from tsai.models.InceptionTime import InceptionBlock
from tsai.models.layers import Permute, Concat, GAP1d

noop = torch.nn.Sequential()

def ifnone(a, b):
    return b if a is None else a

class _RNNInceptionTimeXFX_Base(torch.nn.Module):
    def __init__(self,
                 c_in, c_out, seq_len,
                 # XFX 
                 xfx_c_in, xfx_feat_len, xfx_c_out=128, xfx_hidden_sizes=[256],
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
        self.fc = torch.nn.Linear(hidden_size * (1 + bidirectional) + inception_head_nf + xfx_c_out, c_out)

        # XFX
        xfx_input_size = xfx_c_in * xfx_feat_len
        xfx_layers = []
        for xfx_hidden_size in xfx_hidden_sizes:
            xfx_layers.append(torch.nn.Linear(xfx_input_size, xfx_hidden_size))
            xfx_layers.append(torch.nn.ReLU())
            xfx_layers.append(torch.nn.Dropout(0.1)) # Dropout for regularization
            xfx_input_size = xfx_hidden_size
        xfx_layers.append(torch.nn.Linear(xfx_input_size, xfx_c_out))

        self.xfx_block = torch.nn.Sequential(*xfx_layers)

    def forward(self, x, x_xfx):
        # RNN
        rnn_in = self.shuffle(x) # permute --> (batch_size, seq_len, n_vars) when batch_first=True
        rnn_out, _ = self.rnn(rnn_in)
        rnn_out = rnn_out[:, -1] # out of last sequence step (many-to-one)
        rnn_out = self.rnn_dropout(rnn_out)
        
        # InceptionTime
        inception_out = self.inception_block(x)
        inception_out = self.gap(inception_out)

        # XFX
        N, _, _ = x_xfx.shape
        x_xfx = x_xfx.reshape(N, -1)
        xfx_out = self.xfx_block(x_xfx)

        # Head
        out = self.concat([rnn_out, inception_out, xfx_out])
        out = self.fc_dropout(out)
        out = self.fc(out)
        return out

class RNNInceptionTimeXFX(_RNNInceptionTimeXFX_Base):
    _cell = torch.nn.RNN
    
class GRUInceptionTimeXFX(_RNNInceptionTimeXFX_Base):
    _cell = torch.nn.GRU

class LSTMInceptionTimeXFX(_RNNInceptionTimeXFX_Base):
    _cell = torch.nn.LSTM

if __name__ == '__main__':
    x = torch.randn(1, 500, 16).permute(0, 2, 1)
    x_xfx = torch.randn(1, 16, 11).permute(0, 2, 1)
    model = RNNInceptionTimeXFX(c_in=16, c_out=4, seq_len=500, 
                                xfx_c_in=11, xfx_c_out=128, xfx_feat_len=16)
    print(model(x, x_xfx).shape)
    