
import torch.nn as nn

from tsai.models.layers import Module, Permute, ConvBlock, SqueezeExciteBlock, GAP1d, Concat, noop
from pykan.kan import KAN

class MLSTMFCNKAN(Module):
    def __init__(self, c_in, c_out, seq_len=None, hidden_size=100, rnn_layers=1, bias=True, cell_dropout=0, rnn_dropout=0.8, bidirectional=False, shuffle=True, 
                 fc_dropout=0., conv_layers=[128, 256, 128], kss=[7, 5, 3], se=0,
                 # KAN parameters
                 kan_layers=[1], kan_grid=1, kan_k=1, kan_device='cuda'):
        
        if shuffle: assert seq_len is not None, 'need seq_len if shuffle=True'
        
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.conv_layers = conv_layers
            
        # RNN
        self._cell = nn.LSTM
        self.rnn = self._cell(seq_len if shuffle else c_in, hidden_size, num_layers=rnn_layers, bias=bias, batch_first=True, 
                              dropout=cell_dropout, bidirectional=bidirectional)
        self.rnn_dropout = nn.Dropout(rnn_dropout) if rnn_dropout else noop
        self.shuffle = Permute(0,2,1) if not shuffle else noop # You would normally permute x. Authors did the opposite.
        
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
        self.fc = nn.Linear(hidden_size * (1 + bidirectional) + conv_layers[-1], c_out)
        
        # KAN
        self.kan = KAN(width=([hidden_size * (1 + bidirectional) + conv_layers[-1]] + kan_layers + [c_out]), 
                       grid=kan_grid, 
                       k=kan_k,
                       device=kan_device) # !!! refactor this code !!!

    def forward(self, x):  
        # RNN
        rnn_input = self.shuffle(x) # permute --> (batch_size, seq_len, n_vars) when batch_first=True
        output, _ = self.rnn(rnn_input)
        last_out = output[:, -1] # output of last sequence step (many-to-one)
        last_out = self.rnn_dropout(last_out)
        
        # FCN
        x = self.convblock1(x)
        x = self.se1(x)
        x = self.convblock2(x)
        x = self.se2(x)
        x = self.convblock3(x)
        x = self.gap(x)

        # Concat
        x = self.concat([last_out, x])
        x = self.fc_dropout(x)
        # x = self.fc(x)
        x = self.kan(x)
        return x