import torch

from tsai.models.RNNAttention import _TSTEncoder
from tsai.models.layers import ConvBlock, SqueezeExciteBlock, GAP1d, Concat

noop = torch.nn.Sequential()

def ifnone(a, b):
    return b if a is None else a

class RNNTransformerFCN(torch.nn.Module):
    def __init__(self, 
                 # RNN-Transformer  
                 c_in, c_out, seq_len, hidden_size=128, rnn_layers=1, bias=True, rnn_dropout=0, bidirectional=False,
                 encoder_layers=3, n_heads:int=16, d_k=None, d_v=None, 
                 d_ff=256, encoder_dropout=0.1, act="gelu",
                 # FCN
                 fc_dropout=0., conv_layers=[128, 256, 128], kss=[7, 5, 3], se=0
                 ):
        super().__init__()
        
        q_len = seq_len
        
        # RNN
        self.rnn = torch.nn.RNN(c_in, hidden_size, num_layers=rnn_layers, bias=bias, batch_first=True, dropout=rnn_dropout, 
                              bidirectional=bidirectional)
        self.gap_attn = GAP1d(1)
        
        # Transformer Encoder
        d_model = hidden_size * (1 + bidirectional)
        self.encoder = _TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, dropout=encoder_dropout, activation=act, n_layers=encoder_layers)
        
        # FCN
        assert len(conv_layers) == len(kss)
        self.convblock1 = ConvBlock(c_in, conv_layers[0], kss[0])
        self.se1 = SqueezeExciteBlock(conv_layers[0], se) if se != 0 else noop
        self.convblock2 = ConvBlock(conv_layers[0], conv_layers[1], kss[1])
        self.se2 = SqueezeExciteBlock(conv_layers[1], se) if se != 0 else noop
        self.convblock3 = ConvBlock(conv_layers[1], conv_layers[2], kss[2])
        self.gap = GAP1d(1)
        
        # Concat
        self.concat = Concat()
        self.fc_dropout = torch.nn.Dropout(fc_dropout) if fc_dropout else noop
        self.fc = torch.nn.Linear(hidden_size * (1 + bidirectional) + conv_layers[-1], c_out)
    
    def forward(self, x):
        # RNN
        rnn_input = x                   # [bs x q_len x nvars]
        output, _ = self.rnn(rnn_input) # output from all sequence steps: [bs x q_len x hidden_size * (1 + bidirectional)]
        
        # Transformer Encoder
        z = self.encoder(output)          # z: [bs x q_len x d_model]
        z = z.transpose(2,1).contiguous() # z: [bs x d_model x q_len]
        z = self.gap_attn(z)
        
        # FCN
        x = x.transpose(2, 1)
        x = self.convblock1(x)
        x = self.se1(x)
        x = self.convblock2(x)
        x = self.se2(x)
        x = self.convblock3(x)
        x = self.gap(x)
        
        # Concat
        x = self.concat([z, x])
        x = self.fc_dropout(x)
        x = self.fc(x)
        return x