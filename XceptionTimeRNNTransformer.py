import torch

from tsai.models.XceptionTime import XceptionBlock
from tsai.models.RNNAttention import _TSTEncoder, get_activation_fn
from tsai.models.layers import Flatten, SigmoidRange

def ifnone(a, b):
    return b if a is None else a

class _XceptionTimeRNNTransformer_Base(torch.nn.Module):
    def __init__(
        self, c_in, c_out, seq_len,
        # XceptionTime  
        nf=16, nb_filters=None, #adaptive_size=50,
        # RNN
        hidden_size=128, rnn_layers=1, bias=True, rnn_dropout=0, bidirectional=False,
        # Transformer
        encoder_layers:int=3, n_heads:int=16, d_k=None, d_v=None, d_ff=256, encoder_dropout=0.1, act="gelu", 
        # Head
        fc_dropout=0., y_range=None,
    ):
        super().__init__()
        
        # XceptionTime
        nf = ifnone(nf, nb_filters)
        self.xception_block = XceptionBlock(c_in, nf)
        xception_head_nf = nf * 32
        
        # RNN
        q_len = seq_len
        self.rnn = self._cell(xception_head_nf, hidden_size, num_layers=rnn_layers, bias=bias, batch_first=True, dropout=rnn_dropout, 
                              bidirectional=bidirectional)
        
        # Transformer
        d_model = hidden_size * (1 + bidirectional)
        self.encoder = _TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, dropout=encoder_dropout, activation=act, n_layers=encoder_layers)
        
        # Head
        self.head_nf = q_len * d_model
        self.head = self.create_head(self.head_nf, c_out, act=act, fc_dropout=fc_dropout, y_range=y_range)
    
    def create_head(self, nf, c_out, act="gelu", fc_dropout=0., y_range=None):
        layers = [get_activation_fn(act), Flatten()]
        if fc_dropout: layers += [torch.nn.Dropout(fc_dropout)]
        layers += [torch.nn.Linear(nf, c_out)]
        if y_range: layers += [SigmoidRange(*y_range)]
        return torch.nn.Sequential(*layers)    
    
    def forward(self, x):
        # XceptionTime
        x = self.xception_block(x)

        # RNN
        x = x.transpose(2, 1)   # [bs x nvars x q_len] --> [bs x q_len x nvars]
        output, _ = self.rnn(x) # output from all sequence steps: [bs x q_len x hidden_size * (1 + bidirectional)]
        
        # Transformer
        z = self.encoder(output)           # z: [bs x q_len x d_model]
        z = z.transpose(2, 1).contiguous() # z: [bs x d_model x q_len]
        
        # Head
        output = self.head(z) # output: [bs x c_out]
        return output

class XceptionTimeRNNTransformer(_XceptionTimeRNNTransformer_Base):
    _cell = torch.nn.RNN
    
class XceptionTimeGRUTransformer(_XceptionTimeRNNTransformer_Base):
    _cell = torch.nn.GRU

class XceptionTimeLSTMTransformer(_XceptionTimeRNNTransformer_Base):
    _cell = torch.nn.LSTM