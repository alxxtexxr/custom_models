import torch

from tsai.models.InceptionTime import InceptionBlock
from tsai.models.RNNAttention import _TSTEncoder, get_activation_fn
from tsai.models.layers import Flatten, SigmoidRange

class InceptionTimeTransformer(torch.nn.Module):
    def __init__(
        self, c_out, seq_len,
        # InceptionTime parameters
        c_in, nf=32,
        # Transformer parameters
        encoder_layers:int=3, n_heads:int=16, d_k=None, d_v=None, d_ff=256, encoder_dropout=0.1, act="gelu", 
        # Head parameters
        fc_dropout=0., y_range=None,
    ):
        super().__init__()
        
        # InceptionTime layers
        self.inceptionblock = InceptionBlock(c_in, nf)
        
        # Transformer layers
        q_len = seq_len
        d_model = nf * 4
        self.encoder = _TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, dropout=encoder_dropout, activation=act, n_layers=encoder_layers)
        
        # Head layers
        self.head_nf = q_len * d_model
        self.head = self.create_head(self.head_nf, c_out, act=act, fc_dropout=fc_dropout, y_range=y_range)
    
    def create_head(self, nf, c_out, act="gelu", fc_dropout=0., y_range=None):
        layers = [get_activation_fn(act), Flatten()]
        if fc_dropout: layers += [torch.nn.Dropout(fc_dropout)]
        layers += [torch.nn.Linear(nf, c_out)]
        if y_range: layers += [SigmoidRange(*y_range)]
        return torch.nn.Sequential(*layers)    
    
    def forward(self, x):
        # InceptionTime forward
        x = self.inceptionblock(x)

        # Transformer forward
        x = x.transpose(2, 1)              # [bs x nvars x q_len] --> [bs x q_len x nvars]
        z = self.encoder(x)                # z: [bs x q_len x d_model]
        z = z.transpose(2, 1).contiguous() # z: [bs x d_model x q_len]
        
        # Head forward
        output = self.head(z) # output: [bs x c_out]
        return output
