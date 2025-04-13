import fire
import torch

from tsai.imports import default_device
from tsai.models.RNNAttention import _TSTEncoder, get_activation_fn
from tsai.models.HydraPlus import HydraBackbonePlus
from tsai.models.layers import Concat, Flatten, SigmoidRange

noop = torch.nn.Sequential()

class RNNTransformerHydra(torch.nn.Module):
    def __init__(self,
                 # RNN-Transformer  
                 c_in, c_out, seq_len, hidden_size=128, rnn_layers=1, bias=True, rnn_dropout=0, bidirectional=False,
                 encoder_layers=3, n_heads:int=16, d_k=None, d_v=None, 
                 d_ff=256, encoder_dropout=0.1, act="gelu", fc_dropout:float=0.,
                 y_range=None,
                 # Hydra  
                 k = 8, g = 64, max_c_in = 8, clip=True, device=default_device(), zero_init=True):
        super().__init__()
        
        # RNN
        self.rnn = torch.nn.RNN(c_in, hidden_size, num_layers=rnn_layers, bias=bias, batch_first=True, dropout=rnn_dropout, 
                              bidirectional=bidirectional)
        
        # Transformer
        q_len = seq_len
        d_model = hidden_size * (1 + bidirectional)
        self.encoder = _TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, dropout=encoder_dropout, activation=act, n_layers=encoder_layers)
        
        # RNN-Transformer Head
        self.rnnattn_head_nf = q_len * d_model
        self.rnnattn_head = self.create_rnnattn_head(nf=self.rnnattn_head_nf, c_out=hidden_size, act=act, fc_dropout=fc_dropout, y_range=y_range)
        
        # Hydra
        self.hydra_backbone = HydraBackbonePlus(c_in, c_out, seq_len, k=k, g=g, max_c_in=max_c_in, clip=clip, device=device, zero_init=zero_init)
        
        # Head
        self.concat = Concat()
        self.fc_dropout = torch.nn.Dropout(fc_dropout) if fc_dropout else noop
        self.fc = torch.nn.Linear(hidden_size + self.hydra_backbone.num_features, c_out)
    
    def create_rnnattn_head(self, nf, c_out, act="gelu", fc_dropout=0., y_range=None):
        layers = [get_activation_fn(act), Flatten()]
        if fc_dropout: layers += [torch.nn.Dropout(fc_dropout)]
        layers += [torch.nn.Linear(nf, c_out)]
        if y_range: layers += [SigmoidRange(*y_range)]
        return torch.nn.Sequential(*layers)    
    
    def forward(self, x):
        # RNN
        rnn_input = x.transpose(2,1)    # [bs x nvars x q_len] --> [bs x q_len x nvars]
        output, _ = self.rnn(rnn_input) # output from all sequence steps: [bs x q_len x hidden_size * (1 + bidirectional)]
        
        # Transformer
        z = self.encoder(output)          # z: [bs x q_len x d_model]
        z = z.transpose(2,1).contiguous() # z: [bs x d_model x q_len]
        
        # RNN-Transformer Head
        rnnattn_output = self.rnnattn_head(z)
        
        # Hydra
        x = self.hydra_backbone(x)
        
        # Concat
        x = self.concat([rnnattn_output, x])
        x = self.fc_dropout(x)
        x = self.fc(x)
        return x

class RNNTransformerHydra_v0(torch.nn.Module):
    def __init__(self,
                 # RNN-Transformer  
                 c_in, c_out, seq_len, hidden_size=128, rnn_layers=1, bias=True, rnn_dropout=0, bidirectional=False,
                 encoder_layers=3, n_heads:int=16, d_k=None, d_v=None, 
                 d_ff=256, encoder_dropout=0.1, act="gelu", fc_dropout:float=0.,
                 y_range=None,
                 # Hydra  
                 k = 8, g = 64, max_c_in = 8, clip=True, device=default_device(), zero_init=True):
        super().__init__()
        
        # RNN
        self.rnn = torch.nn.RNN(c_in, hidden_size, num_layers=rnn_layers, bias=bias, batch_first=True, dropout=rnn_dropout, 
                              bidirectional=bidirectional)
        
        # Transformer
        q_len = seq_len
        d_model = hidden_size * (1 + bidirectional)
        self.encoder = _TSTEncoder(q_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, dropout=encoder_dropout, activation=act, n_layers=encoder_layers)
        
        # RNN-Transformer Head
        self.rnnattn_head = torch.nn.Linear(q_len * d_model, hidden_size)
        
        # Hydra
        self.hydra_backbone = HydraBackbonePlus(c_in, c_out, seq_len, k=k, g=g, max_c_in=max_c_in, clip=clip, device=device, zero_init=zero_init)
        
        # Head
        self.concat = Concat()
        self.fc_dropout = torch.nn.Dropout(fc_dropout) if fc_dropout else noop
        self.fc = torch.nn.Linear(hidden_size + self.hydra_backbone.num_features, c_out)
    
    def forward(self, x):
        # RNN
        rnn_input = x.transpose(2,1)    # [bs x nvars x q_len] --> [bs x q_len x nvars]
        output, _ = self.rnn(rnn_input) # output from all sequence steps: [bs x q_len x hidden_size * (1 + bidirectional)]
        
        # Transformer
        z = self.encoder(output)          # z: [bs x q_len x d_model]
        z = z.transpose(2,1).contiguous() # z: [bs x d_model x q_len]
        
        # RNN-Transformer Head
        B, D, L = z.shape
        rnnattn_output = self.rnnattn_head(z.reshape(B, D*L))
        
        # Hydra
        x = self.hydra_backbone(x)
        
        # Head
        x = self.concat([rnnattn_output, x])
        x = self.fc_dropout(x)
        x = self.fc(x)
        return x