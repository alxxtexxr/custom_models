import torch
import torch.nn as nn
import torch.nn.functional as F

from tsai.models.InceptionTime import InceptionBlock
from tsai.models.layers import Permute, Concat, GAP1d

noop = torch.nn.Sequential()

def ifnone(a, b):
    return b if a is None else a

class _RNNInceptionTimeMoEEncoder_Base(torch.nn.Module):
    def __init__(self,
                 c_in, 
                 # c_out, 
                 seq_len,
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
        # self.fc_dropout = torch.nn.Dropout(fc_dropout) if fc_dropout else noop
        # self.fc = torch.nn.Linear(hidden_size * (1 + bidirectional) + inception_head_nf, c_out)
    
    def forward(self, x):
        # RNN
        rnn_input = self.shuffle(x) # permute --> (batch_size, seq_len, n_vars) when batch_first=True
        output, _ = self.rnn(rnn_input)
        last_out = output[:, -1] # output of last sequence step (many-to-one)
        last_out = self.rnn_dropout(last_out)
        
        # InceptionTime
        x = self.inception_block(x)
        x = self.gap(x)
        
        # Head
        x = self.concat([last_out, x])
        # x = self.fc_dropout(x)
        # x = self.fc(x)
        return x

class RNNInceptionTimeMoEEncoder(_RNNInceptionTimeMoEEncoder_Base):
    _cell = torch.nn.RNN
    
class GRUInceptionTimeMoEEncoder(_RNNInceptionTimeMoEEncoder_Base):
    _cell = torch.nn.GRU

class LSTMInceptionTimeMoEEncoder(_RNNInceptionTimeMoEEncoder_Base):
    _cell = torch.nn.LSTM

# class Gate(torch.nn.Module):
#     def __init__(self, c_in, n_expert):
#         super(Gate, self).__init__()
#         self.fc = torch.nn.Linear(c_in, n_expert)

#     def forward(self, x):
#         x = self.fc(x)
#         x = F.softmax(x, dim=-1)
#         return x
    
class Gate(nn.Module):
    def __init__(self, c_in, n_expert, hidden_size):
        super(Gate, self).__init__()
        self.fc1 = nn.Linear(c_in, hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_expert)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=-1) # Softmax to get the gating probabilities
        return x

class RNNInceptionTimeMoE(torch.nn.Module):
    def __init__(self, c_in, c_out, seq_len, 
                 # Experts
                 n_expert=3, hidden_size=1024,
                 # RNN
                 rnn_hidden_size=128, bidirectional=True,
                 # InceptionTime
                 inception_nf=32,
                 ):
        super(RNNInceptionTimeMoE, self).__init__()
        
        # Shared Encoder: RNNInceptionTime
        self.shared_encoder = RNNInceptionTimeMoEEncoder(c_in=c_in, seq_len=seq_len, hidden_size=rnn_hidden_size, bidirectional=bidirectional)
        shared_encoder_nf = (rnn_hidden_size * (1 + bidirectional) + (inception_nf * 4))

        # Experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(shared_encoder_nf, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, c_out),
            ) for _ in range(n_expert)
        ])
        
        # Gate
        self.gate = Gate(c_in=shared_encoder_nf, n_expert=n_expert, hidden_size=hidden_size)

    def forward(self, x):
        emb = self.shared_encoder(x) # [batch_size, shared_encoder_nf]
        # print(f'{emb.shape=}')
        
        gate_out = self.gate(emb) # [batch_size, num_experts]
        # print(f'{gate_out.shape=}')

        expert_outs = [expert(emb) for expert in self.experts]
        expert_outs = torch.stack(expert_outs, dim=1) # [batch_size, n_expert, c_out]
        # print(f'{expert_outs.shape=}')

        final_out = torch.bmm(gate_out.unsqueeze(1), expert_outs).squeeze(1) # [batch_size, c_out]
        # print(f'{final_out.shape=}')
        return final_out

if __name__ == '__main__':
    x = torch.randn(1, 500, 16).permute(0, 2, 1)
    model = RNNInceptionTimeMoE(c_in=16, c_out=4, seq_len=500)
    print(model(x).shape)
    