import torch
import torch.nn as nn
class VariationalDropout(nn.Module):
    """
    Variational (locked) dropout module.
    Applies the same dropout mask across all time steps.
    """
    def __init__(self, dropout):
        super(VariationalDropout, self).__init__()
        self.dropout = dropout

    def forward(self, x):
        if not self.training or self.dropout == 0:
            return x
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1 - self.dropout)
        mask = mask / (1 - self.dropout)
        mask = mask.expand_as(x)
        return x * mask
class LM_model(nn.Module):
    def __init__(self, model, hidden_dim, emb_dim, output_size, var_dropout, pad_index, emb_dropout=0.6, out_dropout=0.6,tie_weights=True):
        super(LM_model, self).__init__()
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.output_size = output_size
        self.pad_index = pad_index
        self.model = model

        if tie_weights:
            if emb_dim != hidden_dim:
                raise ValueError('When using the tied flag, emb_dim must be equal to hidden_dim')
        
        self.embedding = nn.Embedding(output_size, emb_dim, padding_idx=pad_index)
            
        if model == 'RNN':
            self.rnn = nn.RNN(emb_dim, hidden_dim, batch_first=True)
        elif model == 'LSTM':
            self.rnn = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        
        self.rnn.to('cuda')
        self.rnn.flatten_parameters()
            
        if var_dropout:
            self.emb_drop = VariationalDropout(emb_dropout)
            self.out_drop = VariationalDropout(out_dropout)
            
        self.linear = nn.Linear(hidden_dim, output_size)

        if tie_weights:
            self.linear.weight = self.embedding.weight
        
    def forward(self, x):
        x = self.embedding(x)
        
        if hasattr(self, 'emb_drop'):
            x = self.emb_drop(x)
        
        out, _ = self.rnn(x)
        
        if hasattr(self, 'out_drop'):
            out = self.out_drop(out)
      
        out = self.linear(out).permute(0,2,1)
        return out