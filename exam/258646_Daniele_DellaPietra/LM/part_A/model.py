import torch
import torch.nn as nn
class LM_model(nn.Module):
    def __init__(self, model, hidden_dim, emb_dim, output_size, use_dropout, pad_index, emb_dropout=0.1, out_dropout=0.1):
        super(LM_model, self).__init__()
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.output_size = output_size
        self.pad_index = pad_index
        self.model = model
        
        self.embedding = nn.Embedding(output_size, emb_dim, padding_idx=pad_index)
            
        if model == 'RNN':
            self.rnn = nn.RNN(emb_dim, hidden_dim, batch_first=True)
        elif model == 'LSTM':
            self.rnn = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
            
        if use_dropout:
            self.emb_drop = nn.Dropout(emb_dropout)
            self.out_drop = nn.Dropout(out_dropout)
            
        self.linear = nn.Linear(hidden_dim, output_size)
        
    def forward(self, x):
        x = self.embedding(x)
        
        if hasattr(self, 'emb_drop'):
            x = self.emb_drop(x)
        
        out, _ = self.rnn(x)
        
        if hasattr(self, 'out_drop'):
            out = self.out_drop(out)
      
        out = self.linear(out).permute(0,2,1)
        return out