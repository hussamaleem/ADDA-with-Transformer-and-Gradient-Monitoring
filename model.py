import torch.nn as nn
import torch
import math


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_seq_len):
        super(PositionalEncoding, self).__init__()
        self.model_dim = model_dim
        self.pos_encoding = self.generate_positional_encoding(max_seq_len, model_dim).to('cuda')
    
    def generate_positional_encoding(self, max_seq_len, model_dim):
        pe = torch.zeros(max_seq_len, model_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(1500.0) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe
        
    def forward(self, x):
        x = x * math.sqrt(self.model_dim)
        seq_len = x.size(1)
        x = x + self.pos_encoding[:, :seq_len, :]
        return x

class TransformerEncoder(nn.Module):
    def __init__(self,
                 pos_enc,
                 model_dim,
                 max_seq_len,
                 num_head,
                 num_layers,
                 drop_out_ratio):
        super(TransformerEncoder, self).__init__()
        
        self.pos_enc = pos_enc
        if self.pos_enc:
            self.pos_encoding = PositionalEncoding(model_dim, max_seq_len)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=model_dim,
                                       nhead=num_head,
                                       batch_first=True,
                                       dim_feedforward = model_dim*2,
                                       dropout=drop_out_ratio
                                       ),
            num_layers=num_layers
        )
        
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(16*model_dim, model_dim))

    def forward(self, src):
        
        if self.pos_enc:
            src = self.pos_encoding(src)
        out_1 = self.transformer_encoder(src)
        out_1 = out_1.permute(0,2,1)
        out = self.projection(out_1)
    
        return out
    
class Transformer_RUL_Predictor(nn.Module):
    def __init__(self,
                 hidden_size,
                 model_dim,
                 ):
        
        super(Transformer_RUL_Predictor,self).__init__()
        
        self.decoder = nn.Sequential(nn.Linear(model_dim, hidden_size),
                                     nn.LeakyReLU(),
                                     nn.Linear(hidden_size, hidden_size//2),
                                     nn.LeakyReLU(),
                                     nn.Linear(hidden_size//2,hidden_size//4),
                                     nn.LeakyReLU(),
                                     nn.Linear(hidden_size//4,1)
                                     )
        
    def forward(self,src):

        out = self.decoder(src)
        
        return out

class Transformer_Model(nn.Module):
    
    def __init__(self,model_dim,
                 num_head,
                 num_layers, 
                 max_seq_len,
                 pos_enc,
                 hidden_size,
                 drop_out_ratio):
        
        super(Transformer_Model,self).__init__()
        
        self.pos_enc = pos_enc
        
        self.transformer = TransformerEncoder(pos_enc=self.pos_enc,
                                              model_dim=model_dim,
                                              num_head=num_head,
                                              num_layers=num_layers,
                                              max_seq_len=max_seq_len,
                                              drop_out_ratio=drop_out_ratio)
        
        self.rul_predictor = Transformer_RUL_Predictor(hidden_size=hidden_size,
                                                       model_dim=model_dim,
                                                       )
        
        self.linear_embedding = nn.Linear(14,model_dim)
        
    def forward(self,data):
        
        embedd = self.linear_embedding(data)
        out_1 = self.transformer(embedd)
        out = self.rul_predictor(out_1)

        return out
    
'---------------------------------------------------------------------------------'