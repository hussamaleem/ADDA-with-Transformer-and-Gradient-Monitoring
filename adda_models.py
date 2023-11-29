import torch.nn as nn
import copy

class ADDAEncoder():
    def __init__(self,base_model,src_encoder):
        self.base_model = copy.deepcopy(base_model)
        self.src_encoder = src_encoder
        
    def get_encoder(self):
        
        encoder = nn.Sequential(self.base_model.linear_embedding,
                                self.base_model.transformer)
        if self.src_encoder:
            for param in encoder.parameters():
                param.requires_grad = False
        
        return encoder

class ADDARULPredictor():
    def __init__(self,base_model):
        self.base_model = copy.deepcopy(base_model)
        
    def get_decoder(self):
        
        decoder = self.base_model.rul_predictor
        
        for param in decoder.parameters():
            param.requires_grad = False
        return decoder
    
class Discriminator(nn.Module):
    def __init__(
            self,
            in_features,
            out_features
            ):
        
        super(Discriminator,self).__init__()
        
        self.discr = nn.Sequential(nn.Linear(in_features,out_features),
                                   nn.LeakyReLU(),
                                   nn.Linear(out_features,out_features//2),
                                   nn.Linear(out_features//2,2))
        
    def forward(self,features):
        
        out = self.discr(features)
        
        
        return out