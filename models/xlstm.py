import torch 
import torch.nn as nn
import torch.nn.functional as F 
from .mlstm import mLSTM_block
from .layers import *

class ViL_layer(nn.Module):
    def __init__(self,dim,heads,mlp_dim):
        super(ViL_layer,self).__init__()
    
        #self.reset_parameters()
        self.first_norm = nn.LayerNorm(dim,dim)
        self.swish = nn.SiLU()
        self.id = nn.Identity()
        self.block = mLSTM_block(dim,heads)
        self.proj_1 = nn.Linear(dim,dim)
        self.proj_2 = nn.Linear(dim,dim)
        self.proj_3 = nn.Linear(dim,dim)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self,x,flip=False):
        out = self.first_norm(x)
        right = self.swish(self.proj_1(out))
        left = self.proj_2(out)
        if flip:
            left = torch.flip(left,dims=(1,))
        left = self.block(left,use_conv=True)
        if flip:
            left = torch.flip(left,dims=(1,))
        out = left*out
        out = self.dropout(self.proj_3(out))
        return out+x


class ViL(nn.Module):
    def __init__(self,config=None):
        super(ViL,self).__init__()
        self.layers = nn.ModuleList()
        for i in range(config.m_layers):
            block = ViL_layer(config.dim,config.heads,config.mlp_dim)
            self.layers.append(block)
        self.patch_embedding = patch_embedding( height=config.height,
                                                width=config.width,
                                                n_channels=config.channels,
                                                patch_size=config.patch_size,
                                                dim=config.dim)
        #self.reset_parameters()

        self.fc = nn.Linear(config.dim,config.n_classes)

    def forward(self,img):
        x = self.patch_embedding(img) # bs*n_p*dim
        for layer in self.layers:
            x = layer(x)
        out = 0.5*(x[:,0,:]+x[:,-1,:])
        out = self.fc(out)
        return out
