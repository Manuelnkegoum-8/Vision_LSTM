import torch 
import torch.nn as nn
import torch.nn.functional as F 
from .mlstm import mLSTM_block
from .layers import *


class ViL(nn.Module):
    def __init__(self,config=None):
        super(ViL,self).__init__()
        self.layers = nn.ModuleList()
        for i in range(config.m_layers):
            block = mLSTM_block(config.dim,config.heads)
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
        flip = 1
        for layer in self.layers:
            x = layer(x,use_conv=True,flip=flip)
            flip = 1 - flip
        out = 0.5*(x[:,0,:]+x[:,-1,:])
        out = self.fc(out)
        return out
