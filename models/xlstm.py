import torch 
import torch.nn as nn
import torch.nn.functional as F 
from .mlstm import mLSTM_block
from .layers import *


class ViL(nn.Module):
    def __init__(self,config=None):
        super(ViL,self).__init__()
        self.layers = nn.ModuleList()
        self.classif = config.classif
        for i in range(config.m_layers):
            block = mLSTM_block(config.dim,config.qk_size)
            self.layers.append(block)
        self.patch_embedding = patch_embedding( height=config.height,
                                                width=config.width,
                                                n_channels=config.channels,
                                                patch_size=config.patch_size,
                                                dim=config.dim)
        #self.reset_parameters()
        if config.classif =='bilateral_avg':
            self.fc = nn.Linear(config.dim,config.n_classes)
        elif config.classif == 'bilateral_concat':
            self.fc = nn.Linear(config.dim*2,config.n_classes)
        else:
            raise Exception(f"Classification mode should be one of bilateral_avg or bilateral_concat but got {config.classif}")

    def forward(self,img):
        x = self.patch_embedding(img) # bs*n_p*dim
        flip = 0
        for layer in self.layers:
            x = layer(x,flip=flip)
            flip = 1 - flip

        if self.classif =='bilateral_avg':
            out = 0.5*(x[:,0,:]+x[:,-1,:])
        else :
            out = torch.cat((x[:,0,:],x[:,-1,:]),dim=-1)
        out = self.fc(out)
        return out
    

    def no_weight_decay(self):
            no_decay = [p for n, p in self.named_parameters() if 'pos_embedding' in n]
            decay = [p for n, p in self.named_parameters() if 'pos_embedding' not in n]
            return no_decay, decay
