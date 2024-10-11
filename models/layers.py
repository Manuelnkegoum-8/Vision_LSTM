import torch 
import torch.nn as nn
import torch.nn.functional as F 
from einops import rearrange,repeat
import math
from einops.layers.torch import Rearrange

class BlockDiagonalLinear(nn.Module):
    def __init__(self, input_size, output_size, num_blocks=4,bias=True):
        super().__init__()

        assert input_size % num_blocks == 0, "Input size must be divisible by the number of blocks"
        assert output_size % num_blocks == 0, "Output size must be divisible by the number of blocks"

        self.num_blocks = num_blocks
        self.block_in_size = input_size // num_blocks  # Input size of each block
        self.block_out_size = output_size // num_blocks  # Output size of each block

        # Create weight matrices for each block and bias
        self.weight = nn.Parameter(torch.randn(self.num_blocks, self.block_in_size, self.block_out_size))
        self.need_bias = bias
        if bias:
            self.bias = nn.Parameter(torch.randn(output_size))

        self._reset_parameters()

        
    def forward(self, x):
        batch_size,N,dim = x.size()
        x_blocks = x.contiguous().view(batch_size, N, self.num_blocks, self.block_in_size)
        # Apply block diagonal matrix multiplication
        out = torch.einsum('bnki,kij->bnkj', x_blocks, self.weight)
        out = out.contiguous().view(batch_size, N, dim)
        if self.need_bias:
            out = out + self.bias
        return out

    def _reset_parameters(self):
        # Xavier initialization for weights
        nn.init.xavier_uniform_(self.weight)
        if self.need_bias:
            nn.init.zeros_(self.bias.data)

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        # Causal padding: we pad only on the left side so that the convolution doesn't look ahead
        self.padding = (kernel_size - 1) * dilation
        # Causal convolution layer
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation,groups=in_channels)
    
    def forward(self, x):
        # Apply convolution and then trim the extra padding on the right (causal behavior)
        out = self.conv1d(x)
        return out[:, :, :-self.padding]  # Trimming the output to maintain causality

class SwishActivation(nn.Module):
    def __init__(self, beta =1.0, learnable=False):
        super(SwishActivation, self).__init__()
        # Define a learnable parameter if `learnable` is True
        if learnable:
            self.beta = nn.Parameter(torch.tensor(1.0))
        else:
            # Use a fixed parameter if not learnable
            self.beta = beta

    def forward(self, x):

        return x * torch.sigmoid(self.beta * x)

def equidistant_bias_init(tensor, a, b):
    """
    Initializes the bias of a given tensor with equidistant values between a and b.
    
    Args:
    - tensor: The bias tensor to initialize.
    - a: The starting value of the interval.
    - b: The ending value of the interval.
    """
    # Get the number of elements in the bias tensor
    num_params = tensor.numel()
    
    # Generate equidistant values between a and b
    equidistant_values = torch.linspace(a, b, num_params)
    
    # Assign the equidistant values to the tensor
    with torch.no_grad():
        tensor.copy_(equidistant_values)

def small_init_(param: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Fills the input Tensor with values according to the method described in Transformers without Tears: Improving
    the Normalization of Self-Attention - Nguyen, T. & Salazar, J. (2019), using a normal distribution.
    Adopted from https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py.
    """
    std = math.sqrt(2 / (5 * dim))
    torch.nn.init.normal_(param, mean=0.0, std=std)
    return param


def wang_init_(param: torch.Tensor, dim: int, num_blocks: int):
    """ Adopted from https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py. """
    std = 2 / num_blocks / math.sqrt(dim)
    torch.nn.init.normal_(param, mean=0.0, std=std)
    return param
class feedforward(nn.Module):
    def __init__(self,embed_dim=768,ff_hidden_dim=1024,dropout_rate = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.GELU(),
            nn.Linear(ff_hidden_dim, embed_dim),
        )
    def forward(self, x):
        outputs = self.mlp(x)
        return outputs

def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)

class patch_embedding(nn.Module):
    def __init__(self,height=224,width=224,n_channels=1,patch_size=16,dim=768):
        super().__init__()
        
        assert height%patch_size==0 and width%patch_size==0 ,"Height and Width should be multiples of patch size wich is {0}".format(patch_size)
        #self.class_token = nn.Parameter(torch.randn(1, 1, dim))
        self.patch_size = patch_size
        self.n_patchs = height*width//(patch_size**2)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.n_patchs, dim))
        self.projection = nn.Conv2d(n_channels, dim, kernel_size=patch_size, stride=patch_size)


    def forward(self, x):
        #x bs,h,w,c
        #projection on the dim of model
        x = self.projection(x)
        bs,dim,h,w = x.size()
        x = x.permute(0,2,3,1).contiguous().view(bs,h*w,dim)
        outputs = x + self.pos_embedding
        return outputs


class Conv2DLayer(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        # Reshape input to (batch, dim, m, m)
        batch,dim,L = x.size()
        m = int(math.sqrt(L))
        x = x.contiguous().view(batch, dim, m, m)

        # Apply convolution
        x = super().forward(x)
        x = x.contiguous().view(batch, dim, L)
        return x