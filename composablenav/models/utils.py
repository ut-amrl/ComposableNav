import sys
import math 
import torch 
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from functools import partial
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
allow_ops_in_compiled_graph()

class SinusoidalPosEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim 
    
    def forward(self, t):
        """
        t: batch_size
        output: batch_size x dim
        """
        half_dim = self.dim // 2
        embed = math.log(10000) / (half_dim - 1)
        embed = torch.exp(-torch.arange(half_dim, device=t.device) * embed)
        embed = t[:, None] * embed[None, :]
        return torch.cat((embed.sin(), embed.cos()), dim=-1)
    
    
class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones([1, dim, 1])) # Batch x States x Horizon
        self.b = nn.Parameter(torch.zeros([1, dim, 1]))
        
    def forward(self, x):
        """
        x: batch_size x state_chn x horizon
        """
        mu = x.mean(dim=1, keep_dim=True)
        var = x.var(dim=1, keep_dim=True)
        return self.g * (x - mu) / torch.sqrt(var + self.eps) + self.b 

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones([1, dim, 1]))
        
    def forward(self, x):
        """
        x: batch_size x state_chn x horizon
        """

        return F.normalize(x, p=2, dim=1) * (x.shape[1] ** 0.5) * self.g

class PreNorm(nn.Module):
    def __init__(self, norm_fn, fn):
        super().__init__()
        self.norm_fn = norm_fn
        self.fn = fn
    
    def forward(self, x):
        return self.fn(self.norm_fn(x))

class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, dim_head=32, num_heads=4, dropout=0., max_seq_len=128):
        super().__init__()
        # treat as an image
        hidden_dim = dim_head * num_heads
        self.num_heads = num_heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, kernel_size=1)
        self.out = nn.Conv1d(hidden_dim, dim, kernel_size=1) 
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        assume x is pre-normalized
        x: batch_size x state_chn x horizon
        qkv: batch_size x num_heads x horizon x hidden_dim
        """
        # assume x is prenormed
        qkv = self.to_qkv(x)
        qkv = rearrange(qkv, "b (h d) l -> b h l d", h=self.num_heads)
        q, k, v = qkv.chunk(3, dim=-1)
        
        sim = torch.einsum("b h i d, b h j d -> b h i j", q, k) * q.shape[-1] ** -0.5 # B x head x horizon x horizon
        if mask is not None:
            mask_mat = repeat(mask, 'b j -> b h i j', h=self.num_heads, i=sim.shape[-2]) # convert to matrix with last column to be masked by 0
            sim.masked_fill_(mask_mat.bool() == 0, -float("inf")) 
        attn = F.softmax(sim, dim=-1)
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h s d -> b (h d) s")
        out = self.dropout(out)
        
        return self.out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.out = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout) 
        )
        
    def forward(self, x):
        return self.out(x)
    
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, *args, **kwargs):
        return x + self.fn(x, *args, **kwargs)

class DownSample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.down = nn.Conv1d(dim, dim, kernel_size=4, stride=2, padding=1)
        
    def forward(self, x):
        """
        x: batch_size x state_chn x horizon
        """
        return self.down(x)
    
class UpSample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1)
        
    def forward(self, x):
        """
        x: batch_size x state_chn x horizon
        """
        return self.up(x)
    
"""
Below are for Context Encoders
"""
class ResDoubleConv2d(nn.Module):
    def __init__(self, input_chn, hidden_chn, num_groups, dropout):
        super(ResDoubleConv2d, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(input_chn, hidden_chn, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, hidden_chn),
            nn.Mish(),
            nn.Dropout2d(dropout)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden_chn, hidden_chn, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups, hidden_chn),
            nn.Mish(),
            nn.Dropout2d(dropout)
        )
        self.residual_conv = nn.Conv2d(input_chn, hidden_chn, kernel_size=1) if input_chn != hidden_chn else nn.Identity()
        
    def forward(self, x):
        """
        x: N x C x H x W
        """
        out = self.block1(x)
        out = self.block2(out)
        return self.residual_conv(x) + out

class DownSample2D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.down = nn.Conv2d(dim, dim, kernel_size=4, stride=2, padding=1)
        
    def forward(self, x):
        """
        x: batch_size x state_chn x horizon
        """
        return self.down(x)
    
class SquareRearrange2D(nn.Module):
    def __init__(self, fn):
        super(SquareRearrange2D, self).__init__()
        self.fn = fn
    
    def forward(self, x):
        x = rearrange(x, "n c h w -> n c (h w)")
        x = self.fn(x)

        last_dim = x.shape[-1]
        h = int(last_dim ** 0.5)
        if h * h != last_dim:
            raise ValueError("The last dimension is not a perfect square.")

        return rearrange(x, "n c (h w) -> n c h w", h=h)
    
class OccupancyGridEncoder(nn.Module):
    def __init__(self, cfg, input_dim, hidden_dim, output_dim, dropout, 
                 dim_head, num_heads, num_groups, grid_size=32,
                 norm_fn="RMSNorm", layer_mults=[1,2,4]):
        super(OccupancyGridEncoder, self).__init__()
        norm_fn = eval(norm_fn)
        dims = [input_dim, *map(lambda m: hidden_dim * m, layer_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        attn_block = partial(SelfAttentionBlock, dim_head=dim_head, num_heads=num_heads, dropout=dropout)
        
        self.layers = nn.ModuleList([])
        for in_chn, out_chn in in_out:
            self.layers.append(nn.ModuleList([
                    ResDoubleConv2d(in_chn, out_chn, num_groups, dropout),
                    SquareRearrange2D(Residual(PreNorm(norm_fn(out_chn), attn_block(out_chn)))),
                    DownSample2D(out_chn)
                ])
            )
        
        size = grid_size // (2 ** len(layer_mults))
        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dims[-1] * size * size, output_dim),
            nn.Mish(),
            nn.Linear(output_dim, output_dim)
        )
        
    def forward(self, x, obs_mask=None):
        """
        x: N x C x H x W
        """
        for conv, attn, down in self.layers:
            x = conv(x)
            x = attn(x)
            x = down(x)

        out = self.out(x)  
        return out

class ObstacleLinearEncoder(nn.Module):
    def __init__(self, cfg, input_dim, hidden_dim, output_dim):
        super(ObstacleLinearEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim*4),
            nn.Mish(),
            nn.Linear(hidden_dim*4, output_dim),
        )
        
    def forward(self, x, obs_mask=None):
        return self.encoder(x)

class GoalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GoalEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.encoder(x)

class ViT1D(nn.Module):
    def __init__(self, cfg, output_dim, embed_dim, dim_head, num_heads, num_layers, dropout, emb_dropout, max_seq_len):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        module = sys.modules[__name__]  # Current module
        self.to_embed_dim = getattr(module, cfg.obs_to_embed_model.name)(**cfg.obs_to_embed_model.args) # ObstacleToEmbedViT or ObstacleToEmbedMLP

        self.transformer = Transformers(num_layers, embed_dim, dim_head, num_heads, dropout, max_seq_len)
        self.dropout = nn.Dropout(emb_dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim*4),
            nn.Mish(),
            nn.Linear(embed_dim*4, output_dim)
        )
    
    def forward(self, x, obs_mask=None):
        """
        x: B x horizon x state_chn
        """
        x = self.to_embed_dim(x)
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=x.shape[0])
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = self.dropout(x) # dropout
        x = self.transformer(x, obs_mask)
        
        return self.mlp_head(x[:, 0])
     
# borrow implementations from: https://github.com/devinluo27/potential-motion-plan-release/blob/main/diffuser/models/vit_vanilla.py
class PositionalEncoding2D(nn.Module):
    """
    Copy From SRT, their default: num_octaves=8, start_octave=0
    To positional encode the wall locations
    e.g., dim after encoded: [6=(wall*2) x (num_octaves*2)] = 48  
    """
    def __init__(self, num_octaves=4, start_octave=0, norm_factor=30):
        super().__init__()
        self.num_octaves = num_octaves
        self.start_octave = start_octave
        self.norm_factor = norm_factor

    
    def forward(self, coords, rays=None):
        ## must be True because sinPos is for every location
        ## rearrange is done in vit1d
        if coords.ndim == 3: # True for maze2d
            return self.forward_3Dinput(coords, rays)
        else:
            raise NotImplementedError
    

    ## [Not used] Can be deleted
    def forward_3Dinput(self, coords, rays=None):
        # print('coords', coords.shape) ## B, 6
        # embed_fns = [] # not used
        batch_size, num_points, dim = coords.shape
        coords = coords / self.norm_factor

        ## we assume self.start_octaves=0, self.num_octaves=4 in the example below
        # torch.arange(0, 0+4)
        octaves = torch.arange(self.start_octave, self.start_octave + self.num_octaves)

        # to the device if coords
        octaves = octaves.float().to(coords)
        ## (2 ^ [0., 1., ..., 7.]) * pi
        multipliers = 2**octaves * math.pi
        ## after coords: batch_size, num_points, dim, 1
        coords = coords.unsqueeze(-1)
        ## multipliers: (4,) -> (1,1,1,4)
        while len(multipliers.shape) < len(coords.shape):
            multipliers = multipliers.unsqueeze(0)

        ## (batch_size, num_points, dim, 1) * (1,1,1,4)
        scaled_coords = coords * multipliers

        ## (batch_size, num_points, dim, 4) -> (batch_size, num_points, dim * 4)
        sines = torch.sin(scaled_coords).reshape(batch_size, num_points, dim * self.num_octaves)
        cosines = torch.cos(scaled_coords).reshape(batch_size, num_points, dim * self.num_octaves)

        ## (batch_size, num_points, dim * num_octaves + dim * num_octaves)
        result = torch.cat((sines, cosines), -1)
        return result

class ObstacleToEmbedViT(nn.Module):
    def __init__(self, input_dim, embed_dim, output_dim, 
                 dim_head, num_heads, num_layers, 
                 dropout, emb_dropout, max_seq_len):
        super().__init__() 
        num_octaves = 4 # hardcoded
        norm_factor = 30 # hardcoded
        encoding_dim = num_octaves * input_dim * 2
        self.input_dim = input_dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.dyn_posemb = nn.Parameter(torch.randn(1, max_seq_len + 1, embed_dim))

        self.to_embed_dim = nn.Sequential(
            PositionalEncoding2D(num_octaves=num_octaves, norm_factor=norm_factor),
            nn.LayerNorm(encoding_dim),
            nn.Linear(encoding_dim, embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.Mish(),
            nn.Linear(embed_dim*4, embed_dim),
            nn.LayerNorm(embed_dim)
        )

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformers(num_layers, embed_dim, dim_head, num_heads, dropout, max_seq_len)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim*4),
            nn.Mish(),
            nn.Linear(embed_dim*4, output_dim)
        )
            
    def extract_mask(self, x):
        last_values = x[..., -1]  # shape (b * h, l)

        # Create a mask: 1 where last value along `d` is 1, 0 where it's 0
        mask = (last_values == 1).bool()  # shape (b * h, l), values are 1 or 0
        return mask

    def forward(self, x):
        """
        x: B x horizon x (traj num x input_dim)
        """
        batch = x.shape[0] # original batch size
        x = rearrange(x, "b h (l d) -> (b h) l d", d=self.input_dim+1) # fake batch size
        mask = self.extract_mask(x)

        x = self.to_embed_dim(x[:, :, :self.input_dim])
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = x.shape[0])

        mask = torch.cat([torch.ones_like(mask[:, :1]), mask], dim=1) # add cls token mask
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.dyn_posemb[:, :x.shape[1]]

        x = self.dropout(x)
        x = self.transformer(x, mask)
        x = self.mlp_head(x[:, 0])
        x = rearrange(x, "(b h) d -> b h d", b=batch) # convert back
        return x

class ObstacleToEmbedMLP(nn.Module):
    def __init__(self, input_dim, embed_dim, output_dim):
        super().__init__() 
        self.to_embed_dim = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.Linear(embed_dim, embed_dim*4),
            nn.Mish(),
            nn.Linear(embed_dim*4, output_dim),
        )
    
    def forward(self, x):
        return self.to_embed_dim(x)

class Transformers(nn.Module):
    def __init__(self, layer_nums, embed_dim, dim_head, num_heads, dropout, max_seq_len):
        super().__init__()
        self.layers = nn.ModuleList([])
        attn_block = partial(SelfAttentionBlock, dim_head=dim_head, num_heads=num_heads, 
                             dropout=dropout, max_seq_len=max_seq_len) # not causal model
        for _ in range(layer_nums):
            self.layers.append(
                nn.ModuleList([
                    nn.LayerNorm(embed_dim),
                    Rearrange("b h c -> b c h"), # the attention block receives input in a different format
                    attn_block(embed_dim),
                    Rearrange("b c h -> b h c"),
                    nn.LayerNorm(embed_dim),
                    FeedForward(embed_dim, dropout)
                ])
            )
                
    def forward(self, x, obs_mask):
        for norm1, rearrange1, attn, rearrange2, norm2, ff in self.layers:
            x_ = rearrange1(norm1(x))
            x = rearrange2(attn(x_, obs_mask)) + x
            x = ff(norm2(x)) + x
        return x 
        
###################### New Architecture ######################
class ViT1DNew(nn.Module):
    def __init__(self, output_dim, embed_dim, dim_head, num_heads, num_layers, dropout, emb_dropout, max_seq_len):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.transformer = Transformers(num_layers, embed_dim, dim_head, num_heads, dropout, max_seq_len)
        self.dropout = nn.Dropout(emb_dropout)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim*4),
            nn.Mish(),
            nn.Linear(embed_dim*4, output_dim)
        )
    
    def forward(self, x, obs_mask):
        """
        x: B x horizon x state_chn
        """
        
        obs_mask = torch.cat([torch.ones_like(obs_mask[:, :1]), obs_mask], dim=1) # add cls token mask
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=x.shape[0])
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = self.dropout(x) # dropout
        x = self.transformer(x, obs_mask)
        
        return self.mlp_head(x[:, 0])

###### New Encoder ######
class ObstacleMLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ObstacleMLPEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim*4),
            nn.Mish(),
            nn.Linear(hidden_dim*4, output_dim),
        )
        
    def forward(self, x):
        return self.encoder(x)

class StaticObstacleMLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_octaves, norm_factor):
        super(StaticObstacleMLPEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            PositionalEncoding2D(num_octaves=num_octaves, norm_factor=norm_factor),
            nn.Flatten(),
            nn.Linear(input_dim * num_octaves * 2, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim*4),
            nn.Mish(),
            nn.Linear(hidden_dim*4, output_dim),
        )
        
    def forward(self, x):
        return self.encoder(x)
    
class GoalMLPEncoder(nn.Module): # also used for terrain
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GoalMLPEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.encoder(x)