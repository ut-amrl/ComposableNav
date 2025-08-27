import torch 
import torch.nn as nn
from functools import partial
from einops import rearrange
from einops.layers.torch import Rearrange
from typing import Union
from omegaconf import DictConfig
from composablenav.models.utils import (
    SinusoidalPosEmbedding, 
    PreNorm, 
    RMSNorm, 
    SelfAttentionBlock,
    DownSample,
    UpSample,
    Residual,
    ViT1DNew
)
from composablenav.models import utils
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
allow_ops_in_compiled_graph()

class ContextEncoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(ContextEncoder, self).__init__()
        self.obs_encoder = getattr(utils, cfg.obs_encoder_name)(cfg=cfg, **cfg.obs) 
        self.obs_input_name = cfg.obs_input_name
        self.obs_mask_name = cfg.obs_mask_name
        # goal might not be relevant
        if "goal_encoder_name" not in cfg:
            self.goal_encoder = None
        else:
            self.goal_encoder = getattr(utils, cfg.goal_encoder_name)(**cfg.goal)
            self.goal_input_name = cfg.goal_input_name
        
    def forward(self, context_cond):
        obs = context_cond[self.obs_input_name]
        obs_mask = context_cond[self.obs_mask_name]
        obs_embed = self.obs_encoder(obs, obs_mask=obs_mask)
        
        if self.goal_encoder is None:
            return obs_embed
        
        goal = context_cond[self.goal_input_name]
        goal_embed = self.goal_encoder(goal)
        
        return torch.cat((obs_embed, goal_embed), dim=1)
    

class ContextEncoderViT(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(ContextEncoderViT, self).__init__()
        self.vit = ViT1DNew(**cfg.vit_args)
        
        # self.pos_embedding = SinusoidalPosEmbedding(cfg.vit_args.embed_dim)
        # encoder_names = []
        # instantiate model
        for idx, (key, encoder) in enumerate(cfg.encoders.items()):
            setattr(self, encoder.name, getattr(utils, encoder.model_name)(**encoder.args))
            # encoder_names.append(encoder.name)
        
        # encoder_names = sorted(encoder_names) # to make sure the order is consistent
        # self.pos_dict = {}
        # for idx, encoder_name in enumerate(encoder_names):
        #     self.pos_dict[encoder_name] = idx
        
    def forward(self, context_cond):
        data = []
        masks = []
        
        # add position embedding to indicate which is which
        for encoder_name, values in context_cond.items():
            assert isinstance(values, dict), f"Context Encoder ViT expects a dictionary of values, got {type(values)}"

            if len(values) == 0:
                continue 
            input_vals = values["input_vals"]
            mask = values["mask"]
            masks.append(mask)
            
            # encoder_idx = self.pos_dict[encoder_name]
            # t = torch.full((mask.shape[0], ), encoder_idx).to(mask.device)
            # pos_embed = self.pos_embedding(t)
            
            if isinstance(input_vals, list) and isinstance(input_vals[0], torch.Tensor):
                for input_val in input_vals:
                    d = getattr(self, encoder_name)(input_val)
                    # d = d + pos_embed # add positional embedding
                    data.append(d)
            else:
                d = getattr(self, encoder_name)(input_vals)
                data.append(d)
        concat_mask = torch.cat(masks, dim=1)
        stacked_input = torch.stack(data, dim=1)

        return self.vit(stacked_input, concat_mask)
    
class TemporalResnetBlockCond(nn.Module):
    def __init__(self, in_chn, out_chn, time_context_dim, norm_fn=RMSNorm, dropout=0., use_scale_shift=True):
        super().__init__()
        self.use_scale_shift = use_scale_shift
        self.block1 = nn.ModuleList([
            nn.Conv1d(in_chn, out_chn, kernel_size=3, padding=1),
            norm_fn(out_chn),
            nn.Mish(),
            nn.Dropout(dropout)
        ])
        self.block2 = nn.Sequential(
            nn.Conv1d(out_chn, out_chn, kernel_size=3, padding=1),
            norm_fn(out_chn),
            nn.Mish(),
            nn.Dropout(dropout)
        )
        self.time_context_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(time_context_dim, out_chn * 2 if use_scale_shift else out_chn),
            Rearrange("b c -> b c 1")
        )
        self.residual_conv = nn.Conv1d(in_chn, out_chn, kernel_size=1) if in_chn != out_chn else nn.Identity()
        
    def forward(self, x, t, context):
        """
        x: batch_size x state_chn x horizon
        t: batch_size x time_dim
        context: batch_size x context_dim 
        """
        t_context = torch.cat((t, context), dim=1)
            
        out = x.clone()
        for layer in self.block1:
            if isinstance(layer, nn.Mish):
                if self.use_scale_shift:
                    scale, shift = self.time_context_mlp(t_context).chunk(2, dim=1)
                    out = out * (1 + scale) + shift
                else:
                    out = out + self.time_context_mlp(t_context)
            out = layer(out)

        out = self.block2(out)
        
        return out + self.residual_conv(x)

class TemporalUnetCond(nn.Module):
    def __init__(self, input_dim, hidden_dim, dim_head=32, num_heads=4, dropout=0, layer_mults=[1,2,4,8],
                 context_args: Union[None, DictConfig]=None):
        super().__init__()
        """
        Borrowed Implementation from Diffuser
        This Unet Implementation does not have the skip connection at the highest level
        4 downs and 3 ups
        no downsample at the last down layer
        output layer is the last up layer        
        """
        self.context_encoding = ContextEncoderViT(context_args) # legacy: ContextEncoder
            
        dims = [input_dim, *map(lambda m: hidden_dim * m, layer_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        # print(f'[ models/temporal ] Channel dimensions: {in_out}')
        
        time_dim = hidden_dim
        self.context_dim = context_args.context_dim 
        t_res_block = partial(TemporalResnetBlockCond, 
                              time_context_dim=self.context_dim+time_dim, 
                              norm_fn=RMSNorm, dropout=dropout, use_scale_shift=True)
        attn_block = partial(SelfAttentionBlock, dim_head=dim_head, num_heads=num_heads, dropout=dropout)
        
        self.time_encoding = nn.Sequential(
            SinusoidalPosEmbedding(time_dim),
            nn.Linear(time_dim, time_dim*4),
            nn.Mish(),
            nn.Linear(time_dim*4, time_dim)
        )
        
        self.down_layers = nn.ModuleList([])
        for index, (in_dim, out_dim) in enumerate(in_out):
            self.down_layers.append(nn.ModuleList([
                t_res_block(in_dim, out_dim),
                t_res_block(out_dim, out_dim),
                Residual(PreNorm(RMSNorm(out_dim), attn_block(out_dim))),
                DownSample(out_dim) if index < len(in_out) - 1 else nn.Identity()
            ]))
        
        mid_dim = in_out[-1][-1]
        self.middle_layers = nn.ModuleList([
            t_res_block(mid_dim, mid_dim),
            Residual(PreNorm(RMSNorm(mid_dim), attn_block(mid_dim))),
            t_res_block(mid_dim, mid_dim)
        ])
        
        self.up_layers = nn.ModuleList([])
        for index, (in_dim, out_dim) in enumerate(reversed(in_out[1:])):
            self.up_layers.append(nn.ModuleList([
                t_res_block(out_dim * 2, in_dim),
                t_res_block(in_dim, in_dim),
                Residual(PreNorm(RMSNorm(in_dim), attn_block(in_dim, dropout=dropout))),
                UpSample(in_dim) if index < len(in_out) - 1 else nn.Identity() # if condition will always be true
            ]))
        
        in_dim, out_dim = in_out[0]
        self.output_layer = nn.Sequential(
            nn.Conv1d(out_dim, out_dim, kernel_size=3, padding=1),
            RMSNorm(out_dim),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Conv1d(out_dim, in_dim, kernel_size=1)
        )

    def forward(self, x, t, context_cond, cfg_mask):
        """
        x: batch_size x horizon x state_chn
        t: batch_size
        """
        x = rearrange(x, "b h c -> b c h")
        t = self.time_encoding(t)

        context = self.context_encoding(context_cond)
        context = context * cfg_mask.unsqueeze(1)
        h = []
        for block1, block2, attention, downsample in self.down_layers:
            x = block1(x, t, context)
            x = block2(x, t, context)
            x = attention(x)
            h.append(x)
            x = downsample(x)
        x = self.middle_layers[0](x, t, context)
        x = self.middle_layers[1](x)
        x = self.middle_layers[2](x, t, context)
        
        for block1, block2, attention, upsample in self.up_layers:
            x = block1(torch.cat((x, h.pop()), dim=1), t, context)
            x = block2(x, t, context)
            x = attention(x)
            x = upsample(x)
            
        x = self.output_layer(x)
        return rearrange(x, "b c h -> b h c")