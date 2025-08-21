import torch 
from copy import deepcopy
import numpy as np 
from composablenav.models.diffusion import make_timesteps
from composablenav.train.train_utils import load_model
from composablenav.misc.common import repeat_context
from composablenav.misc.process_data import construct_normalized_dynamic_obstacle_from_obj, construct_normalized_static_obstacle_from_obj
from composablenav.train.dataloader_base import ProcessObsHelper
from torch.func import functional_call
from composablenav.models.diffusion import extract
from composablenav.models.diffusion_components import exponential_beta_schedule

def q_sample(x0, t):
    n_timesteps = 25
    beta = exponential_beta_schedule(n_timesteps)
    alpha = 1 - beta
    alpha_hat = torch.cumprod(alpha, dim=0)
    sqrt_alpha_hat = torch.sqrt(alpha_hat)
    sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat)
    
    noise = torch.randn_like(x0)
    return extract(sqrt_alpha_hat, t, x0.shape) * x0 + extract(sqrt_one_minus_alpha_hat, t, x0.shape) * noise

def p_sample_loop(model_context_pair, batch_size, state_cond, 
                  xt, guidance_weight, cfg_mask, device):
    anchor_model = model_context_pair[0][0]
    
    xt = anchor_model.apply_cond_fn(xt, state_cond) # apply cond at the beginning
    chain = [xt]
        
    for t_ in reversed(range(anchor_model.n_timesteps)):
        t_cat = make_timesteps(t_, batch_size*2, device)
        # p mean variance
        noise_sum = 0
        xt_cat = torch.cat([xt, xt], dim=0)
        for idx, (diffusion_model, context_cond) in enumerate(model_context_pair):
            noise_cond, noise_uncond = diffusion_model.model(xt_cat, t_cat, context_cond, cfg_mask).chunk(2)
            noise = noise_uncond + guidance_weight * (noise_cond - noise_uncond)
            noise_sum += noise
            
        noise_sum = noise_sum / len(model_context_pair)
        t = t_cat[:batch_size]
        
        x0_recon = anchor_model.predict_start_from_noise(xt, t, noise_sum)
        if anchor_model.clipped_denoised: # clip the denoised x0_recon
            x0_recon = torch.clamp(x0_recon, -1, 1)
        # end p mean variance
        mean, variance, log_var_clipped = anchor_model.q_posterior(xt, t, x0_recon)
        
        added_noise = torch.randn_like(mean)
        added_noise[t == 0] = 0
        std = torch.exp(0.5 * log_var_clipped)
        xt = mean + std * added_noise
        # end sample
        
        xt = anchor_model.apply_cond_fn(xt, state_cond) # apply cond at the end
        
        chain.append(xt)
        
    return chain

from torch import vmap
import time 
from torch.cuda.amp import autocast
torch.set_float32_matmul_precision('high')

def p_sample_loop_vmap(params, buffers, stacked_context_cond, base_model, anchor_model, num_models,
                       batch_size, state_cond, n_timesteps, xt, guidance_weight, cfg_mask, device, compile=False):
    def fmodel(params, buffers, xt_cat, t_cat, stacked_context_cond, cfg_mask):
        return functional_call(base_model, (params, buffers), (xt_cat, t_cat, stacked_context_cond, cfg_mask))
    xt = anchor_model.apply_cond_fn(xt, state_cond) # apply cond at the beginning
    chain = [xt]

    # compile vmap
    if compile:
        vmap_fn = torch.compile(vmap(fmodel), mode="reduce-overhead")
    else:
        vmap_fn = vmap(fmodel)

    for t_ in reversed(range(n_timesteps)):
        noise_sum = 0
        t_cat = make_timesteps(t_, 1, device)
        t_cat = t_cat.repeat(num_models, batch_size*2)
        xt_cat = xt.repeat(num_models, 2, 1, 1)

        with torch.no_grad():
            predictions1_vmap = vmap_fn(params, buffers, xt_cat, t_cat, stacked_context_cond, cfg_mask)
        
        predictions1_vmap = predictions1_vmap.detach()
        # noise_sum = predictions1_vmap[:, 1] + guidance_weight * (predictions1_vmap[:, 0] - predictions1_vmap[:, 1]) # not correct
        diff = predictions1_vmap[:, :batch_size] - predictions1_vmap[:, batch_size:]
        # noise_sum = predictions1_vmap[:, batch_size:].mean(dim=0) + (guidance_weight * diff).mean(dim=0) # for original cosine
        
        noise_sum = predictions1_vmap[:, batch_size:].mean(dim=0) + (guidance_weight * diff).sum(dim=0) # for exponential

        t = t_cat[0, :batch_size]
        
        x0_recon = anchor_model.predict_start_from_noise(xt, t, noise_sum)
        if anchor_model.clipped_denoised: # clip the denoised x0_recon
            x0_recon = torch.clamp(x0_recon, -1, 1)
        # end p mean variance
        mean, variance, log_var_clipped = anchor_model.q_posterior(xt, t, x0_recon)
        
        added_noise = torch.randn_like(mean)
        added_noise[t == 0] = 0
        std = torch.exp(0.5 * log_var_clipped)
        xt = mean + std * added_noise
        # end sample
        
        xt = anchor_model.apply_cond_fn(xt, state_cond) # apply cond at the end
        
        chain.append(xt)
        
    return chain

def load_diffusion_models(cfg, device):
    models = {}
    for key, model_cfg in cfg.inference.eval_models.items():
        print(f"Loading model {key}")
        model_type = model_cfg.model_type
        checkpoint = model_cfg.checkpoint
        if model_type == "diffusion_models":
            diffusion_model = load_model(model_cfg, model_path=checkpoint, device=device) 
            models[key] = diffusion_model
        else:
            raise ValueError("Invalid model name")

    return models

def context_constructor(cfg, dynamic_obs, static_obs, terrain_obs, 
                        goal, num_repeats, device, mult_x, mult_y):
    grid_size = cfg.dataset_generation.env.grid_size
    dt = cfg.dataset_generation.env.env_dt
    max_planning_time = cfg.dataset_generation.robot.max_planning_time
    max_obj_traj_len = cfg.data.dataset_args.max_obj_traj_len
    max_padded_obj_num = cfg.data.dataset_args.max_padded_obj_num
    max_padded_terrain_num = cfg.data.dataset_args.max_padded_terrain_num

    process_obs_helper = ProcessObsHelper(max_obj_traj_len=max_obj_traj_len, 
                                          max_padded_obj_num=max_padded_obj_num, 
                                          max_padded_terrain_num=max_padded_terrain_num)
    normalized_dynamic = construct_normalized_dynamic_obstacle_from_obj(dynamic_obs, 
                                                                        grid_size, max_planning_time, dt, 
                                                                        mult_x=mult_x, mult_y=mult_y) 
    normalized_static = construct_normalized_static_obstacle_from_obj(static_obs, grid_size=grid_size, 
                                                                      mult_x=mult_x, mult_y=mult_y) 
    normalized_goal = np.array(goal) / (grid_size / 2)

    dynamic_obs, dynamic_obs_mask = process_obs_helper.get_obs_cond(normalized_dynamic)
    static_obs, static_obs_mask = process_obs_helper.get_static_cond(normalized_static)
    dynamic_obs = dynamic_obs.to(device)
    dynamic_obs_mask = dynamic_obs_mask.to(device)
    static_obs = static_obs.to(device)
    static_obs_mask = static_obs_mask.to(device)
    
    context_cond = {
        "dynamic_obs_encoder": {
            "mask": dynamic_obs_mask.unsqueeze(0),
            "input_vals": [o.unsqueeze(0) for o in dynamic_obs]
        },
        "static_obs_encoder": {
            "mask": static_obs_mask.unsqueeze(0),
            "input_vals": [o.unsqueeze(0) for o in static_obs]
        },
        "goal_encoder": {
            "mask": torch.ones([1,1], device=device).float(),
            "input_vals": torch.tensor([normalized_goal], device=device).float()
        },
        "terrain_encoder": {
            
        }
    }
    context_cond = repeat_context(context_cond, num_repeats)
    
    state_cond = {
        "0": torch.tensor([[-1.0, 0]], device=device),
    }
    return context_cond, state_cond

def construct_state_cond_from_traversed_path(traversed_path, device):
    state_cond = {}
    for idx, p in enumerate(traversed_path):
        state_cond[str(idx)] = torch.tensor([p], device=device)
    return state_cond

def construct_context_state_conds(diffusion_models, context_cond, meta_data):
    model_key = meta_data["model_key"]
    obs_idxs = meta_data["obs_idx"]
    context_field = meta_data["context_field"]
            
    tmp_context_cond = deepcopy(context_cond)
    tmp_context_cond["static_obs_encoder"]["mask"][:, :] = 0
    tmp_context_cond["dynamic_obs_encoder"]["mask"][:, :] = 0
    tmp_context_cond["terrain_encoder"]["mask"][:, :] = 0
    tmp_context_cond[context_field]["mask"][:, :] = 0
    tmp_context_cond[context_field]["mask"][:, obs_idxs] = 1
    model = diffusion_models[model_key]
    return model, tmp_context_cond