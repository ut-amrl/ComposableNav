import torch 
from copy import deepcopy
import numpy as np 
from composablenav.models.diffusion import make_timesteps
from composablenav.train.utils import load_model
from composablenav.misc.common import repeat_context
from composablenav.misc.process_data import construct_normalized_dynamic_obstacle_from_obj, construct_normalized_static_obstacle_from_obj
from composablenav.train.dataloader_base import ProcessObsHelper
from torch.func import functional_call
from composablenav.models.diffusion import extract

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
        
        # if False:
        #     # weighted mean
        #     weights = torch.tensor([1, 2], device=device)  # Example weights
        #     weights = weights.view(-1, 1, 1)
        #     noise_sum = torch.sum(noise_sum * weights, dim=0, keepdim=True)
        #     noise_sum = noise_sum / weights.sum(dim=0, keepdim=True)
        # else:
        #     noise_sum = noise_sum.mean(dim=0, keepdim=True)

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

def p_sample_loop_ddim(model_context_pair, batch_size, state_cond, 
                  xt, guidance_weight, cfg_mask, device):
    def space_timesteps(
        num_sample_steps = 50,
        total_steps = 1000
    ):
        """
        Adapted from `improved_diffusion.respace.space_timesteps`
        """
        for i in range(1, total_steps):
            if len(range(0, total_steps, i)) == num_sample_steps:
                return list(range(0, total_steps, i))[::-1]
            
        raise ValueError(f'cannot create exactly {total_steps} steps with an integer stride')
    # https://minibatchai.com/2022/07/14/Diffusion_Sampling.html
    timesteps = space_timesteps(13, 25) # hardcoded
    print(timesteps)
    tau = timesteps
    tau_prev = timesteps[1:] + [-1] # for prev
    
    anchor_model = model_context_pair[0][0]
    xt = anchor_model.apply_cond_fn(xt, state_cond) # apply cond at the beginning
    chain = [xt]
    
    print(len(timesteps), len(tau))
    for tau_t, tau_tm1 in zip(tau, tau_prev):
        print("T", tau_t, tau_tm1)
        t_cat = make_timesteps(tau_t, batch_size*2, device)
        # p mean variance
        noise_sum = 0
        xt_cat = torch.cat([xt, xt], dim=0)
        for idx, (diffusion_model, context_cond) in enumerate(model_context_pair):
            noise_cond, noise_uncond = diffusion_model.model(xt_cat, t_cat, context_cond, cfg_mask).chunk(2)
            noise = noise_uncond + guidance_weight * (noise_cond - noise_uncond)
            noise_sum += noise
            
        noise_sum = noise_sum / len(model_context_pair)
        
        # ddim
        tau_t = make_timesteps(tau_t, batch_size, device)
        predicted_x0 = (
            xt - extract(anchor_model.sqrt_one_minus_alpha_hat, tau_t, xt.shape) * noise_sum
        ) * extract(anchor_model.sqrt_recip_alpha_hat, tau_t, xt.shape)
        x0 = torch.clip(predicted_x0, -1, 1)
        
        sigma_tau_t = 0 # hardcoded
        # posterior_variance = (1 - extract(anchor_model.alpha_hat, tau_tm1, xt.shape)) / (1 - alpha_hat) * beta
        tau_tm1 = make_timesteps(tau_tm1, batch_size, device)

        xtau_t_dir = torch.sqrt(
            1 - extract(
                anchor_model.alpha_hat_prev, tau_tm1+1, xt.shape
            ) - sigma_tau_t ** 2
        ) * noise_sum
        
        xt = extract(
            anchor_model.alpha_hat_prev, tau_tm1+1, xt.shape
        ) ** 0.5 * x0 + xtau_t_dir
        # end ddim
        
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
    # legacy code below
    # if "dynamic_obs_encoder" in context_field:
    #     tmp_context_cond["static_obs_encoder"] = {}
    #     tmp_context_cond["dynamic_obs_encoder"]["mask"][:, :] = 0
    # if "static_obs_encoder" in context_field:
    #     tmp_context_cond["dynamic_obs_encoder"] = {}
    #     tmp_context_cond["static_obs_encoder"]["mask"][:, :] = 0
    tmp_context_cond[context_field]["mask"][:, :] = 0
    tmp_context_cond[context_field]["mask"][:, obs_idxs] = 1
    model = diffusion_models[model_key]
    return model, tmp_context_cond