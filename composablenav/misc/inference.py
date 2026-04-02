import torch 
from copy import deepcopy
import numpy as np 
from hydra.utils import instantiate
from torch.func import functional_call
from torch import vmap
torch.set_float32_matmul_precision('high')

from composablenav.models.diffusion import make_timesteps
from composablenav.train.train_utils import load_model
from composablenav.misc.common import repeat_context
from composablenav.misc.process_data import construct_normalized_dynamic_obstacle_from_obj, construct_normalized_static_obstacle_from_obj
from composablenav.train.dataloader_base import ProcessObsHelper

def get_guided_diffusion_path_vmap(stacked_context_cond, anchor_model, base_model, params, buffers, 
                                        num_models, cfg_mask,state_cond, n_timesteps, xt,
                                        max_planning_time, batch_size, guidance_weight,
                                        device, compile=False):  
    if xt is None:
        shape = (batch_size, max_planning_time, 2)
        xt = torch.randn(*shape, device=device)
    if n_timesteps is None:
        n_timesteps = anchor_model.n_timesteps
        
    chain = p_sample_loop_vmap(params, buffers, stacked_context_cond, base_model, anchor_model, num_models,
                       batch_size, state_cond, n_timesteps, xt, guidance_weight, cfg_mask, device, compile)
    return chain

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
        diff = predictions1_vmap[:, :batch_size] - predictions1_vmap[:, batch_size:]
        
        noise_sum = predictions1_vmap[:, batch_size:].mean(dim=0) + (guidance_weight * diff).sum(dim=0)

        t = t_cat[0, :batch_size]
        
        x0_recon = anchor_model.predict_start_from_noise(xt, t, noise_sum)
        if anchor_model.clipped_denoised: # clip the denoised x0_recon
            x0_recon = torch.clamp(x0_recon, -1, 1)

        mean, variance, log_var_clipped = anchor_model.q_posterior(xt, t, x0_recon)
        
        added_noise = torch.randn_like(mean)
        added_noise[t == 0] = 0
        std = torch.exp(0.5 * log_var_clipped)
        xt = mean + std * added_noise
        
        xt = anchor_model.apply_cond_fn(xt, state_cond) # apply cond at the end
        
        chain.append(xt)
        
    return chain

def load_diffusion_models(cfg, device):
    models = {}
    for key, model_cfg in cfg.inference.eval_models.items():
        model = instantiate(cfg.model)
        diffusion_model = load_model(model, model_cfg.checkpoint, device)
        models[key] = diffusion_model

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

def stack_dictionaries_list(dicts_list):
    """
    Stacks tensor values across a list of dictionaries by recursively stacking
    values for each key across all dictionaries in the list.
    """
    # Initialize the result dictionary
    stacked_dict = {}

    # Get the keys from the first dictionary (assuming all dicts have the same structure)
    keys = dicts_list[0].keys()

    for key in keys:
        values = [d[key] for d in dicts_list]
        if isinstance(values[0], dict):
            # Recursively handle nested dictionaries
            stacked_dict[key] = stack_dictionaries_list(values)
        
        elif isinstance(values[0], torch.Tensor):
            # Stack tensors along a new dimension
            stacked_dict[key] = torch.stack(values, dim=0)
        elif isinstance(values[0], list) and isinstance(values[0][0], torch.Tensor):
            # Recursively handle nested dictionaries
            stacked_dict[key] = [torch.stack(o, dim=0) for o in zip(*values)]
        else:
            # If values are not tensors, you can customize this part as needed
            stacked_dict[key] = values
    
    return stacked_dict

def fast_interpolate_path(path: np.ndarray, interval_original: float, interval_new: float):
    """
    Interpolates a path to match a new resolution, even for coarser intervals.
    
    Parameters:
    - path (np.ndarray): Original path of shape (N, 2), where each row is [x, y].
    - interval_original (float): Time interval between points in the original path.
    - interval_new (float): Desired time interval between points in the output path.
    
    Returns:
    - np.ndarray: Interpolated path with the new resolution.
    """
    # Calculate the total time of the path
    total_time = (len(path) - 1) * interval_original
    
    # Generate time indices for original and new paths
    times_original = np.arange(0, total_time + interval_original, interval_original)[:len(path)] # solve the off by one error
    times_new = np.arange(0, total_time + interval_new, interval_new)
    
    # Interpolate x and y separately
    path_interpolated = np.empty((len(times_new), 2), dtype=path.dtype)
    for dim in range(2):
        path_interpolated[:, dim] = np.interp(times_new, times_original, path[:, dim])
    
        # extrapolate the last point
        last_val = path_interpolated[-1, dim]
        second_last_val = path_interpolated[-2, dim]
        dt = total_time - (len(path_interpolated) - 2) * interval_new
        ratio = dt / interval_new
        diff = (last_val - second_last_val) / ratio
        path_interpolated[-1, dim] = second_last_val + diff
    return path_interpolated