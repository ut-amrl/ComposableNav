import torch 
import imageio
import numpy as np 
from composablenav.misc.common import put_batch_on_device, info_obs_goal_from_fn
from composablenav.models.diffusion_components import diffusion_sample_fn
from composablenav.misc.critic import collision_criteria, goal_reaching_criteria
from composablenav.misc.visualize_utils import plot_path_gif

def diffusion_process_batch(batch, device=None):
    context_cond = put_batch_on_device(batch["context_cond"], device)
    path = batch["path"] if device is None else batch["path"].to(device)
    mask = batch["mask"] if device is None else batch["mask"].to(device)
    hard_cond = batch["hard_cond"]
    # important the name should not be changed
    return {
        "x0": path,
        "mask": mask,
        "context_cond": context_cond,
        "state_cond": hard_cond,
        "filename": batch["filename"]
    }

def ddpo_process_batch(batch, max_padded_len, input_dim, device):
    raw_env_info = batch["raw_env_info"]
    start = batch["start"] if device is None else batch["start"].to(device)
    
    num_env = start.shape[0]
    x_shape = torch.tensor([num_env, max_padded_len, input_dim])

    diffusion_context_cond = put_batch_on_device(batch["context_cond"], device)

    # tbd for random start later
    if input_dim == 2:
        start = start 
    elif input_dim == 3:
        start = torch.cat([start, torch.zeros_like(start)[..., :1]], dim=-1)
        assert start.shape[-1] == 3
    state_cond = {
        "0": start,
    }
    # important the name should not be changed
    return x_shape, diffusion_context_cond, state_cond, raw_env_info

def validate(model, x0, dt, mask=None, context_cond=None, state_cond=None, path_normalizer=None, filename=None, 
                logger=None, log_name="val_loss", **additional_kwargs):
    chain = model.p_sample_loop(x0.shape, diffusion_sample_fn, diffusion_context_cond=context_cond, state_cond=state_cond, **additional_kwargs)
    x0_recon = chain[-1]
    if state_cond.get("start_time_idx") is None:
        start_time_idx_arr = np.zeros(x0_recon.shape[0])
    else:
        start_time_idx_arr = state_cond.get("start_time_idx").detach().cpu().numpy()

    result = {
        "val_score": 0,
        "generated_paths": []
    }
    for idx in range(x0_recon.shape[0]):
        # load only a small amount
        fn = filename[idx]
        info, obstacles_list, goal = info_obs_goal_from_fn(fn)
        goal_radius = info["goal_radius"]
        
        path = x0_recon[idx].detach().cpu().numpy()
        # unnormalize
        path = path_normalizer.unnormalize(path)
        result["generated_paths"].append(path.tolist())

        start_time_idx = start_time_idx_arr[idx].item()
        valid_plan = collision_criteria(path, goal, goal_radius, obstacles_list, dt, start_time_idx)
        goal_reached = goal_reaching_criteria(path, goal, goal_radius)
        if valid_plan and goal_reached:
            result["val_score"] += 1
    result["val_score"] /= x0_recon.shape[0]

    if logger:
        logger(log_name,  result["val_score"], prog_bar=True, logger=True, on_epoch=True)

    return result

def visualize(model, x_shape, idx, sample_fn, context_cond, state_cond, filenames, save_name, frame_size=(640, 480)):
    chain = model.p_sample_loop(x_shape, sample_fn, diffusion_context_cond=context_cond, state_cond=state_cond)
    trajs = chain[-1].detach().cpu().numpy()[idx:idx+1]
    fn = filenames[idx]
    info, obstacles_list, goal = info_obs_goal_from_fn(fn)
    goal_radius = info["goal_radius"]
    
    grid_size = info["grid_size"]
    dt = info["env_dt"]
    trajs_unnormalized = (trajs * grid_size / 2).tolist()
    
    if state_cond.get("start_time_idx") is None:
        start_time_idx = 0
    else:
        start_time_idx = state_cond.get("start_time_idx").detach().cpu().numpy()[idx]
    
    plot_path_gif(trajs_unnormalized, obstacles_list, goal_pos=goal,
                  goal_radius=goal_radius, grid_size=grid_size, dt=dt,
                  save_name=save_name, frame_size=frame_size, 
                  start_time_idx=start_time_idx)

    return chain

def load_model(model, checkpoint, device):
    state_dict = torch.load(checkpoint)
    # if there is ema model, load it
    if state_dict.get("ema_state_dict"):
        print("[Preprocess] Loading EMA model...")
        state_dict = state_dict["ema_state_dict"]
    elif state_dict.get("state_dict"):
        state_dict = state_dict["state_dict"]
        
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("diffusion_model.", "")
        new_key = new_key.replace("trainer_model.", "")
        if "to_embed_dim." in new_key and "to_embed_dim.to_embed_dim." not in new_key:
            new_key = new_key.replace("to_embed_dim.", "to_embed_dim.to_embed_dim.")
        new_state_dict[new_key] = v 
        
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    return model

def gif_to_tensor(gif_path):
    frames = imageio.mimread(gif_path)
    frames = np.stack(frames, axis=0)  # (T, H, W, C)
    
    # Drop alpha
    if frames.shape[-1] == 4:
        frames = frames[..., :3]
    
    frames = frames.astype(np.uint8)
    frames = torch.tensor(frames, dtype=torch.float32) / 255.0
    frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W)
    frames = frames.unsqueeze(0)  # (1, T, C, H, W)
    return frames