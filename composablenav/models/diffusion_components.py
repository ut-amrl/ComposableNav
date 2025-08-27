import torch 
import math 
from torch.distributions import Normal
from composablenav.misc.common import forward_motion_rollout_tensor

def cosine_beta_schedule(n_timesteps, s=0.008, dtype=torch.float32):
    x = (torch.arange(0, n_timesteps+1, dtype=dtype) / n_timesteps + s) / (1 + s) * math.pi * 0.5
    fts = torch.pow(x.cos(), 2)
    return torch.clip(1 - fts[1:] / fts[:-1], 0, 0.999)

def exponential_beta_schedule(n_diffusion_steps, beta_start=1e-4, beta_end=1.0):
    # exponential increasing noise from t=0 to t=T
    x = torch.linspace(0, n_diffusion_steps, n_diffusion_steps)
    beta_start = torch.tensor(beta_start)
    beta_end = torch.tensor(beta_end)
    a = 1 / n_diffusion_steps * torch.log(beta_end / beta_start)
    return beta_start * torch.exp(a * x)

def diffusion_sample_fn_log_prob(model, x, t, diffusion_context_cond=None, **no_use_kwargs):
    mean, _, log_var = model.p_mean_variance(x, t, diffusion_context_cond)
    if (t == 0).any() == True:
        # last timestep does not need to calculate log prob since we are not sampling from it
        assert torch.equal(t, torch.zeros_like(t)), "The tensor is not all zeros."
        return mean, None
    
    std = torch.exp(0.5 * log_var)
    std_clipped = torch.clip(std, min=1e-6) # to avoid numerical instability
    normal_dist = Normal(mean, std_clipped)
    xt_1 = normal_dist.sample()
    log_prob = normal_dist.log_prob(xt_1)

    ####### may also need to apply cond for log probs here? #######
    log_prob = log_prob[:, 1:] # remove the first timestep (hardcoded)
    log_prob = log_prob.mean(dim=list(range(1, log_prob.ndim))) # sum over the state dimension
    ####### hardcoded ^

    return xt_1, log_prob

def diffusion_sample_fn(model, x, t, diffusion_context_cond=None, **no_use_kwargs):
    """
    need to not use noise when t = 0
    """
    noise = torch.randn_like(x)
    noise[t == 0] = 0
    mean, _, log_var = model.p_mean_variance(x, t, diffusion_context_cond)
    std = torch.exp(0.5 * log_var)
    return mean + std * noise

def apply_initial_coordinate_cond(x, cond=None, **unused_kwargs):
    # state_dim = 2
    # 2/15 updates
    state_dim = x.shape[-1]
    if cond is None:
        return x
    # dirac delta function to fix known values
    for horizon, val in cond.items():
        x[:, int(horizon), :state_dim] = val[:, :state_dim].clone()
    return x

def apply_fixed_past_cond(x, conds=None, **unused_kwargs):
    state_dim = 2
    if conds is None:
        return x
    # dirac delta function to fix known values
    for idx, cond in conds:
        for horizon, val in cond.items():
            x[idx, int(horizon), :state_dim] = val[:, :state_dim].clone()
    return x

def apply_kinematic_cond(xt, cond=None, differential_normalizer=None, env_dt=None, **unused_kwargs):
    assert differential_normalizer is not None
    
    # Apply Dirac delta function to fix known values\
    if cond is not None:
        state_dim = 5
        for horizon, val in cond.items():
            xt[:, int(horizon), :state_dim] = val[:, :state_dim].clone()

    # add clip 
    xt = torch.clamp(xt, -1, 1)
    
    # Unnormalize the state variables (x, y, theta, v, w)
    xt = differential_normalizer.unnormalize(xt)

    # Split the tensor into components
    x, y, theta, v, w = xt.chunk(5, dim=2)

    # Initialize starting points
    curr_x, curr_y, curr_theta = x[:, 0:1], y[:, 0:1], theta[:, 0:1]
    curr_theta = (curr_theta + torch.pi) % (2 * torch.pi) - torch.pi

    dt = torch.ones_like(curr_x) * env_dt
    
    # Apply the kinematic model over the horizon
    result_x, result_y, result_theta = [curr_x], [curr_y], [curr_theta]

    # Loop through the horizon
    for horizon_idx in range(1, xt.shape[1]):
        curr_v = v[:, horizon_idx - 1:horizon_idx]
        curr_w = w[:, horizon_idx - 1:horizon_idx]
        
        # Compute next states using forward_motion_rollout_tensor
        curr_x, curr_y, curr_theta = forward_motion_rollout_tensor(
            curr_v, curr_w, curr_x, curr_y, curr_theta, dt
        )

        # Append current results to the list
        result_x.append(curr_x)
        result_y.append(curr_y)
        result_theta.append(curr_theta)

    # Concatenate the results along the time dimension
    result_x = torch.cat(result_x, dim=1)
    result_y = torch.cat(result_y, dim=1)
    result_theta = torch.cat(result_theta, dim=1)

    # Concatenate the results into the full tensor
    xt = torch.cat([result_x, result_y, result_theta, v, w], dim=2)

    # Normalize the state variables back
    xt = differential_normalizer.normalize(xt)

    return xt