import torch 
import torch.nn as nn 
import numpy as np
import copy
from composablenav.models.diffusion_components import diffusion_sample_fn, diffusion_sample_fn_log_prob
from composablenav.misc.common import info_obs_goal_from_fn
from composablenav.misc.critic import collision_criteria, goal_reaching_criteria
from composablenav.datasets.obstacles import Circle, Rectangle
from composablenav.misc.visualizer_utils import visualize_paths_with_rewards

def extract(val, t, x_shape):
    """
    val: T
    t: B -> require to be type long
    x_shape: provides the length of the input
    [helper] get the value from val indexed by each t value
    using reshape here because val.gather may return tensor not contiguous
    """
    b, *_ = x_shape
    out = val.gather(-1, t)
    return out.reshape(b, *((1,)*(len(x_shape) -1)))

def concat_context(context_cond_1, context_cond_2):
    combined = {}
    for key in context_cond_1:
        combined[key] = {}
        for inner_key in context_cond_1[key]:
            if isinstance(context_cond_1[key][inner_key], list):
                combined[key][inner_key] = [
                    torch.cat([t1, t2], dim=0) 
                    for t1, t2 in zip(context_cond_1[key][inner_key], context_cond_2[key][inner_key])
                ]
            else:
                combined[key][inner_key] = torch.cat([
                    context_cond_1[key][inner_key],
                    context_cond_2[key][inner_key]
                ], dim=0)
    return combined

def make_timesteps(t, batch_size, device):
    return torch.full((batch_size,), t, device=device, dtype=torch.long)

class GaussianDiffusion(nn.Module):
    def __init__(self, model, noise_scheduler, n_timesteps, clipped_denoised=True, predict_epsilon=True,
                 loss_fn=None, apply_cond_fn=None,
                 use_traj_mask=True, cfg_drop_prob=0, cfg_guide_w=0, cond_weight=1, compile=False):
        super().__init__()
        """
        this gaussian diffusion model will condition on observation cond and hard condition for trajectories    
        """
        beta = noise_scheduler(n_timesteps)
        alpha = 1 - beta
        alpha_hat = torch.cumprod(alpha, dim=0)
        alpha_hat_prev = torch.cat([torch.ones(1), alpha_hat[:-1]])
        
        self.register_buffer('alpha', alpha)
        self.register_buffer('alpha_hat', alpha_hat)
        self.register_buffer('alpha_hat_prev', alpha_hat_prev)
        
        self.register_buffer('sqrt_alpha_hat', torch.sqrt(alpha_hat))
        self.register_buffer('sqrt_one_minus_alpha_hat', torch.sqrt(1 - alpha_hat))
        self.register_buffer('sqrt_recip_alpha_hat', torch.sqrt(1 / alpha_hat))
        self.register_buffer('sqrt_recip_alpha_hat_m1', torch.sqrt(1 / alpha_hat - 1))
        
        posterior_variance = (1 - alpha_hat_prev) / (1 - alpha_hat) * beta
        self.register_buffer('posterior_mean_coeff_0', torch.sqrt(alpha_hat_prev) * beta / (1 - alpha_hat))
        self.register_buffer('posterior_mean_coeff_t', torch.sqrt(alpha) * (1 - alpha_hat_prev) / (1 - alpha_hat))
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_var_clipped', torch.log(torch.clamp(posterior_variance, min=1e-20)))
                
        self.clipped_denoised = clipped_denoised
        self.predict_epsilon = predict_epsilon
        self.n_timesteps = n_timesteps
        self.loss_fn = loss_fn
        self.model = model
        self.apply_cond_fn = apply_cond_fn
        self.use_traj_mask = use_traj_mask
        self.cfg_drop_prob = cfg_drop_prob
        self.cfg_guide_w = cfg_guide_w
        self.cond_weight = cond_weight
        self.compile = compile

        # print(f"Gaussian Diffusion Model with {n_timesteps} timesteps")
        # print(f"Training with traj mask: {self.use_traj_mask}")
        # if self.cfg_drop_prob > 0:
        #     print(f"Training with CFG with dropout probability of {self.cfg_drop_prob}")
        # if self.cfg_guide_w > 0:
        #     print(f"Evaluating with CFG with guidance weight of {self.cfg_guide_w}")
        # if self.predict_epsilon:
        #     print("PREDICTING epsilon")
        # else:
        #     print("PREDICTING XY")
        # if not self.use_traj_mask:
        #     print("NOT USING TRAJ MASK")
        # else:
        #     print("USING TRAJ MASK")

        assert self.apply_cond_fn is not None
    
    def predict_start_from_noise(self, xt, t, noise):
        """
        [for infernece] derived from forward diffusion process
        xt = sqrt(a_hat) x0 + sqrt(1 - a_hat) e
        => x0 = sqrt(1 / a_hat) xt - sqrt(1 / a_hat - 1) e
        if self.predict_epsilon, model output is (scaled) noise;
        otherwise, model predicts x0 directly
        """
        if self.predict_epsilon:
            return extract(self.sqrt_recip_alpha_hat, t, xt.shape) * xt - \
                extract(self.sqrt_recip_alpha_hat_m1, t, xt.shape) * noise
        else:
            return noise            
        
    def q_posterior(self, xt, t, x0):
        """
        [for inference] estimate true posterior given xt and x0
        x_{t-1} ~ q(x_{t-1} | x_t, x0, t)
        => mu = (sqrt(a_hat_{t-1}) b_t / (1 - a_hat_t)) x0 + (sqrt(a_t) (1 - a_hat_{t-1}) / (1 - a_hat_t)) x_t
        => variance = (1 - a_hat_{t-1}) / (1 - a_hat) * beta
        => log_var_clipped = log(clamp(variance))
        """
        mean = extract(self.posterior_mean_coeff_0, t, xt.shape) * x0 + \
               extract(self.posterior_mean_coeff_t, t, xt.shape) * xt
        variance = extract(self.posterior_variance, t, xt.shape)
        log_var_clipped = extract(self.posterior_log_var_clipped, t, xt.shape)
        return mean, variance, log_var_clipped
    
    def p_mean_variance(self, xt, t, context_cond=None):
        """
        [for inference] estimate prior mean and variance given x_t; there is observation condition for predicting the noise and hard state condition to follow
        => e = model(xt, t, cond) [the noise is the noise from start to t]
        => x0_recon = predict_start_from_noise(xt, t, e)
        => need to clip denoised x0_recon
        => p = q_posterior(xt, t, x0_recon)
        """
        if self.compile:
            model = torch.compile(self.model, mode="reduce-overhead")
        else:
            model = self.model
        cfg_mask_cond = torch.ones(xt.shape[0], device=xt.device)
        
        if self.cfg_guide_w > 1:
            context_cond_double = concat_context(context_cond, context_cond)
            noise = model(
                torch.cat([xt, xt], dim=0), 
                torch.cat([t, t], dim=0), 
                context_cond_double, 
                torch.cat([cfg_mask_cond, torch.zeros_like(cfg_mask_cond)], dim=0)
            )
            noise_cond, noise_uncond = noise.chunk(2, dim=0)
            noise = (1 - self.cfg_guide_w) * noise_uncond + self.cfg_guide_w * noise_cond
        else:
            noise = model(xt, t, context_cond, cfg_mask_cond)
        
        x0_recon = self.predict_start_from_noise(xt, t, noise)
        if self.clipped_denoised: # clip the denoised x0_recon
            x0_recon = torch.clamp(x0_recon, -1, 1)

        mean, variance, log_var_clipped = self.q_posterior(xt, t, x0_recon)
        
        return mean, variance, log_var_clipped
    
    def p_sample_loop(self, shape, sample_fn, diffusion_context_cond=None, guide_context_conds=None, 
                      state_cond=None, **additional_kwargs):
        """
        [for inference] sample x0 from random noise
        => initialize xt
        => for t=T to 0
        =>   convert t to tensor
        =>   xt = sample_fn(self, xt, t, context_cond) [can be either ddpm sample, ddim sample, or guided ddpm]
        =>   apply hard condition on xt [optional]
        """
        device = self.alpha.device
        batch_size = shape[0]
        
        ###### 12/08/2024 ######
        tmp_state_cond = copy.deepcopy(state_cond)
        state_cond = {}
        state_cond["0"] = tmp_state_cond["0"]
        ###### 12/08/2024 ######
        xt = torch.randn(*shape, device=device)
        xt = self.apply_cond_fn(xt, state_cond) # apply cond at the beginning
        chain = [xt]
        
        for t in reversed(range(self.n_timesteps)):
            t = make_timesteps(t, batch_size, device)
            xt = sample_fn(self, xt, t, diffusion_context_cond=diffusion_context_cond, 
                           guide_context_conds=guide_context_conds, state_cond=state_cond, **additional_kwargs)
            xt = self.apply_cond_fn(xt, state_cond) # apply cond at the end
            chain.append(xt)
            
        return chain
    
    def q_sample(self, x0, t, noise=None):
        """
        [for training] get x_t from x0, t, and noise
        => x_t = sqrt(a_hat) x0 + sqrt(1 - a_hat) e
        """
        if noise is None:
            noise = torch.randn_like(x0)
        return extract(self.sqrt_alpha_hat, t, x0.shape) * x0 + extract(self.sqrt_one_minus_alpha_hat, t, x0.shape) * noise
        
    def p_losses(self, x0, t, mask=None, context_cond=None, state_cond=None, **additional_kwargs):
        """
        [for training] compute the loss given particular x0, cond, and t
        => sample noise
        => xt = q_sample(x0, t, noise)
        => noise_pred = model(xt, t, cond)
        => loss = loss_fn(noise, noise_pred)
        """
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        
        # ############# 12/08/2024 #############
        # # need to process state_cond here
        # use_random_hard_cond = torch.rand(1).item() > 0.5
        # if use_random_hard_cond:
        #     # rand_state_cond is full min
        #     # few_rand_state_cond is 1/3
        #     # rand_state_cond_nomaskout is 3/4
        #     max_hard_cond_len = (torch.min(state_cond["path_len"]) // 3).long()
        #     random_sample = torch.randint(1, max_hard_cond_len + 1, (1,)).item()
        # else:
        #     random_sample = 1
        # tmp_state_cond = copy.deepcopy(state_cond)
        # state_cond = {}
        # for idx in range(random_sample):
        #     state_cond[str(idx)] = tmp_state_cond[str(idx)]
        # ############# 12/08/2024 #############
        
        ############# 2/5/2025 #############
        # Only fix start state
        tmp_state_cond = copy.deepcopy(state_cond)
        state_cond = {}
        for idx in range(1):
            state_cond[str(idx)] = tmp_state_cond[str(idx)]
        # ############# 2/5/2025 #############
        
        # apply
        xt = self.apply_cond_fn(xt, state_cond)
        
        cfg_mask = torch.ones(xt.shape[0], device=xt.device)
        cfg_mask[np.random.rand(xt.shape[0]) < self.cfg_drop_prob] = 0

        noise_pred = self.model(xt, t, context_cond, cfg_mask)
        noise_pred = self.apply_cond_fn(noise_pred, state_cond) 
        assert noise.shape == noise_pred.shape
        
        # apply hard cond does not make sense here. so I assume we can also mask it out, [new]
        if self.predict_epsilon:
            loss = self.loss_fn(noise_pred, noise)
        else:
            raise NotImplementedError("Not implemented yet")
            x0_recon = self.predict_start_from_noise(xt, t, noise_pred)
            loss1 = self.loss_fn(x0_recon, x0)
            x0_recon = self.apply_cond_fn(x0_recon, state_cond)
            loss2 = self.loss_fn(x0_recon, x0)
            loss = (1 - self.cond_weight) * loss1 + self.cond_weight * loss2

        # mask[:, 0, :] = 1 # disable for xy
        if mask is None or not self.use_traj_mask:
            return loss.mean()
        else:
            raise NotImplementedError("Not implemented yet")
            loss = (loss * mask).sum() / mask.sum() # sum over the sequence length
            return loss
    
    def loss(self, x0, mask=None, context_cond=None, state_cond=None, **additional_kwargs):
        """
        [for training] compute the loss for some t
        => sample t
        => loss = p_losses(x0, cond, t)
        """
        assert mask is not None and context_cond is not None and state_cond is not None
        t = torch.randint(0, self.n_timesteps, (x0.shape[0],), device=x0.device, dtype=torch.long)
        return self.p_losses(x0, t, mask, context_cond, state_cond, **additional_kwargs)
    
    def forward(self, x, context_cond=None, state_cond=None, **additional_kwargs):
        return self.loss(x, context_cond, state_cond, **additional_kwargs)
    
    # for optimizing the validate script during training
    def reconstruct_obs(self, context_cond, idx):
        grid_size_norm = 10 # hardcoded
        goal_radius = 2 # hardcoded
        dt = 0.25 # hardcoded
        dynamic_obs_mask = context_cond["dynamic_obs_encoder"]["mask"][idx].detach().cpu().numpy()
        dynamic_obs = context_cond["dynamic_obs_encoder"]["input_vals"]
        goal = context_cond["goal_encoder"]["input_vals"][idx].detach().cpu().numpy() * grid_size_norm
        obstacles = []
        for i, m in enumerate(dynamic_obs_mask):
            if m == 0:
                break
            obs = dynamic_obs[i][idx].detach().cpu().numpy() * grid_size_norm
            start_x = obs[0]
            start_y = obs[1]
            next_x = obs[3]
            next_y = obs[4]
            theta = np.arctan2(next_y - start_y, next_x - start_x)
            speed = np.sqrt((next_x - start_x) ** 2 + (next_y - start_y) ** 2) / dt
            obstacles.append(Circle(start_x, start_y, theta, speed, radius=2))
        
        # only used for the 15k hardcode
        static_obs_mask = context_cond["static_obs_encoder"]["mask"][idx].detach().cpu().numpy()
        static_obs = context_cond["static_obs_encoder"]["input_vals"]
        for i, m in enumerate(static_obs_mask):
            if m == 0:
                break
            obs = static_obs[i][idx].detach().cpu().numpy()
            x = obs[0] * 10
            y = obs[1] * 10
            theta = obs[2] * np.pi
            width = obs[3] * 10 + 10
            height = obs[4] * 10 + 10
            obstacles.append(Rectangle(x,y, theta, 0, width, height))
        return obstacles, goal, goal_radius
    
    def validate(self, x0, dt, mask=None, context_cond=None, state_cond=None, path_normalizer=None, filename=None, 
                 logger=None, log_name="val_loss", **additional_kwargs):
        chain = self.p_sample_loop(x0.shape, diffusion_sample_fn, diffusion_context_cond=context_cond, state_cond=state_cond, **additional_kwargs)
        x0_recon = chain[-1]
        
        result = {
            "val_score": 0,
            "generated_paths": []
        }
        for idx in range(x0_recon.shape[0]):
            obstacles_list, goal, goal_radius = self.reconstruct_obs(context_cond, idx)
            # fn = filename[idx]
            # info, obstacles_list, goal = info_obs_goal_from_fn(fn)
            # goal_radius = info["goal_radius"]
            
            path = x0_recon[idx].detach().cpu().numpy()
            # unnormalize
            path = path_normalizer.unnormalize(path)
            result["generated_paths"].append(path.tolist())

            valid_plan = collision_criteria(path, goal, goal_radius, obstacles_list, dt)
            goal_reached = goal_reaching_criteria(path, goal, goal_radius)
            if valid_plan and goal_reached:
                result["val_score"] += 1
        result["val_score"] /= x0_recon.shape[0]

        if logger:
            logger(log_name,  result["val_score"], prog_bar=True, logger=True, on_epoch=True)

        return result
    
    # temporary
    def visualize(self, x_shape, idx, sample_fn, context_cond, state_cond, save_name, frame_size=(640, 480)):
        chain = self.p_sample_loop(x_shape, sample_fn, diffusion_context_cond=context_cond, state_cond=state_cond)
        trajs = chain[-1].detach().cpu().numpy()[idx:idx+1]
        rewards = np.linspace(0, 1, trajs.shape[0]+1)
        
        obstacles_list, goal, goal_radius = self.reconstruct_obs(context_cond, idx)
        trajs_unnormalized = (trajs * 10).tolist() # temp
        
        visualize_paths_with_rewards(trajs_unnormalized, rewards, obstacles_list, 
                goal_pos=goal, goal_radius=goal_radius, 
                grid_size=20, dt=0.25, 
                save_name=save_name, frame_size=frame_size)
        return chain
    
class GassianDiffusionPPO(GaussianDiffusion):
    def __init__(self, model, noise_scheduler, n_timesteps, **kwargs):
        super().__init__(model, noise_scheduler, n_timesteps, **kwargs)
        print("Gaussian Diffusion PPO Model")
    
    def p_sample_loop(self, shape, sample_fn=diffusion_sample_fn_log_prob, diffusion_context_cond=None, state_cond=None, skip_x0=True, **additional_kwargs):
        """
        [for inference] sample x0 from random noise
        => initialize xt
        => for t=T to 0
        =>   convert t to tensor
        =>   xt = sample_fn(self, xt, t, context_cond) [can be either ddpm sample, ddim sample, or guided ddpm]
        =>   apply hard condition on xt [optional]
        """
        device = self.alpha.device
        batch_size = shape[0]
        
        xt = torch.randn(*shape, device=device)
        xt = self.apply_cond_fn(xt, state_cond) # apply cond at the beginning
        chain = [xt]
        log_probs = []
        
        final_xt = None
        for t_ in reversed(range(self.n_timesteps)):
            t = make_timesteps(t_, batch_size, device)
            xt, log_prob = sample_fn(self, xt, t, diffusion_context_cond=diffusion_context_cond, 
                                     state_cond=state_cond, **additional_kwargs)
            xt = self.apply_cond_fn(xt, state_cond) # apply cond at the end
            if log_prob is None and skip_x0:
                assert t_ == 0 # last timestep does not need to calculate log prob since we are not sampling from it
                final_xt = xt
                break
            
            chain.append(xt)
            log_probs.append(log_prob)
            final_xt = xt
            
        return final_xt, chain, log_probs