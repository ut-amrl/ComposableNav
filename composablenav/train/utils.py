from omegaconf import DictConfig, OmegaConf
import torch 
from functools import partial
import numpy as np 

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from composablenav.misc.common import put_batch_on_device
from composablenav.misc.normalizers import TensorDifferentialPathNormalizer
from composablenav.models.temporal_unet_cond import TemporalUnetCond
from composablenav.train.dataloader import CustomDataLoader
from composablenav.models import diffusion_components
from composablenav.models.diffusion import GaussianDiffusion, GassianDiffusionPPO
from composablenav.models import loss

class EMA:
    def __init__(self, beta, step_start_ema):
        self.beta = beta
        self.step_start_ema = step_start_ema
        self.step = 0
    
    def update_params(self, ema_model_params, new_model_params):
        if ema_model_params is None:
            raise ValueError("ema_model_params is None")
            return new_model_params
        return ema_model_params * self.beta + new_model_params * (1 - self.beta) 
        
    def update_model_average(self, ema_model, current_model):
        for ema_param, current_param in zip(ema_model.parameters(), current_model.parameters()):
            ema_param.data = self.update_params(ema_param.data, current_param.data)
    
    def step_ema(self, ema_model, model):
        self.step += 1
        if self.step <= self.step_start_ema: 
            self.reset_parameters(ema_model, model)
            return 
        self.update_model_average(ema_model, model)
    
    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())
        ema_model.eval() # ensure the model is in eval mode

def initialize_diffusion_model(cfg: DictConfig, load_ddpo=False):
    model_cfg = cfg.model
    data_cfg = cfg.dataset_generation
    unet_model = TemporalUnetCond(
        input_dim=model_cfg.unet.input_dim, 
        hidden_dim=model_cfg.unet.hidden_dim, 
        dim_head=model_cfg.unet.dim_head,
        num_heads=model_cfg.unet.num_heads,
        dropout=model_cfg.unet.dropout,
        context_args=model_cfg.context
        )
    loss_fn = getattr(loss, model_cfg.diffusion.loss_fn)(**model_cfg.diffusion.loss_fn_args)
    noise_scheduler = getattr(diffusion_components, model_cfg.diffusion.noise_scheduler)

    differential_normalizer = TensorDifferentialPathNormalizer(grid_size=data_cfg.env.grid_size, max_angle=torch.pi, 
                                                               max_vel=data_cfg.robot.max_v, max_ang_vel=data_cfg.robot.max_w)
    apply_cond_fn = getattr(diffusion_components, model_cfg.diffusion.apply_cond_fn)
    apply_cond_fn = partial(apply_cond_fn, differential_normalizer=differential_normalizer, env_dt=data_cfg.env.env_dt)
    # print("Using apply cond: ", model_cfg.diffusion.apply_cond_fn)
    
    model_stub = GassianDiffusionPPO if load_ddpo else GaussianDiffusion
    
    diffusion_model = model_stub(
        model=unet_model, 
        noise_scheduler=noise_scheduler, 
        n_timesteps=model_cfg.diffusion.n_timesteps, 
        clipped_denoised=model_cfg.diffusion.clipped_denoised,
        predict_epsilon=model_cfg.diffusion.predict_epsilon,
        cond_weight=model_cfg.diffusion.cond_weight,
        loss_fn=loss_fn,
        apply_cond_fn=apply_cond_fn,
        use_traj_mask=model_cfg.diffusion.use_traj_mask,
        compile=model_cfg.compile,
        **model_cfg.diffusion.cfg_args
    )

    return diffusion_model    

def initialize_dataloader(cfg: DictConfig):
    return CustomDataLoader(cfg)

def initialize_logger(cfg: DictConfig):
    wandb_logger = WandbLogger(project=cfg.project_name, name=cfg.run_name)
    return wandb_logger

def initialize_callbacks(cfg: DictConfig):
    # has a bug about what to monitor: I want to monitor val loss
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.checkpoint.monitor,
        dirpath=cfg.checkpoint.dirpath,
        filename = f'diffusion-{{epoch:02d}}-{{{cfg.checkpoint.monitor}:.5f}}', 
        save_top_k=cfg.checkpoint.save_top_k,  # Save top n
        mode=cfg.checkpoint.mode,
        every_n_epochs=cfg.checkpoint.every_n_epochs  # Save a checkpoint every 200 epochs
    )
    return [checkpoint_callback]

def diffusion_process_batch_mm(batch, cfg, device=None):
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

def ddpo_process_batch_mm(batch, cfg, device=None):
    start = batch["start"] if device is None else batch["start"].to(device)
    
    num_env = start.shape[0]
    x_shape = torch.tensor([num_env, cfg.data.dataset_args.max_padded_len, cfg.model.unet.input_dim])

    diffusion_context_cond = put_batch_on_device(batch["context_cond"], device)

    # tbd for random start later
    if cfg.model.unet.input_dim == 2:
        start = start 
    elif cfg.model.unet.input_dim == 3:
        start = torch.cat([start, torch.zeros_like(start)[..., :1]], dim=-1)
        assert start.shape[-1] == 3
    state_cond = {
        "0": start,
    }
    # important the name should not be changed
    return x_shape, diffusion_context_cond, state_cond

def load_model(cfg, model_path=None, device="cuda", load_ddpo=False):
    model = initialize_diffusion_model(cfg, load_ddpo)
    state_dict = torch.load(model_path)
    # if there is ema model, load it
    if state_dict.get("ema_state_dict"):
        # print("[Preprocess] Loading EMA model...")
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

# for validating and visualizing during training
from composablenav.misc.common import load_data, construct_obstacle_from_info
from composablenav.models.diffusion_components import diffusion_sample_fn
from composablenav.misc.critic import collision_criteria, goal_reaching_criteria
from composablenav.misc.visualizer_utils import visualize_paths_with_rewards

def info_obs_goal_from_fn(fn):
    data = load_data(fn)
    info = data["info"]
    dynamic_obstacles = [construct_obstacle_from_info(info) for info in data["obs"]["obstacles_info"]]
    static_obstacles = [construct_obstacle_from_info(info) for info in data["obs"]["terrain_info"]]
    obstacles_list = dynamic_obstacles + static_obstacles
    goal = np.array(data["trajectories"]["0"]["goal"]) * (info["grid_size"] // 2) # hardcoded
    return info, obstacles_list, goal

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
    rewards = np.linspace(0, 1, trajs.shape[0]+1)
    trajs_unnormalized = (trajs * grid_size / 2).tolist()
    
    if state_cond.get("start_time_idx") is None:
        start_time_idx = 0
    else:
        start_time_idx = state_cond.get("start_time_idx").detach().cpu().numpy()[idx]
        
    visualize_paths_with_rewards(trajs_unnormalized, rewards, obstacles_list, 
            goal_pos=goal, goal_radius=goal_radius, 
            grid_size=grid_size, dt=dt, 
            save_name=save_name, frame_size=frame_size, start_time_idx=start_time_idx)
    return chain