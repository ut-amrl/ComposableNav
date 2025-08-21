# Had to add this for Module import errors
# import sys
# from pathlib import Path
# project_root = Path(__file__).parent.parent.parent
# sys.path.insert(0, str(project_root))

import os
import torch 
import lightning as pl 
import numpy as np 
from pathlib import Path
from copy import deepcopy
import hydra 
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from composablenav.train.train_utils import gif_to_tensor

class EMA:
    def __init__(self, beta, step_start_ema):
        self.beta = beta
        self.step_start_ema = step_start_ema
        self.step = 0
    
    def update_params(self, ema_model_params, new_model_params):
        if ema_model_params is None:
            raise ValueError("ema_model_params is None")
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
    
class DiffusionTrainerModel(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super(DiffusionTrainerModel, self).__init__()
        self.cfg = cfg
        self.lr = cfg.lr
        self.dt = cfg.dt

        self.model = instantiate(cfg.model)
        self.ema = EMA(cfg.ema.beta, cfg.ema.step_start_ema)
        self.model_ema = deepcopy(self.model).eval().requires_grad_(False) # create a copy of the model for EMA

        self.process_batch_func = instantiate(cfg.process_batch_func)
        self.validate_func = instantiate(cfg.validate_func)
        self.visualizer_func = instantiate(cfg.visualizer_func)
        self.normalize_func = instantiate(cfg.normalize_func)
        self.sample_func = instantiate(cfg.sample_func)
        
    def training_step(self, batch, batch_idx):
        input_data = self.process_batch_func(batch)
        loss = self.model.loss(**input_data)
        self.log('train_loss', loss, prog_bar=True, logger=True, on_epoch=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_data = self.process_batch_func(batch)

        self.validate_func(**input_data, model=self.model, dt=self.dt, 
                           path_normalizer=self.normalize_func, logger=self.log, 
                           log_name="no_collision_rate")
        self.validate_func(**input_data, model=self.model_ema, dt=self.dt, 
                           path_normalizer=self.normalize_func, logger=self.log, 
                           log_name="no_collision_rate_ema")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
    
    def on_validation_batch_end(
        self, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Called when the validation batch ends."""
        if batch_idx != 0:
            return
        
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', current_lr, prog_bar=True)
        if not self.trainer.is_global_zero: # only log on the first process
            return
        idx = np.random.randint(0, len(batch["filename"])) # randomly sample an index to plot
        input_data = self.process_batch_func(batch)
        x0 = input_data["x0"]
        
        context_cond = input_data["context_cond"]
        state_cond = input_data["state_cond"]
        filenames = batch["filename"]

        output_path = Path(self.cfg.plot_dir) / "model_output.gif"
        output_ema_path = Path(self.cfg.plot_dir) / "model_ema_output.gif"
        os.makedirs(self.cfg.plot_dir, exist_ok=True)
        self.visualizer_func(self.model, x0.shape, idx,
                            self.sample_func, context_cond, state_cond, filenames, save_name=output_path)
        self.visualizer_func(self.model_ema, x0.shape, idx,
                            self.sample_func, context_cond, state_cond, filenames, save_name=output_ema_path)

        output_frames = gif_to_tensor(output_path)
        output_ema_frames = gif_to_tensor(output_ema_path)

        writer = self.logger.experiment

        # Try to add videos with error handling
        try:
            writer.add_video("Validation/Model_Generated", output_frames, 
                            global_step=self.global_step, fps=4)
            writer.add_video("Validation/Model_EMA_Generated", output_ema_frames, 
                            global_step=self.global_step, fps=4)
            writer.flush()
            print(f"Videos logged to TensorBoard at step {self.global_step}")
        except Exception as e:
            print(f"Video logging failed: {e}")

    def on_before_zero_grad(self, *args, **kwargs):
        # apply EMA 
        if self.global_rank == 0:
            self.ema.step_ema(self.model_ema, self.model)
        
    def on_save_checkpoint(self, checkpoint):
        # Include EMA weights in the checkpoint
        if self.global_rank == 0:
            checkpoint['ema_state_dict'] = self.model_ema.state_dict()
   
    
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train(cfg: DictConfig):
    save_dir = Path(cfg.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_config_path = save_dir / "config.yaml"
    pre_existing_checkpoint = save_dir / "last.ckpt"

    # Load config from checkpoint folder
    if checkpoint_config_path.exists():
        print(f"Found config from {checkpoint_config_path}")
        cfg = OmegaConf.load(checkpoint_config_path)
    else:
        unresolved_conf = OmegaConf.to_container(cfg, resolve=False)
        print(f"Saving config file to {save_dir}")
        with open(checkpoint_config_path, "w") as file:
            OmegaConf.save(unresolved_conf, file)

    # load from existing checkpoint if exists
    if pre_existing_checkpoint.exists():
        print(f"Found latest checkpoint at {pre_existing_checkpoint}")
        cfg.checkpoint = pre_existing_checkpoint

    model_trainer = instantiate(cfg.model_trainer)
    dataloader = instantiate(cfg.dataloader)
    pl_trainer = instantiate(cfg.trainer)
    pl_trainer.fit(model_trainer, dataloader, ckpt_path=cfg.checkpoint)
    
if __name__ == "__main__":

    # set seed
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.set_float32_matmul_precision("highest")
    train()