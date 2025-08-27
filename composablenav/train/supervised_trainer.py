import torch 
import pytorch_lightning as pl 
import numpy as np 
import os
import hydra 
from omegaconf import DictConfig, OmegaConf
from abc import ABC, abstractmethod
from composablenav.train.utils import (
    initialize_diffusion_model, 
    initialize_dataloader, 
    initialize_logger, 
    initialize_callbacks,  
    diffusion_process_batch_mm,
    validate, 
    visualize,
    EMA
)
from composablenav.misc.normalizers import PathNormalizer
from composablenav.misc.visualizer_utils import visualize_diffusion
from composablenav.models.diffusion_components import diffusion_sample_fn
import wandb
from copy import deepcopy
from pytorch_lightning.utilities import grad_norm
from pytorch_lightning.strategies import DDPStrategy

class BaseTrainerModel(pl.LightningModule, ABC):
    def __init__(self, model, cfg: DictConfig):
        super(BaseTrainerModel, self).__init__()
        self.trainer_model = model
        self.cfg = cfg
        self.path_normalizer = PathNormalizer(self.cfg.dataset_generation.env.grid_size)
    
    @abstractmethod
    def process_batch(self, batch):
        ... 
        
    def training_step(self, batch, batch_idx):
        input_data = self.process_batch(batch)
        loss = self.trainer_model.loss(**input_data)
        self.log('train_loss', loss, prog_bar=True, logger=True, on_epoch=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_data = self.process_batch(batch)
        result = self.trainer_model.validate(**input_data, path_normalizer=self.path_normalizer, logger=self.log)
        val_score = result["val_score"]
        
        self.log('val_score', val_score, prog_bar=True, logger=True, on_epoch=True)
        return val_score

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.train.optim.lr)

        return optimizer
    
class DiffusionTrainerModel(BaseTrainerModel):
    def __init__(self, model, cfg: DictConfig):
        super(DiffusionTrainerModel, self).__init__(model, cfg)
        self.use_ema = cfg.train.optim.ema.use_ema
        if self.use_ema:
            print("USING EMA")
            self.ema = EMA(cfg.train.optim.ema.beta, cfg.train.optim.ema.step_start_ema)
            self.ema_model = deepcopy(self.trainer_model).eval().requires_grad_(False) # create a copy of the model for EMA
        else:
            print("NOT USING EMA")

    def process_batch(self, batch):
        return diffusion_process_batch_mm(batch, self.cfg.model.context)
    
    def validation_step(self, batch, batch_idx):
        input_data = self.process_batch(batch)
        dt = self.cfg.dataset_generation.env.env_dt
        
        validate(**input_data, model=self.trainer_model, dt=dt, path_normalizer=self.path_normalizer, logger=self.log, log_name="val_loss")
        if self.use_ema:
            validate(**input_data, model=self.ema_model, dt=dt, path_normalizer=self.path_normalizer, logger=self.log, log_name="val_loss_ema")
        
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
        input_data = self.process_batch(batch)
        x0 = input_data["x0"]
        
        sample_fn = diffusion_sample_fn
        context_cond = input_data["context_cond"]
        state_cond = input_data["state_cond"]
        filenames = batch["filename"]
        visualize(self.trainer_model, x0.shape, idx, 
                            sample_fn, context_cond, state_cond, filenames, save_name="diffusion")
        
        self.trainer.logger.experiment.log({"diffusion": wandb.Video("diffusion.gif", format="gif")})
        
        if self.use_ema:
            visualize(self.ema_model, x0.shape, idx, 
                            sample_fn, context_cond, state_cond, filenames, save_name="diffusion_ema")
        
            self.trainer.logger.experiment.log({"diffusion_ema": wandb.Video("diffusion_ema.gif", format="gif")})

    def on_before_zero_grad(self, *args, **kwargs):
        # apply EMA 
        if self.global_rank == 0 and self.use_ema:
            self.ema.step_ema(self.ema_model, self.trainer_model)
        
    def on_save_checkpoint(self, checkpoint):
        # Include EMA weights in the checkpoint
        if self.global_rank == 0 and self.use_ema:
            checkpoint['ema_state_dict'] = self.ema_model.state_dict()

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.trainer_model.model, norm_type=2)
        self.log_dict(norms)      
    
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train(cfg: DictConfig):
    # print(OmegaConf.to_yaml(cfg)) # print the config
    dataloader = initialize_dataloader(cfg.data)
    model = initialize_diffusion_model(cfg)
    wandb_logger = initialize_logger(cfg.log)
    callbacks = initialize_callbacks(cfg.train.callbacks)
    trainer_model = eval(cfg.train.trainer.trainer_model_name)(model, cfg)
    os.makedirs(cfg.train.callbacks.checkpoint.dirpath, exist_ok=True)

    trainer = pl.Trainer(
        max_epochs=cfg.train.trainer.max_epochs,
        devices=cfg.train.trainer.devices, 
        accelerator=cfg.train.trainer.accelerator, 
        precision=cfg.train.trainer.precision,
        log_every_n_steps=cfg.log.log_every_n_steps,
        gradient_clip_val=cfg.train.trainer.gradient_clip_val,
        check_val_every_n_epoch=cfg.train.trainer.check_val_every_n_epoch,
        logger=wandb_logger,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        strategy=DDPStrategy(find_unused_parameters=True),
    )
    trainer.fit(trainer_model, dataloader)
    # trainer.validate(trainer_model, dataloader)
    
if __name__ == "__main__":
    # set seed
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.set_float32_matmul_precision("highest")
    train()