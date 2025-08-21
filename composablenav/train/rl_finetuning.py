# set up env
import os
import torch 
import hydra 
import numpy as np
from pathlib import Path
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

from composablenav.train.train_utils import load_model
from composablenav.train.train_utils import gif_to_tensor

DEVICE = "cuda"
    
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    save_dir = Path(cfg.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_config_path = save_dir / "config.yaml"

    # Load config from checkpoint folder
    if checkpoint_config_path.exists():
        print(f"Found config from {checkpoint_config_path}")
        cfg = OmegaConf.load(checkpoint_config_path)
    else:
        unresolved_conf = OmegaConf.to_container(cfg, resolve=False)
        print(f"Saving config file to {save_dir}")
        with open(checkpoint_config_path, "w") as file:
            OmegaConf.save(unresolved_conf, file)
            
    model = instantiate(cfg.model)
    model = load_model(model, cfg.checkpoint, DEVICE).train()
    env = instantiate(cfg.env)
    agent = instantiate(cfg.agent, actor=model, device=DEVICE)

    tb_dir = save_dir / "version_0"
    writer = SummaryWriter(log_dir=str(tb_dir))

    num_rollout = cfg.num_rollout
    num_episodes = cfg.num_episodes
    save_at_epochs = cfg.save_at_epochs
    
    # training loop
    for episode in range(num_episodes):
        print(f"============= TRAINING EPISODE: {episode} =============" )
        batch_data = env.generate_batch(num_rollout)
        avg_reward = agent.get_batched_rollout(batch_data)
        agent.compute_advantage()
        agent.update_policy()
        print(f"Average Reward: {avg_reward}")
        writer.add_scalar("train/avg_reward", float(avg_reward), episode)
        if (episode+1) % save_at_epochs == 1:
            plot_path = save_dir / "plots"
            os.makedirs(plot_path, exist_ok=True)
            gif_path = agent.visualize_evaluate_episode(plot_path, batch_data)

            output_frames = gif_to_tensor(gif_path)

            # Try to add videos with error handling
            writer.add_video("Validation/Model_Generated", output_frames, 
                            global_step=episode, fps=4)
            
            model_weights_path = f"{save_dir}/last.ckpt"
            torch.save(agent.actor.state_dict(), model_weights_path)
            print("MODEL SAVED")

if __name__ == "__main__":
    # set seed
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.set_float32_matmul_precision("highest")
    main()