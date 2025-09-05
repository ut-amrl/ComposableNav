# set up env
from composablenav.models.ddpo_agent import DDPOAgent, RewardModel, Maze2DEnv, ReplayBuffer, SuccessEvaluator
from composablenav.train.utils import load_model, ddpo_process_batch_mm
import hydra 
from omegaconf import DictConfig
import torch 
import os
import numpy as np
import wandb 
torch.set_float32_matmul_precision('high')

DEVICE = "cuda"

def initailize_DDPO_env_agent(cfg: DictConfig):
    diffusion_model = load_model(cfg, model_path=cfg.train.diffusion_checkpoints, device=DEVICE, load_ddpo=True) 
    diffusion_model.train() # set it to be training model
    evaluator_model = SuccessEvaluator(cfg)
    reward_model = RewardModel(cfg.dataset_generation, cfg.primitive.scenario)
    replay_buffer = ReplayBuffer(batch_size=cfg.train.inner_loop_mini_batch_size, device=DEVICE)
    agent = DDPOAgent(cfg, diffusion_model, reward_model, replay_buffer, evaluator_model, cfg.train.log_at_steps, 
                      cfg.train.inner_loop_epochs, cfg.train.epislon_clip, cfg.train.optim.lr, 
                      use_kl=cfg.train.use_kl, kl_beta=cfg.train.kl_beta)
    
    env = Maze2DEnv(cfg.dataset_generation, 
                    max_obj_traj_len=cfg.data.dataset_args.max_obj_traj_len,
                    max_padded_obj_num=cfg.data.dataset_args.max_padded_obj_num,
                    max_padded_terrain_num=cfg.data.dataset_args.max_padded_terrain_num,)
    print("Environment and Agent Initialized")
    return env, agent
    
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    env, agent = initailize_DDPO_env_agent(cfg)
    save_dir = f"exponential_ppo_exps/{cfg.train.save_name}"
    os.makedirs(save_dir, exist_ok=True)
    wandb.init(project=cfg.log.project_name, name=f"{cfg.train.save_name}")

    # training loop
    for episode in range(cfg.train.num_episodes):
        print(f"============= TRAINING EPISODE: {episode} =============" )
        batch_data = env.generate_batch(cfg.train.num_rollout)
        raw_env_info = batch_data["raw_env_info"]
        x_shape, diffusion_context_cond, state_cond = ddpo_process_batch_mm(batch_data, cfg, device=DEVICE)
        avg_reward = agent.get_batched_rollout(x_shape, diffusion_context_cond, state_cond, raw_env_info)
        agent.compute_advantage()
        agent.update_policy()
        # agent.update_policy_fix_state()
        print(f"Average Reward: {avg_reward}")
        wandb.log({"average_reward": avg_reward}, step=episode)
        if (episode+1) % cfg.train.save_at_epochs == 1:
            save_name, evaluated_results = agent.visualize_evaluate_episode(x_shape, diffusion_context_cond, state_cond, raw_env_info)
            
            wandb.log({f"episode_result": wandb.Video(save_name, format="gif")}, step=episode)
            wandb.log(evaluated_results, step=episode)
            
            model_weights_path = f"{save_dir}/diffusion_{episode}_{avg_reward:.3f}.ckpt"
            torch.save(agent.actor.state_dict(), model_weights_path)
            print("MODEL SAVED")

if __name__ == "__main__":
    # set seed
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.set_float32_matmul_precision("highest")
    main()