"""
A simple implementation of PPO 
"""
import torch 
import numpy as np
from torch.distributions import Normal
from einops import rearrange
from copy import deepcopy
from tqdm import tqdm 
from collections import deque
from omegaconf import DictConfig
from hydra.utils import instantiate

from composablenav.misc.visualize_utils import plot_path_gif
from composablenav.misc.common import construct_obstacle_from_info, find_first_waypoint_within_radius
from composablenav.misc.process_data import construct_normalized_obstacle_seq, construct_normalized_static_obstacle_from_obj
from composablenav.datasets.scenario_generator import gen_noncolliding_obstacles, gen_regions, gen_goal

class Maze2DEnv:
    def __init__(self, cfg: DictConfig):
        self.dataset = cfg.dataset
        self.path_normalizer = instantiate(cfg.path_normalizer)
        self.process_obs = instantiate(cfg.dataloader)
    
    def generate_batch(self, batch_size):
        raw_env_info = []
        obs = []
        obs_mask = []
        avoid_region = []
        avoid_region_mask = []
        walk_over = []
        walk_over_mask = []
        start = []
        goal = []
        count = 0
        while True:
            if count >= batch_size: # ensure batch size can always be met
                break
            env_info = self.generate_one()
            if env_info is None:
                continue
            obs_tensor, obs_mask_tensor, avoid_region_tensor, avoid_region_mask_tensor, \
            walk_over_tensor, walk_over_mask_tensor, start_tensor, goal_tensor = self.build_context_tensor(env_info)
            raw_env_info.append(env_info)
            obs.append([o for o in obs_tensor])
            obs_mask.append(obs_mask_tensor)
            
            avoid_region.append([t for t in avoid_region_tensor])
            avoid_region_mask.append(avoid_region_mask_tensor)
            
            walk_over.append([t for t in walk_over_tensor])
            walk_over_mask.append(walk_over_mask_tensor)
            
            start.append(start_tensor)
            goal.append(goal_tensor)
            count += 1
        batch = {
            "context_cond": {
                "dynamic_obs_encoder": {
                    "mask": torch.stack(obs_mask).float(),
                    "input_vals": [torch.stack(o) for o in zip(*obs)]
                },
                "static_obs_encoder": {
                    "mask": torch.stack(avoid_region_mask).float(),
                    "input_vals": [torch.stack(t) for t in zip(*avoid_region)]
                },
                "goal_encoder": {
                    "mask": torch.ones([len(goal), 1]),                    
                    "input_vals": torch.stack(goal).float(),
                } 
                ,
                "terrain_encoder": {
                    "mask": torch.stack(walk_over_mask).float(),
                    "input_vals": [torch.stack(t) for t in zip(*walk_over)]
                }
            },
            "start": torch.stack(start).float(),
            "raw_env_info": raw_env_info
        }
        return batch
    
    def create_obstacle_info(self, obstacle_list):
        start_loc = self.dataset.objective.start_loc
                
        return [obstacle.to_dict() for obstacle in obstacle_list], start_loc
    
    def generate_one(self):
        if self.dataset.scenarios.scenario == "avoid":
            obstacle_list = []
            avoid_region_list = gen_regions(self.dataset)
            walk_over_list = []
        elif self.dataset.scenarios.scenario == "walk_over":
            obstacle_list = []
            walk_over_list = gen_regions(self.dataset)
            avoid_region_list = []
        else:
            obstacle_list = gen_noncolliding_obstacles(self.dataset)
            walk_over_list = []
            avoid_region_list = []
        goal_loc = gen_goal(self.dataset, obstacle_list, walk_over_list, scenario=self.dataset.scenarios.scenario)
        goal_radius = self.dataset.objective.goal_radius
        
        if goal_loc[0] is None or obstacle_list is None or walk_over_list is None or avoid_region_list is None:
            return None
        
        obstacles_info, start_loc = self.create_obstacle_info(obstacle_list)
        avoid_region_info = [avoid_region.to_dict() for avoid_region in avoid_region_list]
        walk_over_info = [walk_over.to_dict() for walk_over in walk_over_list]

        return [obstacles_info, avoid_region_info, walk_over_info, start_loc, goal_loc, goal_radius] # not handling static for now
    
    def build_context_tensor(self, env_info):
        obstacles_info, avoid_region_info, walk_over_info, start_loc, goal_loc, goal_radius = env_info
        obs_tensor, obs_mask_tensor = self.build_normalized_obstacle_seq(obstacles_info)
        walk_over_tensor, walk_over_mask_tensor = self.build_normalized_walk_over(walk_over_info)
        avoid_region_tensor, avoid_region_mask_tensor = self.build_normalized_avoid_region(avoid_region_info)
        start_tensor = self.build_loc_normalized(start_loc)
        goal_tensor = self.build_loc_normalized(goal_loc)
        return obs_tensor, obs_mask_tensor, avoid_region_tensor, avoid_region_mask_tensor, \
               walk_over_tensor, walk_over_mask_tensor, start_tensor, goal_tensor
        
    def build_normalized_obstacle_seq(self, obstacles_info):
        normalized_obstacle_seq = construct_normalized_obstacle_seq(obstacles_info, 
                                                                    grid_size=self.dataset.env.grid_size, 
                                                                    max_planning_time=self.dataset.robot.max_planning_time, 
                                                                    dt=self.dataset.env.env_dt,
                                                                    offset_x=0, offset_y=0, mult_x=1, mult_y=1
                                                                    )
        obs, obs_mask = self.process_obs.get_obs_cond(normalized_obstacle_seq)
        return obs, obs_mask

    def build_normalized_walk_over(self, region_info):
        # temporary because region has a different encoder from static
        region_list = [construct_obstacle_from_info(info) for info in region_info]
        normalized_regions = []
        for region in region_list:
            top, bottom, left, right = region.get_repr()
            normalized_regions.append([top/10, bottom/10, left/10, right/10, 0]) # TODO hardcoded
        regions, region_mask = self.process_obs.get_region_cond(normalized_regions)
        return regions, region_mask
    
    def build_normalized_avoid_region(self, region_info):
        static_list = [construct_obstacle_from_info(info) for info in region_info]
        normalized_statics = construct_normalized_static_obstacle_from_obj(static_list, grid_size=self.dataset.env.grid_size, 
                                                                            mult_x=1, mult_y=1, offset_x=0, offset_y=0)
        statics, static_mask = self.process_obs.get_static_cond(normalized_statics)
        return statics, static_mask
    
    def build_loc_normalized(self, loc):
        loc_tensor = torch.tensor(loc)
        return self.path_normalizer.normalize(loc_tensor)

class RewardModel:
    def __init__(self, cfg: DictConfig):
        self.env_dt = cfg.env_dt
        
        self.path_normalizer = instantiate(cfg.path_normalizer)   
        self.criteria = instantiate(cfg.criteria)[cfg.primitive]
        
    def decode_trajectory(self, trajectory, goal, goal_radius):
        if len(trajectory.shape) == 2:
            trajectory = trajectory.unsqueeze(0)
        elif len(trajectory.shape) == 3:
            pass
        else:
            raise ValueError("Invalid trajectory shape")
        traj = trajectory.detach().cpu().numpy()
        # truncate trajectory
        unnormalized_traj = self.path_normalizer.unnormalize(traj)[0]
        idx = find_first_waypoint_within_radius(unnormalized_traj, goal, goal_radius)
        if idx != -1:
            unnormalized_traj = unnormalized_traj[:idx+1]
        return [unnormalized_traj.tolist()]
    
    def decode_loc(self, loc):
        loc = np.array(loc)
        return self.path_normalizer.unnormalize(loc).tolist()
    
    def compute_reward(self, trajectory, env_info):
        obstacle_info, avoid_region_info, walk_over_info, start, goal, goal_radius = env_info
        traj = self.decode_trajectory(trajectory, goal, goal_radius)
        vis_rewards = np.zeros(2)
        obstacle_list = [construct_obstacle_from_info(info) for info in obstacle_info]
        avoid_region_list = [construct_obstacle_from_info(info) for info in avoid_region_info]
        walk_over_list = [construct_obstacle_from_info(info) for info in walk_over_info]
        
        obs_terr_list = obstacle_list + avoid_region_list + walk_over_list
        reward = self.criteria(traj[0], obstacles=obs_terr_list, goal=goal, dt=self.env_dt)
        return reward

    def compute_success_criteria(self, trajectory, env_info):
        obstacle_info, avoid_region_info, walk_over_info, start, goal, goal_radius = env_info
        traj = self.decode_trajectory(trajectory, goal, goal_radius)
        obstacle_list = [construct_obstacle_from_info(info) for info in obstacle_info]
        return self.criteria(traj[0], obstacles=obstacle_list, goal=goal, dt=self.env_dt)
    
class ReplayBuffer:
    def __init__(self, cfg: DictConfig, device):
        self.inner_loop_mini_batch_size = cfg.inner_loop_mini_batch_size
        self.states = []
        self.actions = []
        self.rewards = []
        self.advantages = []
        self.log_probs = []
        self.final_states = []
        self.device = device
    
    def add_to_buffer(self, state, action, reward, log_probs, final_state):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_probs)
        self.final_states.append(final_state)
        
    def set_advantages(self, advantages):
        self.advantages = advantages
    
    def stack_dictionaries_list(self, dicts_list):
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
                stacked_dict[key] = self.stack_dictionaries_list(values)
            
            elif isinstance(values[0], torch.Tensor):
                # Stack tensors along a new dimension
                stacked_dict[key] = torch.stack(values).to(self.device)
            elif isinstance(values[0], list) and isinstance(values[0][0], torch.Tensor):
                # Recursively handle nested dictionaries
                stacked_dict[key] = [torch.stack(o) for o in zip(*values)]
            else:
                # If values are not tensors, you can customize this part as needed
                stacked_dict[key] = values
        
        return stacked_dict

    def helper_build_states(self, states):
        # assume states are of the same form
        xts = []
        ts = []
        diffusion_context_cond_list = []
        state_cond_list = []
        
        for state in states:
            xt, t, diffusion_context_cond, state_cond = state
            xts.append(xt)
            ts.append(t)
            diffusion_context_cond_list.append(diffusion_context_cond)
            state_cond_list.append(state_cond)

        # Merge dictionaries from a list of dictionaries
        diffusion_context_cond_tensor = self.stack_dictionaries_list(diffusion_context_cond_list)
        
        state_cond_merged = {key: [d[key] for d in state_cond_list] for key in state_cond_list[0]}
        state_cond_tensor = {key: torch.stack(value).to(self.device) for key, value in state_cond_merged.items()}
        
        xts = torch.stack(xts).to(self.device)
        ts = torch.stack(ts).to(self.device)
        return (xts, ts, diffusion_context_cond_tensor, state_cond_tensor)
        
    def collate(self, sample_batch):
        states_tensor = self.helper_build_states(sample_batch["states"])
        actions_tensor = torch.stack(sample_batch["actions"]).to(self.device)
        log_probs_tensor = torch.stack(sample_batch["log_probs"]).to(self.device)
        advantages_tensor = torch.stack(sample_batch["advantages"]).to(self.device)
        final_states_tensor = torch.stack(sample_batch["final_states"]).to(self.device)
        return {
            "states": states_tensor,
            "actions": actions_tensor,
            "log_probs": log_probs_tensor,
            "advantages": advantages_tensor,
            "final_states": final_states_tensor
        }
    
    def sample(self):
        assert len(self.states) == len(self.actions) == len(self.rewards) == len(self.log_probs) == len(self.advantages)
        NT = len(self.states)
        batch_indices = np.arange(NT) 
        np.random.shuffle(batch_indices)
        batch_indices = batch_indices[: (NT // self.inner_loop_mini_batch_size) * self.inner_loop_mini_batch_size] # truncate
        batch_indices = batch_indices.reshape(-1, self.inner_loop_mini_batch_size)
        
        samples = []
        for batch_idx in batch_indices:
            states = [self.states[i] for i in batch_idx]
            actions = [self.actions[i] for i in batch_idx]
            rewards = [self.rewards[i] for i in batch_idx]
            log_probs = [self.log_probs[i] for i in batch_idx]
            advantages = [self.advantages[i] for i in batch_idx]
            final_states = [self.final_states[i] for i in batch_idx]
            
            samples.append({
                "states": states,
                "actions": actions,
                "rewards": rewards,
                "log_probs": log_probs,
                "advantages": advantages,
                "final_states": final_states
            })
        return samples
    
    def __len__(self):
        return len(self.states) // self.inner_loop_mini_batch_size
        
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.advantages = []
        self.log_probs = []

class DDPOAgent:
    def __init__(self, cfg, actor, device):
        self.actor = actor
        self.original_actor = deepcopy(actor)
        self.actor_old = deepcopy(actor)
        self.reward_model = instantiate(cfg.reward_model)
        self.replay_buffer = instantiate(cfg.replay_buffer, device=device)
        self.process_batch_func = instantiate(cfg.process_batch_func, device=device)

        self.log_at_steps = cfg.log_at_steps
        self.inner_loop_epochs = cfg.inner_loop_epochs
        self.epislon_clip = cfg.epislon_clip
        self.use_kl = cfg.use_kl
        self.kl_beta = cfg.kl_beta
        self.min_count_advantage_normalization = cfg.min_count_advantage_normalization
        self.grid_size = cfg.grid_size
        self.env_dt = cfg.env_dt

        self.advantage_deque = deque(maxlen=cfg.advantage_deque_length)
        self.optimizer = torch.optim.AdamW(self.actor.parameters(), lr=cfg.lr)
        if self.use_kl:
            print("Using KL divergence loss")

    def helper_retrieve_state_dict(self, idx, context):
        new_context = {}
        for k, v in context.items():
            new_context[k] = v[idx]
        return new_context  
    
    def helper_retrieve_context_dict(self, idx, context):
        context = deepcopy(context)
        new_context = {}
        for encoder_name, encoder_dict in context.items():
            new_context[encoder_name] = {}
            if len(encoder_dict) == 0:
                continue
            if encoder_name == "dynamic_obs_encoder" or encoder_name == "terrain_encoder" or encoder_name == "static_obs_encoder":
                new_context[encoder_name]["mask"] = encoder_dict["mask"][idx]
                new_context[encoder_name]["input_vals"] = [obs[idx] for obs in encoder_dict["input_vals"]]
            else:
                new_context[encoder_name]["mask"] = encoder_dict["mask"][idx]
                new_context[encoder_name]["input_vals"] = encoder_dict["input_vals"][idx]
        return new_context  
        
    def get_batched_rollout(self, batch_data):
        x_shape, diffusion_context_cond, state_cond, raw_env_info = self.process_batch_func(batch_data)

        self.actor_old.load_state_dict(self.actor.state_dict())
        self.actor_old.eval()
        N = x_shape[0] # rollout for N trajectories
        T = self.actor.n_timesteps
        
        with torch.no_grad():
            final_xt, chain, log_probs = self.actor.p_sample_loop(x_shape, diffusion_context_cond=diffusion_context_cond, 
                                         state_cond=state_cond, skip_x0=True)

        assert len(chain) == T # [important] we are not using x0 because of batch instability???
        assert len(log_probs) == T - 1

        chain = torch.stack(chain[::-1]) # make it in forward order
        log_probs = torch.stack(log_probs[::-1])
        chain_batched = rearrange(chain, 'T N ... -> N T ...') # reorder to batch first
        log_probs_batched = rearrange(log_probs, 'T N ... -> N T ... ')
        avg_reward = 0
        for i in range(N):
            trajectory = chain_batched[i]
            log_prob = log_probs_batched[i]
            diff_context = self.helper_retrieve_context_dict(i, diffusion_context_cond)
            state_context = self.helper_retrieve_state_dict(i, state_cond)
            
            reward = self.reward_model.compute_reward(final_xt[i], raw_env_info[i])
            avg_reward += reward
            for j in range(T - 1):
                t = j + 1 # skip the first timestep
                s_curr = (trajectory[j+1], torch.tensor(t), diff_context, state_context)
                s_next = trajectory[j]
                lp = log_prob[j]

                self.replay_buffer.add_to_buffer(state=s_curr, action=s_next, reward=reward, log_probs=lp, final_state=final_xt[i])
        avg_reward /= N
        return avg_reward
                
    def compute_advantage(self):
        """
        just normalize rewards for now, same as ddpo where I use a window to keep track
        """
        self.advantage_deque.extend(self.replay_buffer.rewards)
        if len(self.advantage_deque) < self.min_count_advantage_normalization:
            mean = torch.mean(rewards)
            std = torch.std(rewards) + 1e-6
        else:

            mean = torch.tensor(np.mean(self.advantage_deque)).float()
            std = torch.tensor(np.std(self.advantage_deque)).float() + 1e-6
        rewards = torch.tensor(self.replay_buffer.rewards).float()
        advantages = (rewards - mean) / std
        self.replay_buffer.set_advantages(advantages)
        
    def update_policy(self):  
        total_iterations = self.inner_loop_epochs * len(self.replay_buffer)
        running_loss = 0.0
        with tqdm(total=total_iterations, desc="Processing", unit="iteration") as pbar:
            for epoch in range(self.inner_loop_epochs):
                samples = self.replay_buffer.sample()
                for sample in samples:
                    batch = self.replay_buffer.collate(sample)
                    
                    states = batch["states"]
                    actions = batch["actions"]
                    log_probs_old = batch["log_probs"]
                    advantages = batch["advantages"]
                    final_states = batch["final_states"]
                    
                    xt, t, diffusion_context_cond, state_cond = states

                    # refactor later to be sameas sample_fn
                    mean, _, log_var = self.actor.p_mean_variance(xt, t, diffusion_context_cond)
                    std = torch.exp(0.5 * log_var)
                    std_clipped = torch.clip(std, min=1e-6) # to avoid numerical instability
                    normal_dist = Normal(mean, std_clipped)
                    log_prob_new = normal_dist.log_prob(actions)
                    log_prob_new = log_prob_new[:, 1:].mean(dim=list(range(1, log_prob_new.ndim)))
                    
                    # PPO loss
                    ratio = torch.exp(log_prob_new - log_probs_old)
                    surr1 = -ratio * advantages
                    surr2 = -torch.clamp(ratio, 1 - self.epislon_clip, 1 + self.epislon_clip) * advantages
                    
                    loss = torch.max(surr1, surr2).mean()
                    # KL divergence
                    if self.use_kl:
                        with torch.no_grad():
                            mean, _, log_var = self.original_actor.p_mean_variance(xt, t, diffusion_context_cond)
                        std = torch.exp(0.5 * log_var)
                        std_clipped = torch.clip(std, min=1e-6) # to avoid numerical instability
                        normal_dist = Normal(mean, std_clipped)
                        log_prob_original = normal_dist.log_prob(actions)
                        log_prob_original = log_prob_original[:, 1:].mean(dim=list(range(1, log_prob_new.ndim)))
                        kl_loss = 0.5 * torch.mean((log_prob_new - log_prob_original) ** 2)
                        self.kl_beta = 0.01
                        loss += self.kl_beta * kl_loss
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    # clip grad norm
                    max_norm = 1.0
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm)
                    
                    self.optimizer.step()

                    # Update running loss
                    pbar.update(1)
                    running_loss += loss.item()
                    if pbar.n % self.log_at_steps == 0:
                        pbar.set_postfix({"loss": running_loss / pbar.n})
        self.replay_buffer.clear()
        
    def visualize_evaluate_episode(self, save_path, batch_data):
        x_shape, diffusion_context_cond, state_cond, raw_env_info = self.process_batch_func(batch_data)
        
        N = x_shape[0] # rollout for N trajectories
        
        with torch.no_grad():
            final_xt, chain, _ = self.actor.p_sample_loop(x_shape, diffusion_context_cond=diffusion_context_cond, 
                                         state_cond=state_cond, skip_x0=False)

        trajectories = chain[-1]
        #### EVALUATE ####
        trajs = []
        obstacle_lists = []
        avoid_region_lists = []
        walk_over_lists = []
        goals = []
        for idx in range(trajectories.shape[0]):
            trajectory = trajectories[idx:idx+1]
            obstacle_info, avoid_region_info, walk_over_info, start, goal, goal_radius = raw_env_info[idx]
            obstacle_list = [construct_obstacle_from_info(info) for info in obstacle_info]
            avoid_region_lists = [construct_obstacle_from_info(info) for info in avoid_region_info]
            walk_over_list = [construct_obstacle_from_info(info) for info in walk_over_info]
            traj = self.reward_model.decode_trajectory(trajectory, goal, goal_radius)
            trajs.append(traj[0])
            obstacle_lists.append(obstacle_list)
            walk_over_lists.append(walk_over_list)
            goals.append(goal)

        #### VISUALIZE ####
        save_path = save_path / "ddpo_episode.gif"

        visualization_list = obstacle_list + avoid_region_lists + walk_over_list
        plot_path_gif(traj, visualization_list, goal, goal_radius, 
                                     self.grid_size, 
                                     self.env_dt, save_path)
        return save_path
    
