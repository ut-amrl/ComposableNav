"""
A simple implementation of PPO 
"""
import torch 
import numpy as np
from torch.distributions import Normal
from einops import rearrange
from copy import deepcopy
from tqdm import tqdm 
from functools import partial
from omegaconf import DictConfig, OmegaConf

from composablenav.misc import critic
from composablenav.datasets.scenario_generator import gen_noncolliding_obstacles, gen_regions, gen_goal
from composablenav.misc.normalizers import PathNormalizer
from composablenav.misc.process_data import construct_normalized_obstacle_seq, construct_normalized_static_obstacle_from_obj
from composablenav.misc.visualizer_utils import visualize_paths_with_rewards
from composablenav.misc.common import construct_obstacle_from_info, find_first_waypoint_within_radius
from composablenav.models.diffusion_components import diffusion_sample_fn_log_prob
from composablenav.train.dataloader_base import ProcessObsHelper
from composablenav.datasets.obstacles import Rectangle
from collections import deque

class SuccessEvaluator:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.success_criteria = self.cfg.inference.success_criteria
        dt = self.cfg.dataset_generation.env.env_dt

        self.criteria_fns = {}
        for criteria_name, criteria_args in self.success_criteria.items():
            criteria = getattr(critic, criteria_name)
            for name, args in criteria_args.items():
                self.criteria_fns[criteria_name + "_" + name] = partial(criteria, dt=dt, **args)

    def __call__(self, trajectories, obstacle_lists, region_lists, goals):
        results = {}
        total_batch = len(trajectories)
        for traj, obstacles, regions, goal in zip(trajectories, obstacle_lists, region_lists, goals):
            for criteria_name, criteria_fn in self.criteria_fns.items():
                obs_terr_list = obstacles + regions
                sat = criteria_fn(path=traj, obstacles=obs_terr_list, goal=goal)
                if results.get(criteria_name) is None:
                    results[criteria_name] = 0
                results[criteria_name] += int(sat) / total_batch * 100
        self.pretty_print(results)
        return results

    def pretty_print(self, results):
        print("====================================")
        print("Evaluation Results")
        for key, value in results.items():
            print(f"{key}: {value:.2f}%")
        print("====================================")

class RewardModel:
    def __init__(self, cfg: DictConfig, scenario):
        self.cfg = cfg
        self.path_normalizer = PathNormalizer(self.cfg.env.grid_size)
        criteria_func = getattr(critic, scenario.criteria.criteria_fn)
        print(f"Training for Scenario: {scenario.scenario_name}")
        if "criteria_params" not in scenario.criteria:
            self.criteria = criteria_func
        else:
            self.criteria = partial(criteria_func, **scenario.criteria.criteria_params)    
        
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
        obstacle_info, avoid_region_info, prefer_region_info, start, goal, goal_radius = env_info
        traj = self.decode_trajectory(trajectory, goal, goal_radius)
        vis_rewards = np.zeros(2)
        obstacle_list = [construct_obstacle_from_info(info) for info in obstacle_info]
        avoid_region_list = [construct_obstacle_from_info(info) for info in avoid_region_info]
        prefer_region_list = [construct_obstacle_from_info(info) for info in prefer_region_info]
        
        obs_terr_list = obstacle_list + avoid_region_list + prefer_region_list
        # visualize_paths_with_rewards(traj, vis_rewards, obs_terr_list, goal_pos=goal, goal_radius=goal_radius, 
        #                              grid_size=self.cfg.env.grid_size, dt=self.cfg.env.env_dt, save_name="test_ppo")
        #### Compute reward ###
        reward = self.criteria(traj[0], obstacles=obs_terr_list, goal=goal, dt=self.cfg.env.env_dt)
        return reward

    def compute_success_criteria(self, trajectory, env_info):
        obstacle_info, avoid_region_info, prefer_region_info, start, goal, goal_radius = env_info
        traj = self.decode_trajectory(trajectory, goal, goal_radius)
        obstacle_list = [construct_obstacle_from_info(info) for info in obstacle_info]
        return self.criteria(traj[0], obstacles=obstacle_list, goal=goal, dt=self.cfg.env.env_dt)

class Maze2DEnv:
    def __init__(self, cfg, max_obj_traj_len, max_padded_obj_num, max_padded_terrain_num):
        self.cfg = cfg
        self.path_normalizer = PathNormalizer(self.cfg.env.grid_size)
        self.process_obs = ProcessObsHelper(max_obj_traj_len=max_obj_traj_len, 
                         max_padded_obj_num=max_padded_obj_num,
                         max_padded_terrain_num=max_padded_terrain_num)
    
    def generate_batch(self, batch_size):
        raw_env_info = []
        obs = []
        obs_mask = []
        avoid_region = []
        avoid_region_mask = []
        prefer_region = []
        prefer_region_mask = []
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
            prefer_region_tensor, prefer_region_mask_tensor, start_tensor, goal_tensor = self.build_context_tensor(env_info)
            raw_env_info.append(env_info)
            obs.append([o for o in obs_tensor])
            obs_mask.append(obs_mask_tensor)
            
            avoid_region.append([t for t in avoid_region_tensor])
            avoid_region_mask.append(avoid_region_mask_tensor)
            
            prefer_region.append([t for t in prefer_region_tensor])
            prefer_region_mask.append(prefer_region_mask_tensor)
            
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
                    "mask": torch.stack(prefer_region_mask).float(),
                    "input_vals": [torch.stack(t) for t in zip(*prefer_region)]
                }
            },
            "start": torch.stack(start).float(),
            "raw_env_info": raw_env_info
        }
        return batch
    
    def create_obstacle_info(self, obstacle_list):
        start_loc = self.cfg.objective.start_loc
                
        return [obstacle.to_dict() for obstacle in obstacle_list], start_loc
    
    def generate_one(self):
        obstacle_list = gen_noncolliding_obstacles(self.cfg)
        if self.cfg.scenarios.scenario == "static":
            avoid_region_list = gen_regions(self.cfg)
            prefer_region_list = []
        elif self.cfg.scenarios.scenario == "prefer":
            prefer_region_list = gen_regions(self.cfg)
            avoid_region_list = []
        else:
            prefer_region_list = []
            avoid_region_list = []
        goal_loc = gen_goal(self.cfg, obstacle_list, prefer_region_list, scenario=self.cfg.scenarios.scenario)
        goal_radius = self.cfg.objective.goal_radius
        
        if goal_loc[0] is None or obstacle_list is None or prefer_region_list is None or avoid_region_list is None:
            return None
        
        obstacles_info, start_loc = self.create_obstacle_info(obstacle_list)
        avoid_region_info = [avoid_region.to_dict() for avoid_region in avoid_region_list]
        prefer_region_info = [prefer_region.to_dict() for prefer_region in prefer_region_list]

        return [obstacles_info, avoid_region_info, prefer_region_info, start_loc, goal_loc, goal_radius] # not handling static for now
    
    def build_context_tensor(self, env_info):
        obstacles_info, avoid_region_info, prefer_region_info, start_loc, goal_loc, goal_radius = env_info
        obs_tensor, obs_mask_tensor = self.build_normalized_obstacle_seq(obstacles_info)
        # ######## temp: for fixed start position: to be removed later
        # obs_tensor = torch.zeros([10, 5], device="cuda")
        # info = obstacles_info[0]
        # obs_tensor[0] = torch.tensor([float(info['x'])/10, 
        #                               float(info['y'])/10, 
        #                               0, 
        #                               1, 
        #                               float(info['speed'])-1])
        # ######## temp: for fixed start position: to be removed later
        prefer_region_tensor, prefer_region_mask_tensor = self.build_normalized_prefer_region(prefer_region_info)
        avoid_region_tensor, avoid_region_mask_tensor = self.build_normalized_avoid_region(avoid_region_info)
        start_tensor = self.build_loc_normalized(start_loc)
        goal_tensor = self.build_loc_normalized(goal_loc)
        return obs_tensor, obs_mask_tensor, avoid_region_tensor, avoid_region_mask_tensor, \
               prefer_region_tensor, prefer_region_mask_tensor, start_tensor, goal_tensor
        
    def build_normalized_obstacle_seq(self, obstacles_info):
        normalized_obstacle_seq = construct_normalized_obstacle_seq(obstacles_info, 
                                                                    grid_size=self.cfg.env.grid_size, 
                                                                    max_planning_time=self.cfg.robot.max_planning_time, 
                                                                    dt=self.cfg.env.env_dt,
                                                                    offset_x=0, offset_y=0, mult_x=1, mult_y=1
                                                                    )
        obs, obs_mask = self.process_obs.get_obs_cond(normalized_obstacle_seq)
        return obs, obs_mask

    def build_normalized_prefer_region(self, region_info):
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
        normalized_statics = construct_normalized_static_obstacle_from_obj(static_list, grid_size=self.cfg.env.grid_size, 
                                                                            mult_x=1, mult_y=1, offset_x=0, offset_y=0)
        statics, static_mask = self.process_obs.get_static_cond(normalized_statics)
        return statics, static_mask
    
    def build_loc_normalized(self, loc):
        loc_tensor = torch.tensor(loc)
        return self.path_normalizer.normalize(loc_tensor)

class ReplayBuffer:
    def __init__(self, batch_size, device):
        self.batch_size = batch_size
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
        batch_indices = batch_indices[: (NT // self.batch_size) * self.batch_size] # truncate
        batch_indices = batch_indices.reshape(-1, self.batch_size)
        
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
        return len(self.states) // self.batch_size
        
    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.advantages = []
        self.log_probs = []

class DDPOAgent:
    def __init__(self, cfg, actor, reward_model, replay_buffer, evaluator_model, log_at_steps,
                 epochs, epislon_clip, lr, use_kl=False, kl_beta=0.01):
        self.cfg = cfg
        self.actor = actor
        self.original_actor = deepcopy(actor)
        self.actor_old = deepcopy(actor)
        self.reward_model = reward_model
        self.replay_buffer = replay_buffer
        self.evaluator_model = evaluator_model
        
        self.log_at_steps = log_at_steps
        self.epochs = epochs
        self.epislon_clip = epislon_clip
        self.optimizer = torch.optim.AdamW(self.actor.parameters(), lr=lr)
        self.use_kl = use_kl
        self.kl_beta = kl_beta
        self.min_count = 16
        self.advantage_deque = deque(maxlen=64)
        if use_kl:
            print("Using KL divergence loss")

    def helper_retrieve_state_dict(self, idx, context):
        # TODO: hardcode for now
        new_context = {}
        for k, v in context.items():
            new_context[k] = v[idx]
        return new_context  
    
    def helper_retrieve_context_dict(self, idx, context):
        context = deepcopy(context)
        # 11/07/2024: Hardcoded for obs encoder for now
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
        # 11/07/2024: Hardcoded for obs encoder for now
        return new_context  
        
    def get_batched_rollout(self, x_shape, diffusion_context_cond, state_cond, raw_env_info):
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
        # TODO today 1/21/2025
        self.advantage_deque.extend(self.replay_buffer.rewards)
        if len(self.advantage_deque) < self.min_count:
            mean = torch.mean(rewards)
            std = torch.std(rewards) + 1e-6
        else:

            mean = torch.tensor(np.mean(self.advantage_deque)).float()
            std = torch.tensor(np.std(self.advantage_deque)).float() + 1e-6
        rewards = torch.tensor(self.replay_buffer.rewards).float()
        advantages = (rewards - mean) / std
        self.replay_buffer.set_advantages(advantages)
        
    def compute_advantage_legacy(self):
        """
        just normalize rewards for now
        """
        rewards = torch.tensor(self.replay_buffer.rewards).float()
        advantages = (rewards - torch.mean(rewards)) / (torch.std(rewards) + 1e-7)
        self.replay_buffer.set_advantages(advantages)

    def update_policy_fix_state(self):
        # not good
        total_iterations = self.epochs * len(self.replay_buffer)
        running_loss = 0.0
        with tqdm(total=total_iterations, desc="Processing", unit="iteration") as pbar:
            for epoch in range(self.epochs):
                samples = self.replay_buffer.sample()
                for sample in samples:
                    batch = self.replay_buffer.collate(sample)
                    
                    states = batch["states"]
                    actions = batch["actions"]
                    log_probs_old = batch["log_probs"]
                    advantages = batch["advantages"]
                    final_states = batch["final_states"]
                    
                    xt, t, diffusion_context_cond, state_cond = states
                    
                    # fix state
                    state_fix_idx = torch.randint(1, 20, (1,)).item()
                    xt[:, :state_fix_idx] = final_states[:, :state_fix_idx]
                    
                    # compute prob log
                    with torch.no_grad():
                        mean, _, log_var = self.actor_old.p_mean_variance(xt, t, diffusion_context_cond)
                        std = torch.exp(0.5 * log_var)
                        std_clipped = torch.clip(std, min=1e-6) # to avoid numerical instability
                        normal_dist = Normal(mean, std_clipped)
                        log_probs_old = normal_dist.log_prob(actions)
                        log_probs_old = log_probs_old[:, state_fix_idx:].mean(dim=list(range(1, log_probs_old.ndim)))

                    # refactor later to be sameas sample_fn
                    mean, _, log_var = self.actor.p_mean_variance(xt, t, diffusion_context_cond)
                    std = torch.exp(0.5 * log_var)
                    std_clipped = torch.clip(std, min=1e-6) # to avoid numerical instability
                    normal_dist = Normal(mean, std_clipped)
                    log_prob_new = normal_dist.log_prob(actions)
                    log_prob_new = log_prob_new[:, state_fix_idx:].mean(dim=list(range(1, log_prob_new.ndim)))
                    
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
        
    def update_policy(self):  
        total_iterations = self.epochs * len(self.replay_buffer)
        running_loss = 0.0
        with tqdm(total=total_iterations, desc="Processing", unit="iteration") as pbar:
            for epoch in range(self.epochs):
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
        
    def visualize_evaluate_episode(self, x_shape, diffusion_context_cond, state_cond, raw_env_info):
        N = x_shape[0] # rollout for N trajectories
        
        with torch.no_grad():
            final_xt, chain, _ = self.actor.p_sample_loop(x_shape, diffusion_context_cond=diffusion_context_cond, 
                                         state_cond=state_cond, skip_x0=False)

        trajectories = chain[-1]
        #### EVALUATE ####
        trajs = []
        obstacle_lists = []
        avoid_region_lists = []
        prefer_region_lists = []
        goals = []
        for idx in range(trajectories.shape[0]):
            trajectory = trajectories[idx:idx+1]
            obstacle_info, avoid_region_info, prefer_region_info, start, goal, goal_radius = raw_env_info[idx]
            obstacle_list = [construct_obstacle_from_info(info) for info in obstacle_info]
            avoid_region_lists = [construct_obstacle_from_info(info) for info in avoid_region_info]
            prefer_region_list = [construct_obstacle_from_info(info) for info in prefer_region_info]
            traj = self.reward_model.decode_trajectory(trajectory, goal, goal_radius)
            trajs.append(traj[0])
            obstacle_lists.append(obstacle_list)
            prefer_region_lists.append(prefer_region_list)
            goals.append(goal)

        evaluated_results = self.evaluator_model(trajs, obstacle_lists, prefer_region_lists, goals)

        #### VISUALIZE ####
        rewards = np.zeros(2)
        save_name = "ddpo_espisode"

        # # temporary add follow region
        # circle = obstacle_list[0]
        # rec = Rectangle(circle.x-5.75, circle.y, circle.theta, circle.speed, width=3.5, height=4.5) # hardcoded
        # obstacle_list.append(rec)
        # # temportary
        visualization_list = obstacle_list + avoid_region_lists + prefer_region_list
        visualize_paths_with_rewards(traj, rewards, visualization_list, goal, goal_radius, 
                                     self.cfg.dataset_generation.env.grid_size, 
                                     self.cfg.dataset_generation.env.env_dt, save_name) # fix this
        return f"{save_name}.gif", evaluated_results
    
