import json
import yaml
import os 
import torch
from copy import deepcopy
from glob import glob
import numpy as np
from typing import Union
from einops import repeat
from composablenav.datasets import obstacles 

def get_path_from_motion(opt_u, start, dt):
    opt_u_np = np.array(opt_u)

    vw_path = []
    current_x = start[0]
    current_y = start[1]
    current_theta = start[2]
    for k in range(opt_u_np.shape[1]):
        v, w = opt_u_np[:, k]
        vw_path.append((current_x, current_y, current_theta, v, w))
        current_x, current_y, current_theta = forward_motion_rollout(v, w, current_x, current_y, current_theta, dt)
        
    last_v, last_w = opt_u_np[:, -1]
    vw_path.append((current_x, current_y, current_theta, last_v, last_w))    
        
    return vw_path

def save_json_data(json_fn, data):
    with open(json_fn, 'w') as json_file:
        json.dump(data, json_file, indent=4) 
        
def load_data(file_name):
    if file_name.endswith('.json'):
        with open(file_name, 'r') as file:
            data = json.load(file)
    elif file_name.endswith(('.yaml', '.yml')):
        with open(file_name, 'r') as file:
            data = yaml.safe_load(file)
    elif file_name.endswith('.txt'):
        with open(file_name, 'r') as file:
            data = file.read()
    else:
        raise ValueError("Unsupported file format. Please provide a .json or .yaml/.yml file.")
    return data

def load_fns_from_folder(folder, format="json", sort=True):
    if format == "json":
        path_name = os.path.join(folder, "*.json")
    elif format == "directory":
        fns = [os.path.join(folder, d) for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
        fns = sorted(fns)
        return fns
    else:
        raise NotImplementedError("Only support json format for now")
    fns = glob(path_name)
    if len(fns) == 0:
        print(f"No file found in {folder}")
    if sort:
        fns = sorted(fns, key=lambda x: int(x.split("/")[-1].split(".")[0]))
    return fns

def load_all_fns_from_dir(dir_path, format="json"):
    # new helper function to load all json files from a directory and its subdirectories
    json_files = []
    
    # Walk through the directory and subdirectories
    for root, _, files in os.walk(dir_path):
        for file in files:
            # Check if the file is a json file
            if file.endswith((f'.{format}')):
                file_path = os.path.join(root, file)
                json_files.append(file_path)
    
    return json_files

def info_obs_goal_from_fn(fn):
    data = load_data(fn)
    info = data["info"]
    obstacles_list = [construct_obstacle_from_info(info) for info in data["obs"]["obstacles_info"]]
    goal = np.array(data["trajectories"]["0"]["goal"]) * (info["grid_size"] // 2) # hardcoded
    return info, obstacles_list, goal

def construct_obs_from_yaml(fn):
    data = load_data(fn)
    obstacles_list = [construct_obstacle_from_info(info) for info in data["obstacles_info"]] if data.get("obstacles_info") else []
    terrain_list = [construct_obstacle_from_info(info) for info in data["terrain_info"]] if data.get("terrain_info") else []
    goal_pos = data["objective"]["goal"]
    goal_radius = data["objective"]["goal_radius"]
    grid_size = data["env"]["grid_size"]
    return obstacles_list, terrain_list, grid_size, goal_pos, goal_radius

def construct_obstacle_from_info(obstacle_info: dict):
    obstacle_instance = getattr(obstacles, obstacle_info["type"])(**obstacle_info)
    return obstacle_instance

def get_truncate_indices_from_mask(masks: torch.tensor):
    assert type(masks) == torch.Tensor and len(masks.shape) == 3
    sub_tensor = masks[:, 1:, 0]  # Shape is (B, C-1, H)
    indices = (sub_tensor == 0).float().argmax(dim=1)
    indices += 1
    return indices.cpu().numpy()

def find_first_waypoint_within_radius(path, goal, radius):
    path = np.array(path)
    goal = np.array(goal)
    assert len(path.shape) == 2
    for idx, waypoint in enumerate(path):
        distance = np.linalg.norm(waypoint[:goal.shape[0]] - goal)
        if distance <= radius:
            return idx
    return -1  # Return -1 if no waypoint is within the radius

def validate_plan(path: Union[list, np.ndarray], obstacles, dt, start_time_idx=0):
    obstacles = deepcopy(obstacles) # avoid modifying the original obstacles
    start_time = start_time_idx * dt
    for t_idx in range(len(path)-1):
        curr_node = path[t_idx]
        next_node = path[t_idx+1]
        curr_x, curr_y = curr_node[0], curr_node[1]
        next_x, next_y = next_node[0], next_node[1]
        
        t = t_idx * dt
        t_plus1 = (t_idx + 1) * dt
        for obstacle in obstacles:
            if obstacle.collision(curr_x, curr_y, next_x, next_y, t + start_time, 0) or \
               obstacle.collision(curr_x, curr_y, next_x, next_y, t_plus1 + start_time, 0):
                return False
    return True

def forward_motion_rollout(v, w, x, y, theta, planning_dt):
    # constant curvature model
    if w != 0:        
        theta_new = theta + w * planning_dt
        x_new = x + v / w * (np.sin(theta_new) - np.sin(theta))
        y_new = y - v / w * (np.cos(theta_new) - np.cos(theta))
    else:
        x_new = x + v * np.cos(theta) * planning_dt
        y_new = y + v * np.sin(theta) * planning_dt
        theta_new = theta
    return x_new, y_new, theta_new

def forward_motion_rollout_noisy(v, w, x, y, theta, noise_const, planning_dt):
    # constant curvature model
    v = v + np.random.randn() * noise_const[0]
    w = w + np.random.randn() * noise_const[1]
    if w != 0:        
        theta_new = theta + w * planning_dt
        x_new = x + v / w * (np.sin(theta_new) - np.sin(theta))
        y_new = y - v / w * (np.cos(theta_new) - np.cos(theta))
    else:
        x_new = x + v * np.cos(theta) * planning_dt
        y_new = y + v * np.sin(theta) * planning_dt
        theta_new = theta
    return x_new, y_new, theta_new

def forward_motion_rollout_simplified(v, w, x, y, theta, planning_dt):
    # constant curvature model with simplified calculation. Assume w * planning_dt is close to zero
    theta_new = theta + w * planning_dt
    theta_estimate = theta + w * planning_dt / 2
    x_new = x + v * np.cos(theta_estimate) * planning_dt
    y_new = y + v * np.sin(theta_estimate) * planning_dt
    return x_new, y_new, theta_new

def forward_motion_rollout_tensor(v: torch.Tensor, w: torch.Tensor, x: torch.Tensor, y: torch.Tensor, theta: torch.Tensor, 
                                  planning_dt: torch.Tensor, eps: float = 1e-1):
    # constant curvature model
    # is_curved = w.abs() > eps  # Element-wise check if w is not close to zero; eps cannot be set too small to avoid numerical instability
    
    # For curved motion
    # theta_new = torch.where(is_curved, theta + w * planning_dt, theta)
    # theta_new = (theta_new + torch.pi) % (2 * torch.pi) - torch.pi
    # x_new = torch.where(is_curved, x + v / w * (theta_new.sin() - theta.sin()), x + v * theta.cos() * planning_dt)
    # y_new = torch.where(is_curved, y - v / w * (theta_new.cos() - theta.cos()), y + v * theta.sin() * planning_dt)
    theta_new = theta + w * planning_dt
    theta_new = (theta_new + torch.pi) % (2 * torch.pi) - torch.pi
    theta_estimate = theta + w * planning_dt / 2
    x_new = x + v * theta_estimate.cos() * planning_dt
    y_new = y + v * theta_estimate.sin() * planning_dt
    return x_new, y_new, theta_new

def pos_in_bounds(x, y, grid_size):
    return x >= -grid_size//2 and x <= grid_size//2 and y >= -grid_size//2 and y <= grid_size//2

def put_batch_on_device(batch, device):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
        elif isinstance(value, list) and isinstance(value[0], torch.Tensor):
            batch[key] = [v.to(device) for v in value]
        elif isinstance(value, dict):  # If the value is a dictionary, recursively call the function
            batch[key] = put_batch_on_device(value, device)
    return batch

def repeat_data(data, repeat_size):
    if isinstance(data, torch.Tensor):
        return repeat(data, "b ... -> (b repeat_size) ...", repeat_size=repeat_size)
    elif isinstance(data, dict):
        for k, v in data.items():
            if len(v) == 0:
                continue
            data[k] = repeat(v, "b ... -> (b repeat_size) ...", repeat_size=repeat_size)
        return data
    else:
        raise ValueError("Invalid data type")

def repeat_context(context, num_repeats):
    for key in context:
        if len(context[key]) == 0:
            continue
        if isinstance(context[key]["input_vals"], list):
            context[key]["mask"] = repeat_data(context[key]["mask"], num_repeats)
            context[key]["input_vals"] = [repeat_data(val, num_repeats) for val in context[key]["input_vals"]]
        elif isinstance(context[key]["input_vals"], torch.Tensor):
            context[key]["mask"] = repeat_data(context[key]["mask"], num_repeats)
            context[key]["input_vals"] = repeat_data(context[key]["input_vals"], num_repeats)
    return context