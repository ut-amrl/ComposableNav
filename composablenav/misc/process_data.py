import os 
import numpy as np 
from omegaconf import OmegaConf

from composablenav.misc.normalizers import PathNormalizer, DifferentialPathNormalizer, TerrainNormalizer
from composablenav.misc.common import construct_obstacle_from_info, pos_in_bounds

def shift_scale_x(x, grid_size, mult_x, offset_x):
    x += offset_x
    return (x + grid_size / 2.0) * mult_x - (grid_size / 2.0)

def shift_scale_y(y, mult_y, offset_y):
    y += offset_y
    return y * mult_y

def shift_descale_x(x, grid_size, mult_x, offset_x):
    x = (x + grid_size / 2.0) / mult_x - (grid_size / 2.0)
    return x - offset_x

def shift_descale_y(y, mult_y, offset_y):
    y = y / mult_y
    return y - offset_y

def construct_normalized_dynamic_obstacle_from_obj(obstacles_list,
                                      grid_size, max_planning_time, dt, 
                                      mult_x=1, mult_y=1, offset_x=-10, offset_y=0) -> list:
    result = []
    normalizer = PathNormalizer(grid_size)
    for i in range(len(obstacles_list)):
        obstacle = obstacles_list[i]
        radius = obstacle.radius
        tmp = []
        for t_idx in range(max_planning_time):
            t = t_idx * dt
            x, y, _ = obstacle.pos_at_dt(t)
            # scale up the x and y
            x = shift_scale_x(x, grid_size, mult_x, offset_x)
            y = shift_scale_y(y, mult_y, offset_y)
            x_norm, y_norm = normalizer.normalize(np.array([x, y])).tolist()
            if pos_in_bounds(x, y, grid_size + radius): # check if the obstacle + its radius is within the grid
                tmp.extend([x_norm, y_norm, 1]) # the third value is hardcoded, later can be changed to radius
            else:
                tmp.extend([0, 0, 0])
        result.append(tmp)
    
    return result

def construct_normalized_obstacle_seq(obstacle_info,
                                      grid_size, max_planning_time, dt, 
                                      offset_x, offset_y, mult_x=1, mult_y=1, ) -> list:
    obstacles_list = [construct_obstacle_from_info(info) for info in obstacle_info]
    return construct_normalized_dynamic_obstacle_from_obj(obstacles_list, grid_size=grid_size, max_planning_time=max_planning_time, dt=dt, 
                                                          mult_x=mult_x, mult_y=mult_y, offset_x=offset_x, offset_y=offset_y)

def construct_normalized_static_obstacle_from_obj(static_obs, grid_size, mult_x=1, mult_y=1, offset_x=-10, offset_y=0):
    result = []
    for obs in static_obs:
        normalizing_constant = grid_size / 2
        
        normalized_left = shift_scale_y(obs.left, mult_y, offset_y) / normalizing_constant
        normalized_right = shift_scale_y(obs.right, mult_y, offset_y) / normalizing_constant
        normalized_top = shift_scale_x(obs.top, grid_size, mult_x, offset_x) / normalizing_constant
        normalized_bottom = shift_scale_x(obs.bottom, grid_size, mult_x, offset_x) / normalizing_constant
        
        normalized_left = 1 if normalized_left > 1 else normalized_left
        normalized_right = -1 if normalized_right < -1 else normalized_right
        normalized_top = 1 if normalized_top > 1 else normalized_top
        normalized_bottom = -1 if normalized_bottom < -1 else normalized_bottom

        top_left = [normalized_top, normalized_left]
        top_right = [normalized_top, normalized_right]
        bottom_right = [normalized_bottom, normalized_right]
        bottom_left = [normalized_bottom, normalized_left]
        result.append([top_right, top_left, bottom_left, bottom_right])
    return result

def construct_normalized_prefer_region(prefer_obs, grid_size, mult_x=1, mult_y=1, offset_x=-10, offset_y=0):
    result = []
    for obs in prefer_obs:
        normalizing_constant = grid_size / 2
        
        normalized_left = shift_scale_y(obs.left, mult_y, offset_y) / normalizing_constant
        normalized_right = shift_scale_y(obs.right, mult_y, offset_y) / normalizing_constant
        normalized_top = shift_scale_x(obs.top, grid_size, mult_x, offset_x) / normalizing_constant
        normalized_bottom = shift_scale_x(obs.bottom, grid_size, mult_x, offset_x) / normalizing_constant
        
        normalized_left = 1 if normalized_left > 1 else normalized_left
        normalized_right = -1 if normalized_right < -1 else normalized_right
        normalized_top = 1 if normalized_top > 1 else normalized_top
        normalized_bottom = -1 if normalized_bottom < -1 else normalized_bottom

        result.append([normalized_top, normalized_bottom, normalized_left, normalized_right, 0])
    return result

def normalize_static(static_obs, grid_size, offset_x, offset_y, mult_x=1, mult_y=1):
    center_x = shift_scale_x(static_obs["x"], grid_size, mult_x, offset_x)
    center_y = shift_scale_y(static_obs["y"], mult_y, offset_y)
    width = static_obs["width"] * mult_y
    height = static_obs["height"] * mult_x
    normalizing_constant = grid_size / 2
    normalized_right = (center_y - width / 2) / normalizing_constant
    normalized_left = (center_y + width / 2) / normalizing_constant
    normalized_top = (center_x + height / 2) / normalizing_constant
    normalized_bottom = (center_x - height / 2) / normalizing_constant
    if normalized_right < -1:
        normalized_right = -1
    if normalized_left > 1:
        normalized_left = 1
    if normalized_top > 1:
        normalized_top = 1
    if normalized_bottom < -1:
        normalized_bottom = -1
    
    top_right = [normalized_top, normalized_right]
    top_left = [normalized_top, normalized_left]
    bottom_right = [normalized_bottom, normalized_right]
    bottom_left = [normalized_bottom, normalized_left]

    return [top_right, top_left, bottom_left, bottom_right]

def process_planning_info(planning_info, differnetial_normalizer: DifferentialPathNormalizer):
        start = np.array(planning_info["start"])
        goal = np.array(planning_info["goal"])
        path = np.array(planning_info["path"])
        path_info = planning_info["info"]

        start = differnetial_normalizer.path_normalizer.normalize(start).tolist()
        goal = differnetial_normalizer.path_normalizer.normalize(goal).tolist()
        path = differnetial_normalizer.normalize(path).tolist()

        result = {
            "start": start,
            "goal": goal,
            "path": path,
            "path_info": path_info
        }
        return result
    
def convert_to_saving_format_rrt_hybrid_astar(outputs, folder_name, training_count, cfg):
    # hybrid astar needs to take in max angular and velocity etc
    path_normalizer = PathNormalizer(cfg.env.grid_size)
    differnetial_normalizer = DifferentialPathNormalizer(cfg.env.grid_size, np.pi, cfg.robot.max_v, cfg.robot.max_w)
    terrain_normalizer = TerrainNormalizer(grid_size=cfg.env.grid_size)

    env_info = OmegaConf.to_container(cfg.env, resolve=True)
    robot_info = OmegaConf.to_container(cfg.robot, resolve=True)
    objective_info = OmegaConf.to_container(cfg.objective, resolve=True)

    obstacles_info =  outputs["obstacles_info"]
    terrain_info = outputs["terrain_info"]
    normalized_obstacle_seq = construct_normalized_obstacle_seq(obstacles_info, 
                                    grid_size=cfg.env.grid_size, 
                                    max_planning_time=cfg.robot.max_planning_time, 
                                    dt=cfg.env.env_dt, offset_x=0, offset_y=0)

    normalized_terrain = [terrain_normalizer.normalize(terrain).tolist() for terrain in terrain_info]
    # temporary convert static from center rep to corner rep
    normalized_static = [normalize_static(terrain, cfg.env.grid_size, offset_x=0, offset_y=0) for terrain in terrain_info]
    outputs_dict = {
        "info": {
            **env_info,
            **robot_info,
            **objective_info
        },
        "obs": {
            "obstacles_info": obstacles_info,
            "terrain_info": terrain_info,
            "normalized_obstacle_seq": normalized_obstacle_seq,
            "normalized_terrain": normalized_terrain,
            "normalized_static": normalized_static
        },
        "trajectories": {}
    }
    paths = outputs["paths"]
    for idx, path_data in enumerate(paths):            
        planning_data = process_planning_info(path_data, differnetial_normalizer)
        outputs_dict["trajectories"][str(idx)] = planning_data

    json_fn = os.path.join(cfg.generation.output_dir, folder_name, f"{training_count:07}.json")
    return json_fn, outputs_dict