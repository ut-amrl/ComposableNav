from abc import ABC, abstractmethod 
from torch.utils.data import Dataset 
import torch  
import numpy as np 
from typing import List
from einops import rearrange
from composablenav.misc.common import load_data, construct_obstacle_from_info
from composablenav.misc.process_data import construct_normalized_obstacle_seq, normalize_static
from composablenav.misc.normalizers import PathNormalizer, TerrainNormalizer

class BaseDataset(Dataset, ABC):
    def __init__(self, input_data, fn_or_data: str, max_padded_len: int):
        # load from list of files or just a list json data
        # Dataset().__init__(self)
        self.cum_sum_indexes = []
        self.indexes = []
        
        if fn_or_data == "fn":
            # load from list of files
            self.fns = input_data
            for fn in self.fns:
                data = load_data(fn)
                index = self.get_index(data=data)
                self.indexes.append(index)
        elif fn_or_data == "json_data":
            # load from list of json data
            self.fns = []
            self.json_data = []
            for json_data in input_data:
                data = json_data["data"]
                index = self.get_index(data=data)
                
                self.indexes.append(index)
                self.fns.append(json_data["fn"])
                self.json_data.append(data)
        elif fn_or_data == "meta_file":
            data = load_data(input_data)
            self.fns = data["fns"]
            self.indexes = data["indexes"]
        else:
            raise ValueError("Invalid input data")
            
        flattened_list = list(map(len, self.indexes))
        self.cum_sum_indexes = np.cumsum(flattened_list)
        self.max_padded_len = max_padded_len
        self.fn_or_data = fn_or_data
        
        print(f"Dataset Input: {fn_or_data} with size: {self.cum_sum_indexes[-1]} and max padded length: {self.max_padded_len}")
    
    @abstractmethod
    def get_index(self, data) -> List:
        ...
    
    @abstractmethod
    def parse_traj_representation(self, **kwargs):
        ...

    def __len__(self):
        return self.cum_sum_indexes[-1]
    
    def get_data_from_index(self, index):
        fn_index = int(np.searchsorted(self.cum_sum_indexes, index, side='right'))
        fn = self.fns[fn_index]
        
        if self.fn_or_data == "fn" or self.fn_or_data == "meta_file":
            data = load_data(fn)
        elif self.fn_or_data == "json_data":
            data = self.json_data[fn_index]
        else:
            raise ValueError("Invalid input data")

        traj_index = index - self.cum_sum_indexes[fn_index - 1] if fn_index > 0 else index
        traj_index = self.indexes[fn_index][traj_index]
        return data, traj_index, fn
    
class ProcessObsHelper:
    def __init__(self, max_obj_traj_len, max_padded_obj_num, max_padded_terrain_num) -> None:
        # Separate this out for easier reuse
        self.max_obj_traj_len = max_obj_traj_len
        self.max_padded_obj_num = max_padded_obj_num
        self.max_padded_terrain_num = max_padded_terrain_num

    def create_padded_path_n_mask(self, path, pad, padded_len):
        mask = torch.zeros([padded_len, *path.shape[1:]])
        # 10/09/2024 added: path.shape[0] +1 to include always reaching the goal
        mask[1:path.shape[0] + 1] = 1 # mask out first element and pad the rest # not being used
        
        if path.shape[0] < padded_len:
            rows_to_add = padded_len - path.shape[0]
            padding_tensor = pad.view(1, -1).repeat(rows_to_add, 1)
            padded_path = torch.cat((path, padding_tensor), dim=0)
        else:
            padded_path = path[:padded_len]
        return padded_path, mask
    
    def get_obs_cond(self, normalized_obstacle_seq):
        # 11/25: always pad to max padded obj num
        pad = torch.zeros(self.max_obj_traj_len)
        obs_mask = torch.zeros(self.max_padded_obj_num)
        
        if len(normalized_obstacle_seq) == 0:
            obs = torch.zeros(1, self.max_obj_traj_len)
            obs, _ = self.create_padded_path_n_mask(obs, pad, self.max_padded_obj_num)
            return obs, obs_mask
        
        obs = torch.tensor(normalized_obstacle_seq)
        obs = obs[:, :self.max_obj_traj_len] # max obj traj len is traj_len * 3
        obs_mask[:obs.shape[0]] = 1
        assert obs.shape[0] <= self.max_padded_obj_num, "Too many obstacles" 
        
        obs, _ = self.create_padded_path_n_mask(obs, pad, self.max_padded_obj_num)

        return obs, obs_mask
    
    def get_static_cond(self, normalized_static_obs):
        # normalized_static_obs: 4 x 2
        obs_dim = [1,4,2]
        pad = torch.zeros(obs_dim)
        obs_mask = torch.zeros(self.max_padded_terrain_num)
        
        if len(normalized_static_obs) == 0:
            obs = torch.zeros(obs_dim)
            rows_to_add = self.max_padded_terrain_num - obs.shape[0]
            padding_tensor = pad.repeat(rows_to_add, 1, 1)
            obs = torch.cat((obs, padding_tensor), dim=0)
            return obs, obs_mask
            
        obs = torch.tensor(normalized_static_obs)
        obs_mask[:obs.shape[0]] = 1
        assert obs.shape[0] <= self.max_padded_terrain_num, "Too many obss" 

        rows_to_add = self.max_padded_terrain_num - obs.shape[0]
        padding_tensor = pad.repeat(rows_to_add, 1, 1)
        obs = torch.cat((obs, padding_tensor), dim=0)
        return obs, obs_mask
    
    def get_region_cond(self, normalized_static_obs):
        # temporary will be changed later
        static_obs_dim = 5
        pad = torch.zeros(static_obs_dim)
        terrain_mask = torch.zeros(self.max_padded_terrain_num)
        
        if len(normalized_static_obs) == 0:
            terrain = torch.zeros(1, 5) # hardcoded to 5
            terrain, _ = self.create_padded_path_n_mask(terrain, pad, self.max_padded_terrain_num)
            return terrain, terrain_mask
            
        terrain = torch.tensor(normalized_static_obs)
        terrain_mask[:terrain.shape[0]] = 1
        assert terrain.shape[0] <= self.max_padded_terrain_num, "Too many terrains" 
        
        terrain, _ = self.create_padded_path_n_mask(terrain, pad, self.max_padded_terrain_num)

        return terrain, terrain_mask
    
class TrajectoryDataset(BaseDataset, ProcessObsHelper):
    def __init__(self, cfg, input_data, fn_or_data: str, max_padded_len: int, max_obj_traj_len: int, max_padded_obj_num: int, max_padded_terrain_num: int):
        BaseDataset.__init__(self, input_data, fn_or_data, max_padded_len)
        ProcessObsHelper.__init__(self, max_obj_traj_len=max_obj_traj_len, 
                                  max_padded_obj_num=max_padded_obj_num,
                                  max_padded_terrain_num=max_padded_terrain_num)
        self.cfg = cfg

    def get_index(self, data):
        index = [i for i in range(len(data["trajectories"]))]
        return index
    
    def __getitem__(self, index):
        data, traj_index, fn = self.get_data_from_index(index)
        dynamic_obs, dynamic_obs_mask = self.get_obs_cond(data["obs"]["normalized_obstacle_seq"])
        if data["obs"].get("normalized_static") is None:
            static_obs, static_obs_mask = self.get_static_cond([])
        else:
            static_obs, static_obs_mask = self.get_static_cond(data["obs"]["normalized_static"])
        if data["obs"].get("normalized_prefer") is None:
            region_obs, region_obs_mask = self.get_region_cond([])
        else:
            region_obs, region_obs_mask = self.get_region_cond(data["obs"]["normalized_prefer"])

            
        trajectories = data["trajectories"][str(traj_index)]
        start = torch.tensor(trajectories["start"])
        goal = torch.tensor(trajectories["goal"])

        path, pad, hard_cond = self.parse_traj_representation(traj=trajectories, start=start, goal=goal)
        padded_path, mask = self.create_padded_path_n_mask(path, pad=pad, padded_len=self.max_padded_len)

        # 1/5 use actual preference 
        # # 11/20/2024 temporary add random terrain placeholder
        # random_x = np.random.uniform(-1, 1)
        # random_y = np.random.uniform(-1, 1)
        # random_angle = np.random.uniform(-1, 1)
        # random_width = np.random.uniform(-1, 1)
        # random_height = np.random.uniform(-1, 1)

        # terrain = torch.tensor([[random_x, random_y, random_angle, random_width, random_height]]) 

        # use_terrain = torch.rand(1).item() < 0.2
        # terrain_mask = torch.ones(terrain.shape[0]) if use_terrain else torch.zeros(terrain.shape[0])
        # # 11/20/2024 temporary add terrain placeholder

        result = {
            "filename": fn,
            "context_cond": {
                "dynamic_obs_encoder": {
                    "mask": dynamic_obs_mask,
                    "input_vals": [o for o in dynamic_obs]
                },
                "static_obs_encoder": {
                    "mask": static_obs_mask,
                    "input_vals": [o for o in static_obs]
                },
                "terrain_encoder": {
                    "mask": region_obs_mask,
                    "input_vals": [t for t in region_obs]
                },
                "goal_encoder": {
                    "mask": torch.ones(1),
                    "input_vals": goal
                }
            },
            "start": start,
            "goal": goal,
            "path": padded_path,
            "mask": mask,
            "hard_cond": hard_cond 
        }
        return result

class TrajectoryRandomStartDataset(BaseDataset, ProcessObsHelper):
    def __init__(self, cfg, input_data, fn_or_data: str, max_padded_len: int, max_obj_traj_len: int, max_padded_obj_num: int, max_padded_terrain_num: int):
        BaseDataset.__init__(self, input_data, fn_or_data, max_padded_len)
        ProcessObsHelper.__init__(self, max_obj_traj_len=max_obj_traj_len, 
                                  max_padded_obj_num=max_padded_obj_num,
                                  max_padded_terrain_num=max_padded_terrain_num)
        self.cfg = cfg

    def get_index(self, data):
        index = [i for i in range(len(data["trajectories"]))]
        return index
    
    def __getitem__(self, index):
        data, traj_index, fn = self.get_data_from_index(index)
        
        # compute trajectory first
        trajectories = data["trajectories"][str(traj_index)]
        start = torch.tensor(trajectories["start"])
        goal = torch.tensor(trajectories["goal"])

        path, pad, hard_cond = self.parse_traj_representation(traj=trajectories, start=start, goal=goal)
        start_idx = np.random.randint(0, len(path) - 10)
        path = path[start_idx:]
        hard_cond = {"0": path[0], "start_time_idx": start_idx} 
        padded_path, mask = self.create_padded_path_n_mask(path, pad=pad, padded_len=self.max_padded_len)
        
        # only implement for dynamic obs for now
        obstacles = [construct_obstacle_from_info(info) for info in data["obs"]["obstacles_info"]]
        dt = data["info"]["env_dt"]
        dynamic_obs = []
        dynamic_obs_mask = torch.zeros(self.max_padded_obj_num)
        for i in range(self.max_padded_obj_num):
            if i >= len(obstacles):
                dynamic_obs.append([0, 0, 0, 0, 0])
            else:
                dynamic_obs_mask[i] = 1 # still include the obstacle even if it is out of bound
                obs = obstacles[i]
                x, y, theta = obs.pos_at_dt(dt * start_idx)
                if x > 10 or x < -10 or y > 10 or y < -10:
                    # out of bound
                    dynamic_obs.append([0, 0, 0, 0, 0])
                    continue
                radius = obs.radius
                speed = obs.speed
                dynamic_obs.append([x/10, y/10, theta/np.pi, radius/2, (speed-1)]) # hardcoded
        dynamic_obs = torch.tensor(dynamic_obs).float()
        
        if data["obs"].get("normalized_static") is None:
            static_obs, static_obs_mask = self.get_static_cond([])
        else:
            static_obs, static_obs_mask = self.get_static_cond(data["obs"]["normalized_static"])
        if data["obs"].get("normalized_prefer") is None:
            region_obs, region_obs_mask = self.get_region_cond([])
        else:
            region_obs, region_obs_mask = self.get_region_cond(data["obs"]["normalized_prefer"])

        result = {
            "filename": fn,
            "context_cond": {
                "dynamic_obs_encoder": {
                    "mask": dynamic_obs_mask,
                    "input_vals": [o for o in dynamic_obs]
                },
                "static_obs_encoder": {
                    "mask": static_obs_mask,
                    "input_vals": [o for o in static_obs]
                },
                "terrain_encoder": {
                    "mask": region_obs_mask,
                    "input_vals": [t for t in region_obs]
                },
                "goal_encoder": {
                    "mask": torch.ones(1),
                    "input_vals": goal
                }
            },
            "start": start,
            "goal": goal,
            "path": padded_path,
            "mask": mask,
            "hard_cond": hard_cond
        }
        return result
    
class ScenarioDataset(BaseDataset, ProcessObsHelper):
    def __init__(self, input_data, fn_or_data: str, max_padded_len: int, max_obj_traj_len: int, max_padded_obj_num: int, max_padded_terrain_num: int):
        BaseDataset.__init__(self, input_data, fn_or_data, max_padded_len)
        ProcessObsHelper.__init__(self, max_obj_traj_len=max_obj_traj_len, 
                                  max_padded_obj_num=max_padded_obj_num,
                                  max_padded_terrain_num=max_padded_terrain_num)
        
    def get_index(self, **kwargs):
        return [1]
    
    def build_normalized_dynamic(self, data):
        if data.get("dynamic_obstacles") is None:
            return None, None
        raise NotImplementedError("Need to implement this offset")
        normalized_obstacle_seq = construct_normalized_obstacle_seq(data["dynamic_obstacles"], 
                                                 grid_size=data["env"]["grid_size"], 
                                                 max_planning_time=data["env"]["max_planning_time"], 
                                                 dt=data["env"]["env_dt"],
                                                 mult_x=data["env"]["mult_x"], 
                                                 mult_y=data["env"]["mult_y"])
        obs, obs_mask = self.get_obs_cond(normalized_obstacle_seq)
        return obs, obs_mask
    
    def build_normalized_static(self, data):
        if data.get("static_obstacles") is None:
            return None, None
        grid_size = data["env"]["grid_size"]
        mult_x = data["env"]["mult_x"]
        mult_y = data["env"]["mult_y"]
        normalized_static_obs = [normalize_static(info, grid_size, mult_x=mult_x, mult_y=mult_y) 
                                    for info in data["static_obstacles"]]
        terrains, terrain_mask = self.get_static_cond(normalized_static_obs) # TODO: make it consisteng with ddpo_agent, need to refactor later
        return terrains, terrain_mask
    
    def get_start_goal_normalized(self, data):
        normalizer = PathNormalizer(data["env"]["grid_size"])
        start = torch.tensor(data["objective"]["start"])
        goal = torch.tensor(data["objective"]["goal"])
        return normalizer.normalize(start), normalizer.normalize(goal)
    
    def get_obstacles(self, data):
        obstacles, obstacles_mask = self.build_normalized_dynamic(data)
        terrains, terrain_mask = self.build_normalized_static(data)
        return obstacles, obstacles_mask, terrains, terrain_mask
    
    def __getitem__(self, index):
        data, traj_index, fn = self.get_data_from_index(index)
        dynamic_obs, dynamic_obs_mask, static_obs, static_obs_mask = self.get_obstacles(data)
        start, goal = self.get_start_goal_normalized(data)

        hard_cond = {
            "0": start,
        }
        
        result = {
            "filename": fn,
            "context_cond": {
                "dynamic_obs_encoder": {
                    "mask": dynamic_obs_mask,
                    "input_vals": [o for o in dynamic_obs]
                } if dynamic_obs is not None else {},
                "static_obs_encoder": {
                    "mask": static_obs_mask,
                    "input_vals": [o for o in static_obs]
                } if static_obs is not None else {},
                "terrain_encoder": {
                    # tbd
                },
                "goal_encoder": {
                    "mask": torch.ones(1),
                    "input_vals": goal
                }
            },
            "start": start,
            "goal": goal,
            "state_cond": hard_cond,
            "grid_size": data["env"]["grid_size"]
        }
        return result