from composablenav.datasets.obstacles import Human, RectangleCornerAvoid, RectangleCornerPrefer
from composablenav.misc.common import load_data, repeat_context
from composablenav.misc.inference import (
    construct_context_state_conds, load_diffusion_models, get_guided_diffusion_path_vmap, stack_dictionaries_list
)
from composablenav.misc.normalizers import PathNormalizer
from composablenav.misc.process_data import (
    construct_normalized_dynamic_obstacle_from_obj, construct_normalized_static_obstacle_from_obj, construct_normalized_walk_over
)
from composablenav.train.dataloader_base import ProcessObsHelper

from omegaconf import OmegaConf
import numpy as np 
import torch 
from typing import List
from torch.func import stack_module_state
import copy 
import math 

class ComposableNavNode:
    def __init__(self, cfg: OmegaConf, device="cuda"):
        """
        This should operate in global diffusion view
        """
        self.cfg = cfg 
        self.device = device

        # load parameters
        self.grid_size: int = self.cfg.grid_size
        self.max_planning_time: int = self.cfg.max_planning_time
        self.diffusion_planning_dt: float = self.cfg.diffusion_planning_dt
        self.max_obj_traj_len: int = self.cfg.max_obj_traj_len
        self.max_padded_obj_num: int = self.cfg.max_padded_obj_num
        self.max_padded_terrain_num: int = self.cfg.max_padded_terrain_num
        self.pos_offset: List[float, float] = self.cfg.pos_offset
        self.compile = self.cfg.compile
        self.diffusion_models: dict = load_diffusion_models(self.cfg, self.device) 

    def initialize(self, scenario_fn):
        scenario_data = load_data(scenario_fn) # to be refactored
        scenario_data = OmegaConf.create(scenario_data)
        self.process_scenario_data(scenario_data)
        
        # to be initialized
        self.path_normalizer: PathNormalizer = None 
        self.process_obs_helper: ProcessObsHelper = None
    
        self.path_normalizer = PathNormalizer(self.grid_size)
        self.process_obs_helper = ProcessObsHelper(self.max_obj_traj_len, self.max_padded_obj_num, self.max_padded_terrain_num)
        
        goal_pos_offset = np.array(self.goal_pos) + np.array(self.pos_offset)
        self.goal_pos_normalized = self.path_normalizer.normalize(np.array(goal_pos_offset))
        self.pos_offset_normalized = self.path_normalizer.normalize(np.array(self.pos_offset))[:2]
        
        # set up neccessary diffusion context
        models = []
        self.anchor_model = self.diffusion_models[self.cfg.anchor_model_key]
        for task in self.tasks:
            model_key = task["model_key"]
            model = self.diffusion_models[model_key]
            models.append(model.model)
        self.params, self.buffers = stack_module_state(models)
        self.base_model = copy.deepcopy(self.anchor_model.model).to('meta')
        cfg_mask_cond = torch.ones(self.total_batch_size, device=self.device)
        cfg_mask_uncond = torch.zeros(self.total_batch_size, device=self.device)
        cfg_mask = torch.cat([cfg_mask_cond, cfg_mask_uncond], dim=0)
        self.num_models = len(self.tasks)
        self.cfg_mask = cfg_mask.unsqueeze(0).repeat(self.num_models, 1)
        
    def plan(self):
        dynamic_obs, static_obs, terrain_obs = self.parse_msg_to_obs(self.obs_data_raw)
        
        # construct context cond
        context_cond, state_cond = self.context_constructor(dynamic_obs, static_obs, terrain_obs, self.total_batch_size)

        context_conds = []
        for task in self.tasks:
            # task = OmegaConf.to_container(task, resolve=True)
            task["obs_idx"] = torch.tensor(task["obs_idx"])
            model_context = construct_context_state_conds(self.diffusion_models, context_cond, task)
            context_conds.append(model_context[1])
        stacked_context_cond = stack_dictionaries_list(context_conds)
        xt = None 
        n_timesteps = None         
                
        # core of the diffusion compose algorithm
        chain = get_guided_diffusion_path_vmap(stacked_context_cond, self.anchor_model, self.base_model, 
                                            self.params, self.buffers, self.num_models, self.cfg_mask,
                                            state_cond, n_timesteps, xt, self.max_planning_time, 
                                            self.total_batch_size, self.guidance_weight,
                                            self.device, compile=self.compile)

        return chain
    
    #### helper functions ###
    def process_scenario_data(self, scenario_data):
        # set up scenario data
        self.goal_pos: List[float, float] = scenario_data.goal_pos
        self.tasks: list[dict] = self.construct_new_meta(scenario_data.tasks)
        self.total_batch_size:int = scenario_data.total_batch_size
        self.guidance_weight = scenario_data.guidance_weight
        self.mult_x: float = scenario_data.mult_x
        self.mult_y: float = scenario_data.mult_y
        
        # parse different entities
        dynamic_data = []
        terrain_data = []
        static_data = []
        
        assert "entities" in scenario_data, "No entities or dynamic_obstacles found."
        for entity_dict in scenario_data["entities"]:
            if entity_dict["type"] == "RectangleCornerPrefer":
                terrain_data.append([entity_dict["top"], entity_dict["bottom"], entity_dict["left"], entity_dict["right"]])
            elif entity_dict["type"] == "RectangleCornerAvoid" or entity_dict["type"] == "RectangleCorner":
                static_data.append([entity_dict["top"], entity_dict["bottom"], entity_dict["left"], entity_dict["right"]]) 
            elif entity_dict["type"] == "Circle":
                x0 = entity_dict.get("x", 0.0)
                y0 = entity_dict.get("y", 0.0)
                theta = entity_dict.get("theta", 0.0)
                speed = entity_dict.get("speed", 0.0)

                # Generate a simple linear motion trajectory of length num_init_steps
                traj_points = []
                for i in range(self.max_planning_time):
                    t = i * self.diffusion_planning_dt
                    x_t = x0 + speed * t * math.cos(theta)
                    y_t = y0 + speed * t * math.sin(theta)
                    traj_points.append((x_t, y_t))
                dynamic_data.append(traj_points)
            else:
                raise ValueError(f"Unknown entity type: {entity_dict['type']}")           
          
        num_humans = len(dynamic_data)
        num_static = len(static_data)
        num_terrain = len(terrain_data)
        unscaled_data = [num_humans, num_static, num_terrain]
        for traj_points in dynamic_data:
            for traj_point in traj_points:
                unscaled_data.extend(traj_point)
                
        for static in static_data:
            unscaled_data.extend(static)

        for terrain in terrain_data:
            unscaled_data.extend(terrain)
    
        self.obs_data_raw = unscaled_data
        
    def construct_new_meta(self, tasks):
        new_tasks = []
        dynamic_obs = 0
        static_obs = 0
        terrain_obs = 0
        # merge static
        static_task = {"context_field": "static_obs_encoder", "model_key": "avoid", "weight": 1, "obs_idx": []}
        for task in tasks:
            context_field, model_key, obs_idx = task["context_field"], task["model_key"], task["obs_idx"]
            tmp_task = {"context_field": context_field, "model_key": model_key, "weight": 1}
            if len(obs_idx) == 1:
                if context_field == "dynamic_obs_encoder":
                    tmp_task["obs_idx"] = dynamic_obs
                    dynamic_obs += 1
                elif context_field == "static_obs_encoder":
                    static_task["obs_idx"].append(static_obs)
                    static_obs += 1
                    continue
                elif context_field == "terrain_encoder":
                    tmp_task["obs_idx"] = terrain_obs
                    terrain_obs += 1
                new_tasks.append(tmp_task)
            else:
                assert model_key == "pretrain"
                tmp_task["obs_idx"] = obs_idx
                new_tasks.append(tmp_task)
            
        # merge all static
        if len(static_task["obs_idx"]) > 0:
            new_tasks.append(static_task)
        return new_tasks

    def parse_msg_to_obs(self, data: list):
        """
        helper function
        """
        # need to scale observation accordingly
        num_dynamic_obs = int(data[0])
        num_static_obs = int(data[1])
        num_terrain_obs = int(data[2])
        raw_data = data[3:]
        dynamic_obs = []
        end_idx = 0
        for i in range(num_dynamic_obs):
            start_idx = i * 128 * 2
            end_idx = (i + 1) * 128 * 2
            future_poses = np.array(raw_data[start_idx:end_idx]).reshape(-1, 2)
            name = "human" + str(i)
            radius = 0.5
            dynamic_obs.append((Human(future_poses, name, radius, dt_interval=self.diffusion_planning_dt)))
        
        raw_data = raw_data[end_idx:]
        static_obs = []
        end_idx = 0
        for i in range(num_static_obs):
            start_idx = i * 4
            end_idx = (i + 1) * 4
            top, bottom, left, right = raw_data[start_idx:end_idx]
            static_obs.append((RectangleCornerAvoid(top, bottom, left, right)))
            
        raw_data = raw_data[end_idx:]
        terrain_obs = []
        for i in range(num_terrain_obs):
            start_idx = i * 4
            end_idx = (i + 1) * 4
            top, bottom, left, right = raw_data[start_idx:end_idx]
            terrain_obs.append((RectangleCornerPrefer(top, bottom, left, right)))
        return dynamic_obs, static_obs, terrain_obs
    
    def context_constructor(self, dynamic_obs, static_obs, terrain_obs, batch_size: int):
        """
        helper function
        """
        normalized_dynamic = construct_normalized_dynamic_obstacle_from_obj(dynamic_obs, 
                                                                            self.grid_size, self.max_planning_time, 
                                                                            self.diffusion_planning_dt,
                                                                            mult_x=self.mult_x, mult_y=self.mult_y,
                                                                            offset_x=self.pos_offset[0], offset_y=self.pos_offset[1]) 
        normalized_static = construct_normalized_static_obstacle_from_obj(static_obs, grid_size=self.grid_size,
                                                                          mult_x=self.mult_x, mult_y=self.mult_y,
                                                                          offset_x=self.pos_offset[0], offset_y=self.pos_offset[1])
        normalized_walk_over = construct_normalized_walk_over(terrain_obs, grid_size=self.grid_size,
                                                                          mult_x=self.mult_x, mult_y=self.mult_y,
                                                                          offset_x=self.pos_offset[0], offset_y=self.pos_offset[1])

        dynamic_obs, dynamic_obs_mask = self.process_obs_helper.get_obs_cond(normalized_dynamic)
        static_obs, static_obs_mask = self.process_obs_helper.get_static_cond(normalized_static)
        prefer_obs, prefer_obs_mask = self.process_obs_helper.get_region_cond(normalized_walk_over)

        dynamic_obs = dynamic_obs.to(self.device)
        dynamic_obs_mask = dynamic_obs_mask.to(self.device)
        static_obs = static_obs.to(self.device)
        static_obs_mask = static_obs_mask.to(self.device)
        prefer_obs = prefer_obs.to(self.device)
        prefer_obs_mask = prefer_obs_mask.to(self.device)
        
        # temporary for batch size = 2
        context_cond = {
            "dynamic_obs_encoder": {
                "mask": dynamic_obs_mask.unsqueeze(0),
                "input_vals": [o.unsqueeze(0) for o in dynamic_obs]
            },
            "static_obs_encoder": {
                "mask": static_obs_mask.unsqueeze(0),
                "input_vals": [o.unsqueeze(0) for o in static_obs]
            },
            "goal_encoder": {
                "mask": torch.ones([1,1]).to(self.device).float(),
                "input_vals": torch.tensor([self.goal_pos_normalized], device=self.device).float()
            },
            "terrain_encoder": {
                "mask": prefer_obs_mask.unsqueeze(0),
                "input_vals": [o.unsqueeze(0) for o in prefer_obs]
            }
        }

        context_cond = repeat_context(context_cond, 2 * batch_size) # repeat for batch size *2 because cond and uncond
    
        state_cond = {
            "0": torch.tensor([self.pos_offset_normalized]).to(self.device),
        }
        return context_cond, state_cond