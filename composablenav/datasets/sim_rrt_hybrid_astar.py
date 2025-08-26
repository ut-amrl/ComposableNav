import time
import os 
from datetime import datetime
from joblib import delayed, Parallel
from copy import deepcopy
import hydra
from omegaconf import DictConfig
import numpy as np
import random 
import json 

from composablenav.datasets.scenario_generator import gen_noncolliding_obstacles, gen_regions, gen_goal, gen_plans_rrt_subgoal
from composablenav.misc.process_data import convert_to_saving_format_rrt_hybrid_astar
from composablenav.misc.common import load_data, load_fns_from_folder, save_json_data

def game_loop(obstacle_list, terrain_list, cfg: DictConfig, folder_name: str, training_idx: int):
    print(f"Start Generating: Training idx: {training_idx}")
    start_x, start_y = cfg.objective.start_loc
                
    for goal_idx in range(cfg.generation.num_goals):
        save_idx = training_idx * cfg.generation.num_goals + goal_idx
        goal_x, goal_y = gen_goal(cfg, obstacle_list, terrain_list, scenario=cfg.scenarios.scenario)
        if goal_x is None:
            print(f"No goal generated. Training idx: {training_idx}, Goal idx: {goal_idx}")
            return []
                  
        copied_obstacle_list = deepcopy(obstacle_list)
        copied_terrain_list = deepcopy(terrain_list)
        outputs = gen_plans_rrt_subgoal(cfg, start_loc=(start_x, start_y), goal_loc=(goal_x, goal_y), 
                            obstacles_list=copied_obstacle_list, terrain_list=copied_terrain_list)
       
        paths = []
        for i in range(len(outputs["paths"])):
            paths.append(outputs["paths"][i]["path"])
        
        if len(outputs) == 0:
            print(f"No plan generated. Training idx: {training_idx}, Goal idx: {goal_idx}")
            return  
        
        json_fn, outputs_dict = convert_to_saving_format_rrt_hybrid_astar(outputs, folder_name, save_idx, cfg)
            
        save_json_data(json_fn, outputs_dict)
        print(f"Generated: Training idx: {training_idx}, Goal idx: {goal_idx} Save idx: {save_idx}")
      
def generate_dynamic_env(cfg: DictConfig, folder_name: str, training_idx: int):
    obstacle_list = gen_noncolliding_obstacles(cfg)
    terrain_list = []
    game_loop(obstacle_list, terrain_list, cfg, folder_name, training_idx)

def generate_avoid_region_env(cfg: DictConfig, folder_name: str, training_idx: int):
    obstacle_list = []
    terrain_list = gen_regions(cfg) 
    game_loop(obstacle_list, terrain_list, cfg, folder_name, training_idx)

def generate_prefer_region_env(cfg: DictConfig, folder_name: str, training_idx: int):
    # find avoid region file
    avoid_idx = training_idx - cfg.generation.num_avoid
    load_avoid_json_fn = os.path.join(cfg.generation.output_dir, folder_name, f"{avoid_idx:07}.json")
    data = load_data(load_avoid_json_fn)
    
    data['obs']["normalized_obstacle_seq"] = []
    data['obs']["normalized_static"] = []
    data["obs"]["obstacles_info"] = []
    data["obs"]["terrain_info"] = []

    if data["trajectories"].get("0") is None:
        return None
    path = data["trajectories"]["0"]["path"]
    path_len = len(path)
    for _ in range(500):
        path_idx = np.random.randint(0, path_len)

        x = path[path_idx][0] * 10 # unnormalize
        y = path[path_idx][1] * 10 # unnormalize
        if x < -9 or x > 5 or y < -6 or y > 6:
            continue

        height = np.random.uniform(2, 6)
        width = np.random.uniform(0.5, 3)
        top = x + height / 2
        bottom = x - height / 2
        left = y + width / 2
        right = y - width / 2
        if top > 5 or bottom < -9 or left > 6 or right < -6:
            continue
        
        data['obs']["normalized_prefer"] = [[top/10, bottom/10, left/10, right/10, 0]]
        data["obs"]["terrain_info"].append({"type": "RectangleCornerPrefer", "top": top, "bottom": bottom, "left": left, "right": right})

        json_fn = os.path.join(cfg.generation.output_dir, folder_name, f"{training_idx:07}.json")
        save_json_data(json_fn, data)

def split_data(filenames, train_split_ratio, seed):
    random.seed(seed)
    random.shuffle(filenames)
    split_idx = int(len(filenames) * train_split_ratio)
    train_filenames = filenames[:split_idx]
    val_filenames = filenames[split_idx:]
    return train_filenames, val_filenames

def create_meta_file(cfg, saved_data_folder, verbose=True):
    filenames = load_fns_from_folder(saved_data_folder, format="json", sort=True)
    train_fns, val_fns = split_data(filenames, cfg.generation.split_train_percentage, seed=42)

    def build_metadata(fns):
        indexes, total = [], 0
        for i, fn in enumerate(fns):
            n = len(load_data(fn)["trajectories"])
            indexes.append(list(range(n)))
            total += n
            if verbose: print(i)
        return {"fns": fns, "indexes": indexes}, total

    save_dir = f"{saved_data_folder}_meta"
    os.makedirs(save_dir, exist_ok=True)

    train_meta, train_total = build_metadata(train_fns)
    with open(os.path.join(save_dir, "train_metadata.json"), "w") as f:
        json.dump(train_meta, f)
    if verbose: print("total train count", train_total)

    val_meta, val_total = build_metadata(val_fns)
    with open(os.path.join(save_dir, "val_metadata.json"), "w") as f:
        json.dump(val_meta, f)
    if verbose: print("total val count", val_total)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    
    cfg = cfg.dataset_generation
    create_meta_file(cfg, "../generated_data/pretrain_2025_08_22_16_07")
    exit()
    num_dynamic = cfg.generation.num_dynamic
    num_avoid = cfg.generation.num_avoid
    num_prefer = cfg.generation.num_prefer

    now_time = datetime.now().strftime("%Y_%m_%d_%H_%M")
    folder_name = f"{cfg.scenarios.save_name}_{now_time}"
    output_folder = os.path.join(cfg.generation.output_dir, folder_name)
    os.makedirs(output_folder, exist_ok=True)
    
    start_time = time.time()
    Parallel(n_jobs=cfg.generation.num_proc)(delayed(generate_dynamic_env)(cfg, folder_name, training_idx) for training_idx in range(0, num_dynamic))  
    print(f"Dynamic Env Generation Completed. Time taken: {time.time() - start_time}")
    
    start_time = time.time()
    Parallel(n_jobs=cfg.generation.num_proc)(delayed(generate_avoid_region_env)(cfg, folder_name, training_idx) for training_idx in range(num_dynamic, num_dynamic+ num_avoid))  
    print(f"Avoid Region Env Generation Completed. Time taken: {time.time() - start_time}")
    
    start_time = time.time()
    Parallel(n_jobs=cfg.generation.num_proc)(delayed(generate_prefer_region_env)(cfg, folder_name, training_idx) for training_idx in range(num_dynamic + num_avoid, num_dynamic + num_avoid + num_prefer))  
    print(f"Prefer Region Env Generation Completed. Time taken: {time.time() - start_time}")

    create_meta_file(cfg, output_folder)

if __name__ == '__main__':
    main()
