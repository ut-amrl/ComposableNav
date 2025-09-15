from torch.utils.data import DataLoader 
from pytorch_lightning import LightningDataModule
import torch  
from omegaconf import DictConfig
import random 
import pickle 

from composablenav.train.dataloader_base import (
    TrajectoryDataset,
    TrajectoryRandomStartDataset,
    ScenarioDataset
)
from composablenav.misc.common import load_fns_from_folder


class XYTrajectoryDataset(TrajectoryDataset):
    def __init__(self, cfg, input_data, fn_or_data: str, max_padded_len: int, max_obj_traj_len: int, max_padded_obj_num: int, max_padded_terrain_num: int):
        super().__init__(cfg, input_data, fn_or_data, max_padded_len, max_obj_traj_len, max_padded_obj_num, max_padded_terrain_num)
    
    def __len__(self):
        return self.cum_sum_indexes[-1]
    
    def parse_traj_representation(self, traj, start, goal):
        path = torch.tensor(traj["path"])[:, :2]
        pad = goal
        
        # ######### 12/08/2024 #########
        # hard_cond = {}
        # for idx in range(self.max_obj_traj_len//3): # hardcoded
        #     if idx >= len(path):
        #         hard_cond[str(idx)] = goal
        #     else:
        #         hard_cond[str(idx)] = path[idx]
        # hard_cond["path_len"] = len(path)
        # ######### 12/08/2024 #########
        hard_cond = {"0": path[0]}
        return path, pad, hard_cond

class XYThetaStartTrajectoryDataset(TrajectoryRandomStartDataset):
    def __init__(self, cfg, input_data, fn_or_data: str, max_padded_len: int, max_obj_traj_len: int, max_padded_obj_num: int, max_padded_terrain_num: int):
        super().__init__(cfg, input_data, fn_or_data, max_padded_len, max_obj_traj_len, max_padded_obj_num, max_padded_terrain_num)
    
    def __len__(self):
        return int(self.cum_sum_indexes[-1] // 10)
    
    def parse_traj_representation(self, traj, start, goal):
        path = torch.tensor(traj["path"])[:, :3]
        pad = torch.cat([goal, torch.tensor([0.0])])
        hard_cond = {"0": path[0]} # not used
        return path, pad, hard_cond
    
class XYThetaTrajectoryDataset(TrajectoryDataset):
    def __init__(self, cfg, input_data, fn_or_data: str, max_padded_len: int, max_obj_traj_len: int, max_padded_obj_num: int, max_padded_terrain_num: int):
        super().__init__(cfg, input_data, fn_or_data, max_padded_len, max_obj_traj_len, max_padded_obj_num, max_padded_terrain_num)
    
    def __len__(self):
        return int(self.cum_sum_indexes[-1] // 10)
    
    def parse_traj_representation(self, traj, start, goal):
        path = torch.tensor(traj["path"])[:, :3]
        # fake path to 0
        # path[:, 2] = 0 only for fake cond
        pad = torch.cat([goal, torch.tensor([0.0])])
        hard_cond = {"0": path[0]}
        hard_cond["path_len"] = 1
        return path, pad, hard_cond
     
class XYEnsembleTrajectoryDataset(TrajectoryDataset):
    def __init__(self, cfg, input_data, fn_or_data: str, max_padded_len: int, max_obj_traj_len: int, max_padded_obj_num: int, max_padded_terrain_num: int):
        super().__init__(cfg, input_data, fn_or_data, max_padded_len, max_obj_traj_len, max_padded_obj_num, max_padded_terrain_num)
    
    def __len__(self):
        return int(self.cum_sum_indexes[-1] // 10)
    
    def parse_traj_representation(self, traj, start, goal):
        path = torch.tensor(traj["path"])[:, :2]
        pad = goal
        hard_cond = {"0": start}
        hard_cond["path_len"] = 1
        return path, pad, hard_cond
    
class XYScenarioDataset(ScenarioDataset):
    # multimodality
    def __init__(self, input_data, fn_or_data: str, max_padded_len: int, max_obj_traj_len: int, max_padded_obj_num: int, max_padded_terrain_num: int):
        super().__init__(input_data, fn_or_data, max_padded_len, max_obj_traj_len, max_padded_obj_num, max_padded_terrain_num)
        
    def parse_traj_representation(self, traj, start, goal):
        path = torch.tensor(traj["path"])[:, :2]
        pad = goal
        hard_cond = {"0": start}
        return path, pad, hard_cond

class CustomDataLoader(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        CustomDataset = eval(cfg.dataset_name)
        print("Custom Dataloader loading...")
        if cfg.dataset_args.fn_or_data == "fn":
            filenames = load_fns_from_folder(cfg.data_path, format="json", sort=True)
            train_input_data, val_input_data = self.split_data(filenames, cfg.train_split_ratio, seed=42)
        elif cfg.dataset_args.fn_or_data == "json_data":
            input_data = pickle.load(open(cfg.data_path, "rb"))
            train_input_data, val_input_data = self.split_data(input_data, cfg.train_split_ratio, seed=42)
        elif cfg.dataset_args.fn_or_data == "meta_file":
            train_input_data = cfg.data_path + "/train_metadata.json"
            val_input_data = cfg.data_path + "/val_metadata.json"
  
        print("Training dataset:")
        self.train_dataset = CustomDataset(cfg=cfg, input_data=train_input_data, **cfg.dataset_args)
        print("Validation dataset:")
        self.val_dataset = CustomDataset(cfg=cfg, input_data=val_input_data, **cfg.dataset_args)
        self.batch_size = cfg.batch_size
        print("Custom Dataloader data loaded!")
        
    def split_data(self, filenames, train_split_ratio, seed):
        random.seed(seed)
        random.shuffle(filenames)
        split_idx = int(len(filenames) * train_split_ratio)
        train_filenames = filenames[:split_idx]
        val_filenames = filenames[split_idx:]
        return train_filenames, val_filenames
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, 
                          pin_memory=True, persistent_workers=True, num_workers=4, prefetch_factor=2)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, 
                          pin_memory=True, persistent_workers=True, num_workers=4, prefetch_factor=2)
    
class EvalDataLoader(LightningDataModule):
    def __init__(self, cfg: DictConfig, filenames: list):
        super().__init__()
        CustomDataset = eval(cfg.dataset_name)
        
        self.val_dataset = CustomDataset(input_data=filenames, **cfg.dataset_args)
        self.batch_size = cfg.batch_size
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

import hydra
@hydra.main(config_path="../conf", config_name="config")
def test_main(cfg):
    dataloader = CustomDataLoader(cfg.data)
    masks = []
    for batch in dataloader.train_dataloader():
        data = batch["context_cond"]["obs_encoder"]["input_vals"]
        mask = batch["context_cond"]["obs_encoder"]["mask"]
        goal_mask = batch["context_cond"]["goal_encoder"]["mask"]
        inst = batch["context_cond"]["instruction_encoder"]
        print(inst)
        print(len(data))
        print(data[0].shape)
        print(mask.shape)
        print(goal_mask.shape)
        masks.append(mask)
        masks.append(goal_mask)
        print(torch.cat(masks, dim=1))
        break
    
if __name__ == "__main__":
    test_main()