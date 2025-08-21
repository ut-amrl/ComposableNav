import os
import torch  
import random 
from termcolor import colored
from omegaconf import DictConfig
from hydra.utils import instantiate
from torch.utils.data import DataLoader 
from lightning import LightningDataModule

from composablenav.train.dataloader_base import TrajectoryDataset
from composablenav.misc.process_data import create_meta_file

class XYTrajectoryDataset(TrajectoryDataset):
    def __init__(self, config, input_data, fn_or_data: str, max_padded_len: int, max_obj_traj_len: int, max_padded_obj_num: int, max_padded_terrain_num: int):
        super().__init__(config, input_data, fn_or_data, max_padded_len, max_obj_traj_len, max_padded_obj_num, max_padded_terrain_num)
    
    def __len__(self):
        return self.cum_sum_indexes[-1]
    
    def parse_traj_representation(self, traj, start, goal):
        path = torch.tensor(traj["path"])[:, :2]
        pad = goal
        
        hard_cond = {"0": start}
        return path, pad, hard_cond
    
class CustomDataLoader(LightningDataModule):
    def __init__(self, config: DictConfig,  seed: int=42):
        super().__init__()

        meta_data_path = config.data_path + "_meta"
        if not os.path.exists(meta_data_path + "/train_metadata.json") or \
           not os.path.exists(meta_data_path + "/val_metadata.json"):
            print("Meta data not found, creating...")
            create_meta_file(config.data_path, config.train_split_ratio, verbose=False)

        train_input_data = meta_data_path + "/train_metadata.json"
        val_input_data = meta_data_path + "/val_metadata.json"

        self.train_dataset = instantiate(config.dataset_name, input_data=train_input_data)
        self.val_dataset = instantiate(config.dataset_name, input_data=val_input_data)
        self.batch_size = config.batch_size
        
        print(colored("\n===== DataLoader Summary =====", "cyan", attrs=["bold"]))
        print(f"📂 Train dataset size : {colored(len(self.train_dataset), 'green')}")
        print(f"📁 Val dataset size   : {colored(len(self.val_dataset), 'green')}")
        print(f"🧮 Batch size         : {colored(self.batch_size, 'yellow')}")
        print(colored("================================\n", "cyan"))
        
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