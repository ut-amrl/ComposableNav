from abc import ABC
from typing import Union
import numpy as np 
import torch 

from composablenav.datasets.obstacles import Obstacle

class UniformNormalizer(ABC):
    def __init__(self, normalizing_constant):
        self.normalizing_constant = normalizing_constant
    
    def normalize_preprocess(self, x):
        return x
    
    def unnormalize_postprocess(self, x):
        return x
    
    def normalize(self, x: Union[float, np.ndarray]):
        assert type(x) != list
        x = self.normalize_preprocess(x)
        return x / self.normalizing_constant
    
    def unnormalize(self, x: Union[float, np.ndarray]):
        assert type(x) != list
        x = x * self.normalizing_constant
        return self.unnormalize_postprocess(x)

class PathNormalizer(UniformNormalizer):
    def __init__(self, grid_size: int) -> None:
        super().__init__(grid_size // 2)
    

class ScaledNormalizer:
    # only used for construct_normalized_obstacle_seq
    def __init__(self, grid_size, mult_x, mult_y):
        self.normalizing_constant = grid_size//2
        self.mult_x = mult_x
        self.mult_y = mult_y
        
    def normalize(self, x):
        assert type(x) != list
        assert len(x.shape) == 1
        assert len(x) == 2
        x[0] = x[0] * self.mult_x / self.normalizing_constant
        x[1] = x[1] * self.mult_y / self.normalizing_constant
        return x

class VelocityNormalizer(UniformNormalizer):
    def __init__(self, max_speed: float) -> None:
        super().__init__(np.sqrt(max_speed))
      
class SpeedNormalizer(UniformNormalizer):
    def __init__(self, max_speed: float) -> None:
        super().__init__(max_speed / 2)
        
    def normalize_preprocess(self, speed):
        # speed goes from 0 to max speed, I want to normalize it between -1, 1
        return speed - self.normalizing_constant
    
    def unnormalize_postprocess(self, speed):
        return speed + self.normalizing_constant
        
class AngleNormalizer(UniformNormalizer):
    def __init__(self, max_angle: float) -> None:
        super().__init__(max_angle)
        
    def normalize_preprocess(self, angle):
        # Normalize the angle to the range [-pi, pi) => this will not affect gradient as it has grad of 1
        normalized_angle = (angle + np.pi) % (2 * np.pi) - np.pi
        return normalized_angle
    
class SizeNormalizer(UniformNormalizer):
    def __init__(self, min_size: float, max_size: float) -> None:
        super().__init__(max_size - min_size)
        self.max_size = max_size
        self.min_size = min_size
        
    def normalize_preprocess(self, size):
        # Normalize to [0, max_size - min_size]
        return 2 * (size - self.min_size) - (self.max_size - self.min_size)
    
    def unnormalize_postprocess(self, size):
        return (size + (self.max_size - self.min_size)) / 2 + self.min_size
    
class DifferentialPathNormalizer:
    def __init__(self, grid_size: int, max_angle: float, max_vel: float, max_ang_vel: float) -> None:
        self.path_normalizer = PathNormalizer(grid_size)
        self.angle_normalizer = AngleNormalizer(max_angle)
        self.vel_normalizer = SpeedNormalizer(max_vel) # 10/24: velocity is changed from (-2,2) to speed, breaking changes
        self.angular_normalizer = UniformNormalizer(max_ang_vel)
        
    def normalize(self, data: np.ndarray):
        """
        data: [x, y, theta, v, w]
        """
        assert type(data) == np.ndarray
        assert len(data.shape) == 2 and data.shape[1] == 5, f"Data shape is invalid with shape {data.shape}"
        path = data[:, :2]
        angle = data[:, 2:3]
        vel = data[:, 3:4]
        angular_vel = data[:, 4:5]
        
        path = self.path_normalizer.normalize(path)
        angle = self.angle_normalizer.normalize(angle)
        vel = self.vel_normalizer.normalize(vel)
        angular_vel = self.angular_normalizer.normalize(angular_vel)
        
        result = np.concatenate([path, angle, vel, angular_vel], axis=1)
        return result
    
    def unnormalize(self, data: list):
        """
        data: [x, y, theta, v, w]
        """
        data = np.array(data)
        assert len(data.shape) == 2 and data.shape[1] == 5, f"Data shape is invalid with shape {data.shape}"
        path = data[:, :2]
        angle = data[:, 2:3]
        vel = data[:, 3:4]
        angular_vel = data[:, 4:5]
        
        path = self.path_normalizer.unnormalize(path)
        angle = self.angle_normalizer.unnormalize(angle)
        vel = self.vel_normalizer.unnormalize(vel)
        angular_vel = self.angular_normalizer.unnormalize(angular_vel)
        
        result = np.concatenate([path, angle, vel, angular_vel], axis=1).tolist()
        return result
    
class TensorDifferentialPathNormalizer:
    def __init__(self, grid_size: int, max_angle: float, max_vel: float, max_ang_vel: float) -> None:
        self.path_norm = PathNormalizer(grid_size)
        self.angle_norm = AngleNormalizer(max_angle)
        self.vel_norm = SpeedNormalizer(max_vel) # 10/24: velocity is changed from (-2,2) to speed
        self.ang_vel_norm = UniformNormalizer(max_ang_vel)
        
    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        assert data.shape == (data.shape[0], data.shape[1], 5), f"Invalid shape: {data.shape}"
        path, angle, vel, ang_vel = data.split([2, 1, 1, 1], dim=2)
        return torch.cat([
            self.path_norm.normalize(path),
            self.angle_norm.normalize(angle),
            self.vel_norm.normalize(vel),
            self.ang_vel_norm.normalize(ang_vel)
        ], dim=2)
    
    def unnormalize(self, data: torch.Tensor) -> torch.Tensor:
        assert data.shape == (data.shape[0], data.shape[1], 5), f"Invalid shape: {data.shape}"
        path, angle, vel, ang_vel = data.split([2, 1, 1, 1], dim=2)
        return torch.cat([
            self.path_norm.unnormalize(path),
            self.angle_norm.unnormalize(angle),
            self.vel_norm.unnormalize(vel),
            self.ang_vel_norm.unnormalize(ang_vel)
        ], dim=2)
        
class ObstacleNormalizer:
    def __init__(self, grid_size: int, max_angle: float, max_speed:float) -> None:
        self.path_norm = PathNormalizer(grid_size)
        self.angle_norm = AngleNormalizer(max_angle)
        self.speed_norm = SpeedNormalizer(max_speed) # speed goes from 0 to 1, I want to normalize it between -1, 1
        
    def normalize(self, obstacle: Union[Obstacle, dict, np.ndarray]) -> torch.Tensor:
        assert type(obstacle) != list
        data = obstacle
        if isinstance(obstacle, Obstacle):
            data = np.array([obstacle.x, obstacle.y, obstacle.theta, obstacle.speed])
        elif type(obstacle) == dict:
            data = np.array([obstacle["x"], obstacle["y"], obstacle["theta"], obstacle["speed"]])
        
        if len(data.shape) == 1:
            pos = self.path_norm.normalize(data[:2])
            angle = self.angle_norm.normalize(data[2:3]) 
            speed = self.speed_norm.normalize(data[3:4]) 
            return np.concatenate([pos, angle, speed], axis=0)
        elif len(data.shape) == 2:
            pos = self.path_norm.normalize(data[:, :2])
            angle = self.angle_norm.normalize(data[:, 2:3])
            speed = self.speed_norm.normalize(data[:, 3:4])
            return torch.cat([pos, angle, speed], dim=1)
        else:
            raise ValueError(f"Invalid shape: {data.shape}")
        
class TerrainNormalizer:
    # this has been changed before training the text 15k
    def __init__(self, grid_size: int) -> None:
        self.path_norm = PathNormalizer(grid_size)
        self.angle_norm = SizeNormalizer(0, np.pi/4) # I am using Rectangle as terrain, so the angle is always between [0, pi/4]
        self.size_norm = SizeNormalizer(0, grid_size/2) # not senstive to values used for generation
        
    def normalize(self, terrain: Union[Obstacle, dict, np.ndarray]) -> torch.Tensor:
        assert type(terrain) != list
        data = terrain
        if isinstance(terrain, Obstacle):
            data = np.array([terrain.x, terrain.y, terrain.theta, terrain.width, terrain.height])
        elif type(terrain) == dict:
            data = np.array([terrain["x"], terrain["y"], terrain["theta"], terrain["width"], terrain["height"]])
        
        if len(data.shape) == 1:
            pos = self.path_norm.normalize(data[:2])
            angle = self.angle_norm.normalize(data[2:3]) 
            width_height = self.size_norm.normalize(data[3:5]) 
            return np.concatenate([pos, angle, width_height], axis=0, dtype=np.float32)
        elif len(data.shape) == 2:
            pos = self.path_norm.normalize(data[:, :2])
            angle = self.angle_norm.normalize(data[:, 2:3]) 
            width_height = self.size_norm.normalize(data[:, 3:5]) 
            return torch.cat([pos, angle, width_height], dim=1)
        else:
            raise ValueError(f"Invalid shape: {data.shape}")