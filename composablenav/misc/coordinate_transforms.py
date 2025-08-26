from typing import Tuple
import numpy as np

import numpy as np

def coordinate_transform(ego_coord, target_coord):
    # Create the transformation matrix from ego frame
    transform_matrix = np.array([
        [np.cos(ego_coord[2]), -np.sin(ego_coord[2]), ego_coord[0]],
        [np.sin(ego_coord[2]), np.cos(ego_coord[2]), ego_coord[1]],
        [0, 0, 1]
    ])
    
    # Convert target coordinates to homogeneous coordinates
    target_coord_homogeneous = np.array([target_coord[0], target_coord[1], 1])
    
    # Transform target coordinates to the ego frame
    target_in_ego_frame = np.linalg.inv(transform_matrix) @ target_coord_homogeneous
    
    # Compute the new orientation (theta) in the ego frame
    new_theta = target_coord[2] - ego_coord[2]
    
    # Normalize new_theta to be within [-pi, pi]
    new_theta = (new_theta + np.pi) % (2 * np.pi) - np.pi
    
    return [target_in_ego_frame[0], target_in_ego_frame[1], new_theta]

def coordinate_transform_to_global(robot_global_coord, target_coord):
    # Create the transformation matrix from global frame
    transform_matrix = np.array([
        [np.cos(robot_global_coord[2]), -np.sin(robot_global_coord[2]), robot_global_coord[0]],
        [np.sin(robot_global_coord[2]), np.cos(robot_global_coord[2]), robot_global_coord[1]],
        [0, 0, 1]
    ])
    
    # Convert target coordinates to homogeneous coordinates
    target_coord_homogeneous = np.array([target_coord[0], target_coord[1], 1])
    
    # Transform target coordinates to the global frame
    target_in_global_frame = transform_matrix @ target_coord_homogeneous
    
    # Compute the new orientation (theta) in the global frame
    new_theta = target_coord[2] + robot_global_coord[2]
    
    # Normalize new_theta to be within [-pi, pi]
    new_theta = (new_theta + np.pi) % (2 * np.pi) - np.pi
    
    return [target_in_global_frame[0], target_in_global_frame[1], new_theta]
