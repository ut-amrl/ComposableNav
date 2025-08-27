import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np 
import cv2 
import imageio
import copy 
from copy import deepcopy

import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
from composablenav.misc.normalizers import PathNormalizer, UniformNormalizer
from composablenav.misc.common import load_data, info_obs_goal_from_fn, forward_motion_rollout, get_path_from_motion
    
def convert_vw_to_traj(filename: str, generated_vws: np.ndarray, traj_idx=0, apply_snap=True):
    assert len(generated_vws.shape) == 2
    data = load_data(filename)
    grid_size =data["info"]["grid_size"]
    max_v = data["info"]["max_v"]
    max_w = data["info"]["max_w"]
    partitions = data["info"]["partitions"]
    planning_dt = 1 # temporary: need fix
    raise NotImplementedError("Need to implement the following function")
    start_normalized = np.array(data["trajectories"][str(traj_idx)]["start"])
    
    path_normalizer = PathNormalizer(grid_size)
    v_normalizer = UniformNormalizer(max_v)
    w_normalizer = UniformNormalizer(max_w)
    
    v_range = np.round(np.linspace(-max_v, max_v, partitions), 4)
    w_range = np.round(np.linspace(-max_w, max_w, partitions), 4)
    
    x, y = path_normalizer.unnormalize(start_normalized)
    theta = 0
    result_traj = []
    for v_normalized, w_normalized in generated_vws:
        v = v_normalizer.unnormalize(v_normalized)
        w = w_normalizer.unnormalize(w_normalized)
        if apply_snap:
            # snap to the closest value
            v = v_range.flat[np.abs(v_range - v).argmin()] 
            w = w_range.flat[np.abs(w_range - w).argmin()]
        
        x, y, theta = forward_motion_rollout(v, w, x, y, theta, planning_dt)
        result_traj.append([x, y])
    result_traj = np.array(result_traj)
    result_traj_normalized = path_normalizer.normalize(result_traj)
    return result_traj_normalized

# for critic gen
def visualize_scenario_with_path(obstacles_list, terrain_list, path_velocities, start_pos, start_theta, goal_pos, goal_radius, grid_size_2d, 
                       save_name, max_time=20, dt=1, frame_size=(640, 480)):
    
    start_offset = np.array([-7.5, 0, 0])
    global_to_start = np.linalg.inv(np.array([[np.cos(start_theta), -np.sin(start_theta), start_pos[0]], 
                                   [np.sin(start_theta), np.cos(start_theta), start_pos[1]],
                                   [0, 0, 1]]))   

    # grid_size_2d: x_min, x_max, y_min, y_max
    obstacles_list = copy.deepcopy(obstacles_list)
    terrain_list = copy.deepcopy(terrain_list)
    fig, ax = plt.subplots()
    frames = []
    
    current_path_x = start_pos[0]
    current_path_y = start_pos[1]
    path_vel_index = 0
    
    
    path_positions = [] 
    path_t = 0
    for t_idx in range(max_time):
        path_positions.append([current_path_x, current_path_y])
        for i in range(len(path_velocities)):
            if path_t >= path_velocities[i][2]:
                path_vel_index = i
          
        current_path_x += path_velocities[path_vel_index][0] * dt
        current_path_y += path_velocities[path_vel_index][1] * dt
        
        path_t += dt
        
    t = 0
    
    for t_idx in range(max_time):
        
        ax.clear()
        
        relative_start_x, relative_start_y, _ = global_to_start @ np.array([start_pos[0], start_pos[1], 1]) + start_offset
        relative_goal_x, relative_goal_y, _ = global_to_start @ np.array([goal_pos[0], goal_pos[1], 1]) + start_offset

        ax.plot(relative_start_y, relative_start_x, "xb")
        ax.plot(relative_goal_y, relative_goal_x, "xr")

        # ax.plot(start_pos[1], goal_pos[0], "xr")
        # ax.plot(goal_pos[1], goal_pos[0], "xr")
        goal_circle = patches.Circle((relative_goal_y, relative_goal_x), radius=goal_radius, color='r', fill=False)
        ax.add_patch(goal_circle)
        
        for obstacle in obstacles_list:
            obstacle.draw_with_offset(t, start_pos[0], start_pos[1], start_theta, -7.5, 0)  # Assuming obstacle has a draw method
            
        for terrain in terrain_list:
            terrain.draw(t)
            
        path_x, path_y = path_positions[t_idx]
            
        if path_x is not None and path_y is not None:
            relative_path_x, relative_path_y, _ = global_to_start @ np.array([path_x, path_y, 1]) + start_offset
            ax.plot(relative_path_y, relative_path_x, "ob")
            
        for i in range(len(path_positions) - 1):
            path_x1, path_y1, _ = global_to_start @ np.array([path_positions[i][0], path_positions[i][1], 1]) + start_offset
            path_x2, path_y2, _ = global_to_start @ np.array([path_positions[i + 1][0], path_positions[i + 1][1], 1]) + start_offset
            ax.plot([path_y1, path_y2], 
                [path_x1, path_x2], 
                color='gray', linestyle='-', linewidth=2)
            
        t += dt

        ax.grid(True)
        ax.axis(grid_size_2d)

    
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # Resize frame if necessary
        frame = cv2.resize(frame, frame_size)
        
        # Convert RGB to BGR (OpenCV uses BGR)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        frames.append(frame)
    # Save the frames as a gif
    imageio.mimsave(f"{save_name}.gif", frames, fps=5, loop=0)
    print(f"Saving to {save_name}.gif")
    print(f"Done generating with name {save_name}")  
    plt.close()

# for critic gen
def visualize_scenario(obstacles_list, terrain_list, start_pos, goal_pos, goal_radius, grid_size_2d, 
                       save_name, max_time=20, dt=1, frame_size=(640, 480)):
    fps = 5 / dt
    # grid_size_2d: x_min, x_max, y_min, y_max
    grid_size_2d[0], grid_size_2d[1] = grid_size_2d[1], grid_size_2d[0] # swap x_min and x_max to enforce robot coord frame
    obstacles_list = copy.deepcopy(obstacles_list)
    terrain_list = copy.deepcopy(terrain_list)
    fig, ax = plt.subplots()
    frames = []
    
    t = 0 
    for _ in range(max_time):
        ax.clear()

        ax.plot(start_pos[1], start_pos[0], "xr")
        ax.plot(goal_pos[1], goal_pos[0], "xr")
        goal_circle = patches.Circle((goal_pos[1], goal_pos[0]), radius=goal_radius, color='r', fill=False)
        ax.add_patch(goal_circle)
        
        for obstacle in obstacles_list:
            obstacle.draw(t)  # Assuming obstacle has a draw method
            
        for terrain in terrain_list:
            terrain.draw(t)
            
        t += dt

        ax.grid(True)
        ax.axis(grid_size_2d)

    
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # Resize frame if necessary
        frame = cv2.resize(frame, frame_size)
        
        # Convert RGB to BGR (OpenCV uses BGR)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        frames.append(frame)
    # Save the frames as a gif
    imageio.mimsave(f"{save_name}.gif", frames, fps=fps, loop=0)
    print(f"Saving to {save_name}.gif")
    print(f"Done generating with name {save_name}")  
    plt.close()
    

def visualize_diffusion(model, fn, x_shape, idx, sample_fn, context_cond, state_cond, save_name, frame_size=(640, 480), **kwargs):
    info, obstacles_list, goal_pos = info_obs_goal_from_fn(fn)
    chain = model.p_sample_loop(x_shape, sample_fn, diffusion_context_cond=context_cond, state_cond=state_cond)
    trajs = chain[-1].detach().cpu().numpy()[idx:idx+1]
    rewards = np.linspace(0, 1, trajs.shape[0]+1)
    trajs_unnormalized = PathNormalizer(info["grid_size"]).unnormalize(trajs).tolist()
    
    visualize_paths_with_rewards(trajs_unnormalized, rewards, obstacles_list, 
               goal_pos=goal_pos, goal_radius=info["goal_radius"], 
               grid_size=info["grid_size"], dt=info["env_dt"], 
               save_name=save_name, frame_size=frame_size)
    return chain

def visualize_paths_with_rewards(paths, rewards, obstacles_list, goal_pos, goal_radius, grid_size, dt, save_name, 
                                 start_time_idx=0, frame_size=(640, 480)):
    """
    paths: B x N x 2 (unnormalized)
    rewards: B (unnormalized)
    obstacles_list: unnormalized
    goal_pos: 2 (unnormalized)
    goal_radius: float (unnormalized)
    """
    fps = 2 / dt
    frames = []  
    if rewards.max() - rewards.min() == 0:
        normalized_rewards = np.zeros_like(rewards)
    else:
        normalized_rewards = (rewards - rewards.min()) / (rewards.max() - rewards.min())
    paths = deepcopy(paths)
    obstacles_list = deepcopy(obstacles_list)
    
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('viridis')
    cmap_bar_set = False
    
    t = start_time_idx * dt
    while True:
        ax.clear()
        
        for path_idx, path in enumerate(paths):
            if len(path) == 0:
                continue
            start = path[0]
            ax.plot(start[1], start[0], "xb")
            color = cmap(normalized_rewards[path_idx])
            
            for i in range(len(path) - 1):
                if np.linalg.norm([path[i][1] - goal_pos[1], path[i][0] - goal_pos[0]]) < goal_radius:
                    break
                ax.plot([path[i][1], path[i+1][1]], [path[i][0], path[i+1][0]], "-", color=color)
        ax.plot(goal_pos[1], goal_pos[0], "xr")
        goal_circle = patches.Circle((goal_pos[1], goal_pos[0]), radius=goal_radius, color='r', fill=False)
        ax.add_patch(goal_circle)
        
        for obstacle in obstacles_list:
            obstacle.draw(t)  # Assuming obstacle has a draw method
        
        t += dt
        ax.grid(True)
        ax.axis([grid_size//2, -grid_size//2, -grid_size//2, grid_size//2])
        
        # Set the colorbar label (acts as a legend)
        if not cmap_bar_set:
            sm = plt.cm.ScalarMappable(cmap=cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label('Value Gradient')
        cmap_bar_set = True
    
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # Resize frame if necessary
        frame = cv2.resize(frame, frame_size)
        
        # Convert RGB to BGR (OpenCV uses BGR)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        frames.append(frame)

        exit_flag = True
        for path in paths:
            if len(path) == 0:
                continue
            if np.linalg.norm([path[0][1] - goal_pos[1], path[0][0] - goal_pos[0]]) >= goal_radius:
                path.pop(0)
                exit_flag = False
        if exit_flag:
            break

    # Save the frames as a gif
    imageio.mimsave(f"{save_name}.gif", frames, fps=fps, loop=0)
    print(f"Saving to {save_name}.gif")
    print(f"Done generating with name {save_name}")  
    plt.close()
    
def visualize_paths_with_rewards_single_frame(paths, rewards, obstacles_list, goal_pos, goal_radius, grid_size, save_name, frame_size=(640, 480)):
    """
    Visualize and save the first frame of paths with rewards.
    
    Parameters:
    - paths: B x N x 2 (unnormalized)
    - rewards: B (unnormalized)
    - obstacles_list: List of obstacles
    - goal_pos: 2 (unnormalized)
    - goal_radius: float (unnormalized)
    - grid_size: Size of the grid
    - save_name: File name for saving the image
    - frame_size: Tuple for resizing the output image (width, height)
    """
    if rewards.max() - rewards.min() == 0:
        normalized_rewards = np.zeros_like(rewards)
    else:
        normalized_rewards = (rewards - rewards.min()) / (rewards.max() - rewards.min())
    
    paths = deepcopy(paths)
    obstacles_list = deepcopy(obstacles_list)
    
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('viridis')
    
    # Plot paths and rewards
    for path_idx, path in enumerate(paths):
        if len(path) == 0:
            continue
        start = path[0]
        ax.plot(start[1], start[0], "xb")  # Start point
        color = cmap(normalized_rewards[path_idx])
        
        for i in range(len(path) - 1):
            if np.linalg.norm([path[i][1] - goal_pos[1], path[i][0] - goal_pos[0]]) < goal_radius:
                break
            ax.plot([path[i][1], path[i+1][1]], [path[i][0], path[i+1][0]], "-", color=color)
    
    # Plot goal position and radius
    ax.plot(goal_pos[1], goal_pos[0], "xr")  # Goal point
    goal_circle = patches.Circle((goal_pos[1], goal_pos[0]), radius=goal_radius, color='r', fill=False)
    ax.add_patch(goal_circle)
    
    # Draw obstacles
    for obstacle in obstacles_list:
        obstacle.draw(0)  # Assuming obstacle has a draw method

    # Set grid and axis
    ax.grid(True)
    ax.axis([grid_size // 2, -grid_size // 2, -grid_size // 2, grid_size // 2])

    # Add colorbar for rewards
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Value Gradient')
    
    # Save the figure as an image
    fig.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"First frame saved as {save_name}.png")
    
    # Close the figure to release memory
    plt.close(fig)


# tempoorary: for grid size change
def visualize_paths_with_rewards_gridsizes(paths, rewards, obstacles_list, goal_pos, goal_radius, 
                                           grid_sizes, dt, save_name, start_t=0, frame_size=(640, 480)):
    """
    paths: B x N x 2 (unnormalized)
    rewards: B (unnormalized)
    obstacles_list: unnormalized
    goal_pos: 2 (unnormalized)
    goal_radius: float (unnormalized)
    """
    fps = 2 / dt
    frames = []  
    if rewards.max() - rewards.min() == 0:
        normalized_rewards = np.zeros_like(rewards)
    else:
        normalized_rewards = (rewards - rewards.min()) / (rewards.max() - rewards.min())
    paths = deepcopy(paths)
    obstacles_list = deepcopy(obstacles_list)
    
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('viridis')
    cmap_bar_set = False
    
    t = start_t
    while True:
        ax.clear()
        
        for path_idx, path in enumerate(paths):
            if len(path) == 0:
                continue
            start = path[0]
            ax.plot(start[1], start[0], "xb")
            color = cmap(normalized_rewards[path_idx])
            
            for i in range(len(path) - 1):
                if np.linalg.norm([path[i][1] - goal_pos[1], path[i][0] - goal_pos[0]]) < goal_radius:
                    break
                ax.plot([path[i][1], path[i+1][1]], [path[i][0], path[i+1][0]], "-", color=color)
        ax.plot(goal_pos[1], goal_pos[0], "xr")
        goal_circle = patches.Circle((goal_pos[1], goal_pos[0]), radius=goal_radius, color='r', fill=False)
        ax.add_patch(goal_circle)
        
        for obstacle in obstacles_list:
            obstacle.draw(t)  # Assuming obstacle has a draw method
        
        t += dt
        ax.grid(True)
        ax.axis(grid_sizes)
        
        # Set the colorbar label (acts as a legend)
        if not cmap_bar_set:
            sm = plt.cm.ScalarMappable(cmap=cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label('Value Gradient')
        cmap_bar_set = True
    
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # Resize frame if necessary
        frame = cv2.resize(frame, frame_size)
        
        # Convert RGB to BGR (OpenCV uses BGR)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        frames.append(frame)

        exit_flag = True
        for path in paths:
            if len(path) == 0:
                continue
            if np.linalg.norm([path[0][1] - goal_pos[1], path[0][0] - goal_pos[0]]) >= goal_radius:
                path.pop(0)
                exit_flag = False
        if exit_flag:
            break

    # Save the frames as a gif
    imageio.mimsave(f"{save_name}.gif", frames, fps=fps, loop=0)
    print(f"Saving to {save_name}.gif")
    print(f"Done generating with name {save_name}")  
    plt.close()
    
def visualize_rrt_astar_single_frame(paths, start, goal, obstacles_list, grid_size, goal_radius, subgoals, save_name):
    """
    Visualize a single frame of RRT-A* paths, start, goal, and obstacles.
    
    Parameters:
    - paths: List of paths, where each path is a list of (x, y) coordinates.
    - start: Tuple representing the start position (x, y).
    - goal: Tuple representing the goal position (x, y).
    - obstacles_list: List of obstacles, each with a `draw` method for visualization.
    - grid_size: Integer representing the size of the grid for visualization.
    - goal_radius: Radius around the goal to indicate success.
    - subgoals: List of subgoal positions (x, y).
    """
    fig, ax = plt.subplots()

    # Define the visualization grid limits
    grid_size_2d = [grid_size // 2, -grid_size // 2, -grid_size // 2, grid_size // 2]

    # Clear the plot
    ax.clear()

    # Plot each path
    for path_idx, path in enumerate(paths):
        if len(path) == 0:
            continue
        ax.plot(path[0][1], path[0][0], "xb")  # Mark the start of each path
        for i in range(len(path) - 1):
            ax.plot([path[i][1], path[i + 1][1]], [path[i][0], path[i + 1][0]], "-m")

    # Plot start and goal points
    ax.plot(start[1], start[0], "xr")
    ax.plot(goal[1], goal[0], "xr")

    # Draw the goal radius
    goal_circle = patches.Circle((goal[1], goal[0]), radius=goal_radius, color='r', fill=False)
    ax.add_patch(goal_circle)

    # Draw obstacles
    for obstacle in obstacles_list:
        obstacle.draw(0)  # Assuming obstacle has a draw method that accepts a time parameter

    # Plot subgoals if any
    if subgoals:
        for subgoal in subgoals:
            ax.plot(subgoal[1], subgoal[0], "xk")

    # Configure grid and display
    ax.grid(True)
    ax.axis(grid_size_2d)
    # Save the figure as an image
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"Frame saved as {save_name}.png")

    # Close the figure to release memory
    plt.close()

def visualize_rrt_astar(paths, start, goal, obstacles_list, dt, grid_size, goal_radius, subgoals, save_name): 
    
    paths = deepcopy(paths)
    obstacles_list = deepcopy(obstacles_list)
    
    fps = 1 / dt       
    # paths: list of path
    fig, ax = plt.subplots()
    frames = []
    
    frame_size=(640, 480)
    
    start = paths[0][0]

    
    grid_size_2d = [grid_size//2, -grid_size//2, -grid_size//2, grid_size//2]
    
    t = 0
    while True:
        ax.clear()

        for path_idx, path in enumerate(paths):
            if len(path) == 0:
                continue
            start = path[0]
            ax.plot(start[1], start[0], "xb")
            
            for i in range(len(path) - 1):
                if np.linalg.norm([path[i][1] - goal[1], path[i][0] - goal[0]]) < goal_radius:
                    break
                ax.plot([path[i][1], path[i+1][1]], [path[i][0], path[i+1][0]], "-m")

        ax.plot(start[1], start[0], "xr")
        ax.plot(goal[1], goal[0], "xr")
        goal_circle = patches.Circle((goal[1], goal[0]), radius=goal_radius, color='r', fill=False)
        ax.add_patch(goal_circle)

        for obstacle in obstacles_list:
            obstacle.draw(t)  # Assuming obstacle has a draw method

        if subgoals:
            for subgoal in subgoals:
                plt.plot(subgoal[1], subgoal[0], "xk")  
            
        t += dt

        ax.grid(True)
        ax.axis(grid_size_2d)

    
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # Resize frame if necessary
        frame = cv2.resize(frame, frame_size)
        
        frames.append(frame)

        exit_flag = True
        for path in paths:
            if len(path) == 0:
                continue
            if np.linalg.norm([path[0][1] - goal[1], path[0][0] - goal[0]]) >= goal_radius:
                path.pop(0)
                exit_flag = False
        if exit_flag:
            break
    # Save the frames as a gif
    imageio.mimsave(f"{save_name}.gif", frames, fps=fps, loop=0)
    print(f"Saving to {save_name}.gif")
    print(f"Done generating with name {save_name}")  
    plt.close()

def visualize_paths_from_controls(reference_path, controls, start, goal, dt, current_time, 
                                  obstacles_list, grid_size, goal_radius, mult_x, mult_y, 
                                  save_name=None, frame_size=(640, 480)): 
    """
    Visualize all paths in a single frame.

    Args:
    - paths: List of paths, where each path is a list of coordinates (x, y).
    - start: Starting point as a tuple (x, y).
    - goal: Goal point as a tuple (x, y).
    - obstacles_list: List of obstacles to be visualized (each should have a draw method).
    - grid_size: The size of the grid for visualization.
    - goal_radius: Radius around the goal for visualization.
    - subgoals: List of subgoal points to be visualized.
    - save_name: Optional file name to save the resulting image.
    """
    
    # paths = deepcopy(paths)
    paths = []
    for control in controls:
        # Assume control is (v controls, w controls)
        path = get_path_from_motion(control, start, dt)
        paths.append(path)
    obstacles_list = deepcopy(obstacles_list)

    # Create figure and axis
    fig, ax = plt.subplots()
    grid_size_2d = [grid_size//2 / mult_y, -grid_size//2 / mult_y, 0, grid_size / mult_x]
    
    for path in paths:
        if len(path) > 0:
            # ax.plot(path[0][1], path[0][0], "xb")  # Mark the start of each path
            for i in range(len(path) - 1):
                ax.plot([path[i][1], path[i+1][1]], [path[i][0], path[i+1][0]], "-m")
    
    for i in range(len(reference_path) - 1):
        ax.plot([reference_path[i][1], reference_path[i+1][1]], [reference_path[i][0], reference_path[i+1][0]], "-g")

    start_circle = patches.Circle((start[1], start[0]), radius=0.2, color='gray', fill=True)
    ax.add_patch(start_circle)
    # ax.plot(goal[1], goal[0], "xr")
    
    # Add goal circle
    goal_circle = patches.Circle((goal[1], goal[0]), radius=goal_radius, color='r', fill=False)
    ax.add_patch(goal_circle)

    # Plot obstacles
    for obstacle in obstacles_list:
        obstacle.draw(current_time)  # Assuming obstacle has a draw method

    # Set grid and axis limits
    ax.grid(True)
    ax.axis(grid_size_2d)

    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    # Resize frame if necessary
    frame = cv2.resize(frame, frame_size)
        
    # Save the visualization or show it
    plt.savefig(f"{save_name}.png")
    print(f"Saved visualization to {save_name}.png")

    plt.close()
    return frame