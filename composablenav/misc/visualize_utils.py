import cv2, imageio, numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from copy import deepcopy
from typing import List, Tuple
from composablenav.datasets.obstacles import Obstacle

import matplotlib.patches as patches


def _draw_path(ax, path, goal_pos, goal_radius, *, color='b', start_r=0.5, arrow_len=0.5):
    """Draw start circle, heading arrow, and polyline until within goal_radius."""
    if len(path) == 0: 
        return
    yx = lambda p: (p[1], p[0])           # (y,x) for plotting
    start = path[0]

    # start circle
    ax.add_patch(patches.Circle(yx(start), start_r, edgecolor=color, facecolor='none', lw=4))

    # heading arrow (from first step)
    if len(path) > 1:
        d = np.array(path[1][::-1][-2:]) - np.array(path[0][::-1][-2:])  # (y,x) diff
        n = np.hypot(*d)
        if n > 1e-6:
            d = d / n * arrow_len
            ax.arrow(*yx(start), d[0], d[1], head_width=0.8, head_length=0.8, fc=color, ec=color)
            ax.plot(*yx(start), "x"+color)

    # polyline until inside goal
    for a, b in zip(path[:-1], path[1:]):
        if np.linalg.norm(np.array(a[::-1][-2:]) - np.array(goal_pos[::-1][-2:])) < goal_radius:
            break
        ax.plot([a[1], b[1]], [a[0], b[0]], "-", c=color, linewidth=3)

def plot_path_gif(paths: List[List], obstacles_list: List[Obstacle], goal_pos: Tuple[float, float], 
                                 goal_radius: float, grid_size: int, dt: float, save_name: str, 
                                 start_time_idx=0, frame_size=(480, 480)):
    """paths: B x N x 2 (unnormalized)"""
    fps = 2 / dt
    frames = []
    paths, obstacles_list = deepcopy(paths), deepcopy(obstacles_list)

    fig, ax = plt.subplots()
    t = start_time_idx * dt

    if len(paths) == 0:
        init_robot_loc_x, init_robot_loc_y = 0, 0
    else:
        init_robot_loc_x, init_robot_loc_y = paths[0][0][:2]

    while True:
        ax.clear()
        
        # obstacles
        for ob in obstacles_list:
            ob.draw(t)
            
        # paths
        for path in paths:
            _draw_path(ax, path, goal_pos, goal_radius, color='b', start_r=0.5, arrow_len=0.5)

        # goal
        ax.plot(goal_pos[1], goal_pos[0], "xr")
        ax.add_patch(patches.Circle((goal_pos[1], goal_pos[0]), goal_radius, color='r', fill=False, lw=4))

        # The limits are modified to align with the robot frame coordinates, e.g. x forward, y left
        ax.set_xlim(init_robot_loc_y+grid_size//2+1, init_robot_loc_y-grid_size//2-1)
        ax.set_ylim(init_robot_loc_x-1, init_robot_loc_x+grid_size+1)
        ax.set_aspect('equal')
        ax.axis('off')
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # render frame
        fig.subplots_adjust(0, 0, 1, 1)  # axes fill figure

        # render frame
        fig.canvas.draw()

        # grab full canvas as RGB
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        W, H = fig.canvas.get_width_height()
        full = buf.reshape(H, W, 3)

        # --- crop exactly to the axes region ---
        # ax.bbox is in display coords with origin at bottom-left; array origin is top-left
        x0, y0, w_ax, h_ax = map(int, ax.bbox.bounds)
        y0_top = H - (y0 + h_ax)          # convert to array row index
        y1_top = H - y0
        crop = full[y0_top:y1_top, x0:x0 + w_ax, :]

        # resize and convert for OpenCV/GIF pipeline
        frame = cv2.cvtColor(cv2.resize(crop, frame_size), cv2.COLOR_RGB2BGR)
        frames.append(frame)

        # advance paths
        t += dt
        exit_flag = True
        for path in paths:
            if len(path) > 0 and np.linalg.norm(np.array(path[0][::-1][-2:]) - np.array(goal_pos[::-1][-2:])) >= goal_radius:
                path.pop(0); exit_flag = False
        if exit_flag:
            break

    imageio.mimsave(save_name, frames, fps=fps, loop=0)
    print(f"Saved {save_name}")
    plt.close()
