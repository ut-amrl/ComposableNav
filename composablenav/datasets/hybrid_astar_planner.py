import numpy as np
import heapq
from typing import Tuple, List
from composablenav.misc.common import forward_motion_rollout_simplified
from scipy.stats import beta

class Node:
    def __init__(self, x: float, y: float, theta: float, t_idx: int, v: float, w: float, parent=None):
        self.x = x
        self.y = y
        self.theta = theta
        self.t_idx = t_idx
        self.v = v
        self.w = w
        self.parent = parent
        self.g = np.inf # cost from start to node
        self.h = np.inf # heuristic cost from node to goal
        self.f = np.inf # total cost of node
        
        self.prev_nodes = [] # this will be useful when we use aggregate planning. in forward order
        
    def add_prev_node(self, prev_nodes):
        self.prev_nodes = prev_nodes

    def __lt__(self, other):
        return self.f < other.f
    
    def __str__(self):
        return f"Node(x={self.x}, y={self.y}, theta={self.theta}, v={self.v}, w={self.w}), t={self.t_idx}, g={self.g}, h={self.h}, f={self.f}"

class AStarPlanner:
    def __init__(self, obstacles_list, grid_size, grid_resolution, angle_resolution, env_dt,
                 max_v, max_w, max_dv, max_dw, v_partitions, w_partitions, 
                 max_planning_time, aggregate_planning_dt, target_vel_alpha, target_vel_beta,
                 time_multiplier, heuristic_multiplier):
        self.obstacles_list = obstacles_list
        self.grid_size = grid_size
        self.grid_resolution = grid_resolution
        self.angle_resolution = angle_resolution

        self.env_dt = env_dt

        self.max_v = max_v
        self.max_w = max_w
        self.max_dv = max_dv
        self.max_dw = max_dw
        self.v_range = np.round(np.linspace(0, self.max_v, v_partitions), 4)[::-1] # reverse the order
        self.w_range = np.round(np.linspace(-self.max_w, self.max_w, w_partitions), 4)
        
        self.max_planning_time = max_planning_time
        self.aggregate_planning_dt = aggregate_planning_dt
        
        self.target_vel_alpha = target_vel_alpha
        self.target_vel_beta = target_vel_beta
        self.time_multiplier = time_multiplier
        self.heuristic_multiplier = heuristic_multiplier

    def heuristic(self, curr_node, goal_node):
        d1 = self.distance(curr_node, goal_node) * self.heuristic_multiplier
        return d1 # temp: bias the heuristic to the one that makes the most progress

    def calculate_target_max_velocity(self):
        # define a distribution of target velocities
        # use a beta distribution
        # [0, 0.4] -> 6.22%
        # [0.4, 0.8] -> 16.82%
        # [0.8, 1.2] -> 24.43%
        # [1.2, 1.6] -> 27.70%
        # [1.6, 2.0] -> 22.32%
        # [2.0, 2.1] -> 2.51%
        alpha = self.target_vel_alpha
        beta_param = self.target_vel_beta

        # Generate Beta-distributed data
        beta_data = beta.rvs(alpha, beta_param, size=1)

        # Scale the data to fit the range [0, 2.1]
        scaled_data = beta_data * (self.max_v + 0.1)
        return scaled_data[0]


    def distance(self, node1, node2):
        return np.linalg.norm((node1.x - node2.x, node1.y - node2.y), 2)

    def compute_g_cost(self, curr_node, neighbor_node, target_max_velocity) -> float:
        curr_cost = curr_node.g
        dist_cost = self.distance(curr_node, neighbor_node)
        
        motion_cost = max(0, neighbor_node.v - target_max_velocity)
        time_cost = 1 * self.env_dt * self.time_multiplier
        g = curr_cost + dist_cost + motion_cost + time_cost
        return g
    
    def get_valid_vw(self, current_v, current_w) -> Tuple[List[float], List[float]]:
        # Calculate the lower and upper bounds based on max change in velocity
        min_v = max(0, current_v - self.max_dv) # cannot go backwards
        max_v = min(self.max_v, current_v + self.max_dv)

        # Filter the predefined velocity range to include only valid velocitie
        valid_v = [v for v in self.v_range if min_v <= v <= max_v]
        
        # Calculate the lower and upper bounds based on max change in velocity
        min_w = max(-self.max_w, current_w - self.max_dw)
        max_w = min(self.max_w, current_w + self.max_dw)

        # Filter the predefined velocity range to include only valid welocities
        valid_w = [w for w in self.w_range if min_w <= w <= max_w]
        return valid_v, valid_w
    
    def get_neighbors(self, node, max_time_idx):
        valid_v, valid_w = self.get_valid_vw(node.v, node.w)
        
        dt = self.env_dt
        neighbors = []
        for v in valid_v:
            for w in valid_w:
                curr_x, curr_y, curr_theta = node.x, node.y, node.theta
                prev_x, prev_y = curr_x, curr_y
                t_idx = node.t_idx
                collided = False
                prev_nodes = []
                for _ in range(self.aggregate_planning_dt):
                    curr_x, curr_y, curr_theta = forward_motion_rollout_simplified(v=v, w=w, x=curr_x, y=curr_y, theta=curr_theta, 
                                                                    planning_dt=dt)
                    curr_theta = self.normalize_angle(curr_theta) # ensure all angles are within [-pi, pi] to be added for the Node
                    
                    t_idx = min(t_idx + 1, max_time_idx)
                    prev_nodes.append(Node(curr_x, curr_y, curr_theta, t_idx, v, w))
                    
                    t_prev = (t_idx - 1) * dt
                    t_curr = t_idx * dt
                    for obstacle in self.obstacles_list:
                        if obstacle.collision(prev_x, prev_y, curr_x, curr_y, t_prev, 0) or \
                        obstacle.collision(prev_x, prev_y, curr_x, curr_y, t_curr, 0):
                            collided = True
                            break
                    prev_x, prev_y = curr_x, curr_y
                    if collided:
                        break
                    
                if collided:
                    continue
                
                if curr_x <= -self.grid_size//2 or curr_x >= self.grid_size//2 or \
                   curr_y <= -self.grid_size//2 or curr_y >= self.grid_size//2:
                    continue
                
                neighbor_node = prev_nodes[-1]
                neighbor_node.add_prev_node(prev_nodes[:-1])
                neighbor_node.parent = node
                
                neighbors.append(neighbor_node)
        return neighbors

    def reconstruct_path(self, curr_node):
        path = []
        curr_v, curr_w = 0, 0 # offset vel with state x y theta
        curr_v, curr_w = curr_node.v, curr_node.w
        while curr_node is not None:
            path.append((curr_node.x, curr_node.y, curr_node.theta, curr_v, curr_w, curr_node.t_idx))
            curr_v, curr_w = curr_node.v, curr_node.w
            prev_nodes = curr_node.prev_nodes[::-1] # reverse the order 
            for prev_node in prev_nodes:
                path.append((prev_node.x, prev_node.y, prev_node.theta, curr_v, curr_w, prev_node.t_idx))
                curr_v, curr_w = prev_node.v, prev_node.w
                
            curr_node = curr_node.parent
        return path[::-1]
    
    def goal_reached(self, curr_node, goal_node, goal_radius) -> bool:
        return self.distance(curr_node, goal_node) < goal_radius
    
    def normalize_angle(self, angle):
        normalized_angle = (angle + np.pi) % (2 * np.pi) - np.pi
        return normalized_angle
    
    def discretize_grid(self, node):
        grid_x = int(round(node.x / self.grid_resolution))
        grid_y = int(round(node.y / self.grid_resolution))
        grid_theta = int(round(node.theta / self.angle_resolution))
        # eliminate double counts when the angle is at the boundary
        max_abs_grid_theta = int(round(np.pi / self.angle_resolution))
        if abs(grid_theta) == max_abs_grid_theta:
            grid_theta = max_abs_grid_theta

        grid_t = node.t_idx
        return grid_x, grid_y, grid_theta, grid_t
    
    def add_to_close_set(self, closed_set, node):
        grid_x, grid_y, grid_theta, grid_t = self.discretize_grid(node)
        
        closed_set.add((grid_x, grid_y, grid_theta, grid_t))
        return closed_set
    
    def is_in_close_set(self, closed_set, node):
        grid_x, grid_y, grid_theta, grid_t = self.discretize_grid(node)
        return (grid_x, grid_y, grid_theta, grid_t) in closed_set
    
    def plan(self, start_node: Node, goal_node: Node, max_planning_time: int, goal_radius: float, max_expand_node_num: int):
        assert max_planning_time > 0 # max_planning_time should be greater than 0   
        target_max_velocity = self.calculate_target_max_velocity()  

        open_heap = []
        closed_set = set()
        start_node.g = 0
        start_node.h = self.heuristic(start_node, goal_node)
        start_node.f = start_node.g + start_node.h
        heapq.heappush(open_heap, (start_node.f, start_node))
        timeout_count = 0

        while open_heap:
            curr_node = heapq.heappop(open_heap)[1]
            if self.goal_reached(curr_node, goal_node, goal_radius):
                return [self.reconstruct_path(curr_node), target_max_velocity]
            
            if self.is_in_close_set(closed_set, curr_node):
                continue
            closed_set = self.add_to_close_set(closed_set, curr_node)
            
            timeout_count += 1
            if timeout_count > max_expand_node_num: # maximally expand 10000 nodes
                print("Astar planning timeout")
                return None
            
            for neighbor_node in self.get_neighbors(curr_node, max_planning_time - 1):
                if self.goal_reached(curr_node, goal_node, goal_radius): # early termination (optimizatioon trick)
                    return self.reconstruct_path(neighbor_node)

                if self.is_in_close_set(closed_set, neighbor_node):
                    continue
                
                neighbor_node.parrent = curr_node
                neighbor_node.g = self.compute_g_cost(curr_node, neighbor_node, target_max_velocity)
                neighbor_node.h = self.heuristic(neighbor_node, goal_node)
                neighbor_node.f = neighbor_node.g + neighbor_node.h
                heapq.heappush(open_heap, (neighbor_node.f, neighbor_node))