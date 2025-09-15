
import numpy as np
from omegaconf import DictConfig
from composablenav.datasets.hybrid_astar_planner import AStarPlanner, Node
from composablenav.datasets.rrt_planner import RRTPlanner, RRTNode
from composablenav.datasets.obstacles import Circle, CircleNoisy, Rectangle, RectangleCorner
from composablenav.misc.common import save_json_data, validate_plan
from composablenav.misc.visualizer_utils import visualize_rrt_astar

#################### Obstacle Generation ####################
def random_obstacle(cfg: DictConfig):
    x = np.random.uniform(-cfg.grid_size//2 + cfg.dynamic_obstacle.obstacle_radius, cfg.grid_size//2 - cfg.dynamic_obstacle.obstacle_radius)
    y = np.random.uniform(-cfg.grid_size//2 + cfg.dynamic_obstacle.obstacle_radius, cfg.grid_size//2 - cfg.dynamic_obstacle.obstacle_radius)
    theta = np.random.uniform(-np.pi, np.pi)
    speed = np.random.uniform(0, cfg.dynamic_obstacle.obstacle_max_speed)
    obstacle = Circle(x, y, theta, speed, cfg.dynamic_obstacle.obstacle_radius)
    return [obstacle]

def follow_obstacle(cfg: DictConfig):
    random_x = np.random.uniform(-8, -5)
    random_y = np.random.uniform(-3, 3)
    
    theta = 0
    speed = np.random.uniform(0.5, 1.0)
    
    obstacle = Circle(random_x, random_y, theta, speed, radius=cfg.dynamic_obstacle.obstacle_radius)
    return [obstacle]

def overtake_obstacle(cfg: DictConfig):
    # x: (-4,4), y: (-8,8), vx: (0, 1), vy: 0
    # hardcoded for overtake 2 for now
    # overtake
    random_x = np.random.uniform(-7.5, -1.5) # hardcoded
    random_y = np.random.uniform(-3, 3)
    theta = 0 # simple
    speed = np.random.uniform(0.5, 1)
    # noise = np.random.normal(loc=cfg.dynamic_obstacle.noise_mean, 
    #                         scale=cfg.dynamic_obstacle.noise_std, 
    #                         size=cfg.dynamic_obstacle.noise_size)
    obstacle = Circle(random_x, random_y, theta, speed, radius=cfg.dynamic_obstacle.obstacle_radius)

    return [obstacle]
    
def pass_obstacle(cfg: DictConfig):
    # x: (-4,4), y: (-8,8), vx: (0, 1), vy: 0
    # hardcoded for overtake 2 for now
    # overtake
    random_x = np.random.uniform(0, 10) # hardcoded
    random_y = np.random.uniform(-3, 3)
    theta = 3.14 # simple
    speed = np.random.uniform(0.5, 1)
    # noise = np.random.normal(loc=cfg.dynamic_obstacle.noise_mean, 
    #                         scale=cfg.dynamic_obstacle.noise_std, 
    #                         size=cfg.dynamic_obstacle.noise_size)
    obstacle = Circle(random_x, random_y, theta, speed, radius=cfg.dynamic_obstacle.obstacle_radius)

    return [obstacle]

def yield_obstacle(cfg: DictConfig):
    random_x = np.random.uniform(-7.5, -2.5) # 3-7m
    random_y = np.random.uniform(5, 6)
    y_sign = np.random.choice([-1, 1])
    speed = np.random.uniform(0.5, 1)
    random_y = y_sign * random_y
    if y_sign < 0:
        theta = np.pi / 2
    else:
        theta = -np.pi / 2
    noise = np.random.normal(loc=cfg.dynamic_obstacle.noise_mean, 
                        scale=cfg.dynamic_obstacle.noise_std, 
                        size=cfg.dynamic_obstacle.noise_size)
    # obstacle = CircleNoisy(random_x, random_y, theta, speed, 
    #                     planning_dt=cfg.env_dt, noise=noise, radius=cfg.dynamic_obstacle.obstacle_radius)
    obstacle = Circle(random_x, random_y, theta, speed, radius=cfg.dynamic_obstacle.obstacle_radius)
    
    return [obstacle]

def yield_frontal_obstacle(cfg: DictConfig):
    random_x = np.random.uniform(0, 10) # 3-7m
    random_y = np.random.uniform(-3, 3)
    speed = np.random.uniform(0.5, 1)
    theta = np.pi
    noise = np.random.normal(loc=cfg.dynamic_obstacle.noise_mean, 
                        scale=cfg.dynamic_obstacle.noise_std, 
                        size=cfg.dynamic_obstacle.noise_size)
    # obstacle = CircleNoisy(random_x, random_y, theta, speed, 
    #                     planning_dt=cfg.env_dt, noise=noise, radius=cfg.dynamic_obstacle.obstacle_radius)
    obstacle = Circle(random_x, random_y, theta, speed, radius=cfg.dynamic_obstacle.obstacle_radius)
    
    return [obstacle]

def gen_obstacles(cfg: DictConfig, scenario: str):
    if scenario in ["pretrain", "static"]:
        obstacles = random_obstacle(cfg.env)
    elif scenario == "follow":
        obstacles = follow_obstacle(cfg.env)
    elif scenario == "overtake":
        obstacles = overtake_obstacle(cfg.env)
    elif scenario == "pass":
        obstacles = pass_obstacle(cfg.env)
    elif scenario == "yield":
        obstacles = yield_obstacle(cfg.env)
    elif scenario == "yield_frontal":
        obstacles = yield_frontal_obstacle(cfg.env)
    else:
        raise ValueError(f"Invalid scenario: {scenario}")
        
    return obstacles

def in_bounds(obstacle, grid_size, dt):
    x, y, _ = obstacle.pos_at_dt(dt)
    return x >= -grid_size//2 and x <= grid_size//2 and y >= -grid_size//2 and y <= grid_size//2

def gen_noncolliding_obstacles(cfg):
    start_x, start_y = cfg.objective.start_loc
    possible_values = np.arange(cfg.env.dynamic_obstacle.min_num_obstacles, cfg.env.dynamic_obstacle.max_num_obstacles + 1)
    if possible_values.sum() == 0:
        return []
    probabilities = (possible_values + 1) / (possible_values.sum() + len(possible_values))

    num_obstacles = np.random.choice(possible_values, p=probabilities)
    if num_obstacles == 0:
        return []
    obstacle_list = []
    for _ in range(1000): # num retry
        valid_scenario = True
        new_obstacles = gen_obstacles(cfg, scenario=cfg.scenarios.scenario)

        for new_obstacle in new_obstacles:
            if new_obstacle.contains(start_x, start_y, 0, 1):
                valid_scenario = False
                break
        if not valid_scenario:
            continue
        
        if cfg.env.gen_noncolliding:
            for obs in obstacle_list:
                for new_obstacle in new_obstacles:
                    for dt in range(cfg.env.gen_noncolliding_time):
                        if not in_bounds(obs, cfg.env.grid_size, dt) or not in_bounds(new_obstacle, cfg.env.grid_size, dt):
                            break
                        if new_obstacle.collide_with_other(obs, dt, buffer=1):
                            valid_scenario = False
                            break
                    if not valid_scenario:
                        break
                if not valid_scenario:
                    break
            
        if not valid_scenario:
            continue

        obstacle_list.extend(new_obstacles)
        
        if len(obstacle_list) == num_obstacles:
            return obstacle_list
        elif len(obstacle_list) > num_obstacles:
            raise ValueError(f"Too many obstacles: {len(obstacle_list)}")

    return []

def gen_one_terrain(cfg, start_loc):
    buffer = cfg.static_obstacle.buffer
    for i in range(1000):
        random_x = np.random.uniform(cfg.static_obstacle.min_x, cfg.static_obstacle.max_x)
        random_y = np.random.uniform(cfg.static_obstacle.min_y, cfg.static_obstacle.max_y)
        random_angle = np.random.uniform(cfg.static_obstacle.min_angle, cfg.static_obstacle.max_angle)
        random_width = np.random.uniform(cfg.static_obstacle.min_width, cfg.static_obstacle.max_width)
        random_height = np.random.uniform(cfg.static_obstacle.min_height, cfg.static_obstacle.max_height)
        # top left and right bottom corner are within range
        top_left = [random_x + random_height / 2, random_y + random_width / 2]
        bot_right = [random_x - random_height / 2, random_y - random_width / 2]
        if top_left[0] > cfg.grid_size / 2 + buffer or top_left[1] > cfg.grid_size / 2 + buffer:
            continue
        if bot_right[0] < -cfg.grid_size / 2 - buffer or bot_right[1] < -cfg.grid_size / 2 - buffer:
            continue
        rectangle = Rectangle(random_x, random_y, random_angle, 0, random_width, random_height)
        if not rectangle.contains(start_loc[0], start_loc[1], 0, buffer):
            return rectangle
    return None

def random_static(cfg):
    if cfg.env.static_obstacle.max_num_obstacles <= 0:
        return []
    # later need to make sure terrain is not blocking path
    possible_values = np.arange(cfg.env.static_obstacle.min_num_obstacles, cfg.env.static_obstacle.max_num_obstacles + 1)
    probabilities = [1 / len(possible_values)] * len(possible_values)
    num_terrains = np.random.choice(possible_values, p=probabilities)
    terrain_list = []
    start_loc = cfg.objective.start_loc
    for _ in range(num_terrains):
        rectangle = gen_one_terrain(cfg.env, start_loc)
        if rectangle is None:
            return []
        terrain_list.append(rectangle)
    return terrain_list

def ft_static(cfg):
    "doorway, corridor, intersection"
    # only with doorway for now
    top = np.random.uniform(-5, 5)
    bottom = top - 1
    # left wall
    left = 10
    right = np.random.uniform(-6, 6)
    rec_left = RectangleCorner(top, bottom, left, right)
    
    left = right - np.random.uniform(2, 5)
    right = -10
    rec_right = RectangleCorner(top, bottom, left, right)
    
    return [rec_left, rec_right]

def ft_prefer_easy(cfg):
    bottom1 = np.random.uniform(-9.5, -6.5)
    top1 = bottom1 + np.random.uniform(2.5, 5.5)
    
    bottom2 = np.random.uniform(-3.5, -0.5)
    top2 = bottom2 + np.random.uniform(2.5, 5.5)
    
    right1 = np.random.uniform(-1, 3)
    left1 = right1 + np.random.uniform(1, 2)
    
    left2 = np.random.uniform(-3, 1)
    right2 = left2 - np.random.uniform(1, 2)
    
    side1 = np.random.choice(["top", "bottom"])
    side2 = np.random.choice(["left", "right"])
    if side1 == "top" and side2 == "left":
        rec = RectangleCorner(top1, bottom1, left1, right1)
    if side1 == "top" and side2 == "right":
        rec = RectangleCorner(top1, bottom1, left2, right2)
    if side1 == "bottom" and side2 == "left":
        rec = RectangleCorner(top2, bottom2, left1, right1)
    if side1 == "bottom" and side2 == "right":
        rec = RectangleCorner(top2, bottom2, left2, right2)
    return [rec]      

def ft_prefer(cfg):
    bottom1 = np.random.uniform(-9.5, -6.5)
    top1 = bottom1 + np.random.uniform(2.5, 5.5)
    
    bottom2 = np.random.uniform(-3.5, -0.5)
    top2 = bottom2 + np.random.uniform(2.5, 5.5)
    
    right1 = np.random.uniform(1, 3)
    left1 = right1 + np.random.uniform(1, 2)
    
    left2 = np.random.uniform(-3, -1)
    right2 = left2 - np.random.uniform(1, 2)
    
    side1 = np.random.choice(["top", "bottom"])
    side2 = np.random.choice(["left", "right"])
    if side1 == "top" and side2 == "left":
        rec = RectangleCorner(top1, bottom1, left1, right1)
    if side1 == "top" and side2 == "right":
        rec = RectangleCorner(top1, bottom1, left2, right2)
    if side1 == "bottom" and side2 == "left":
        rec = RectangleCorner(top2, bottom2, left1, right1)
    if side1 == "bottom" and side2 == "right":
        rec = RectangleCorner(top2, bottom2, left2, right2)
    return [rec]        

def gen_regions(cfg):
    region_list = []
    if cfg.scenarios.scenario == "pretrain":
        region_list = random_static(cfg)
    elif cfg.scenarios.scenario == "static":
        region_list = ft_static(cfg)
    elif cfg.scenarios.scenario == "prefer":
        region_list = ft_prefer(cfg)
    
    return region_list
    

#################### Goal Generation ####################
def overtake_follow_goal(cfg: DictConfig, obstacle_list):
    assert len(obstacle_list) == 1
    obstacle = obstacle_list[0]
    x = 10
    multiplier = np.random.choice([0, 1])

    y = np.random.uniform(obstacle.y-2.25, obstacle.y+2.25) * multiplier
    if y < -3:
        y = -3
    if y > 3:
        y = 3
    return x, y

def overtake_follow_goal_legacy(cfg: DictConfig, obstacle_list):
    assert len(obstacle_list) == 1
    obstacle = obstacle_list[0]
    goal_x = 10
    goal_y = np.random.uniform(-1, 1)
    goal_y = 0 # simple
    # goal_y = np.random.uniform(obstacle.y - 2 if obstacle.y - 2 > -cfg.env.grid_size else -cfg.env.grid_size, 
    #                            obstacle.y + 2 if obstacle.y + 2 < cfg.env.grid_size else cfg.env.grid_size)
    
    # goal_y = np.random.uniform(obstacle.y - 1 if obstacle.y - 1 > -cfg.env.grid_size / 1 else -cfg.env.grid_size / 1, 
    #                            obstacle.y + 1 if obstacle.y + 1 < cfg.env.grid_size / 1 else cfg.env.grid_size / 1)
    return goal_x, goal_y

def random_goal(cfg: DictConfig, obstacle_list):
    for _ in range(2000):
        goal_x = np.random.uniform(-cfg.env.grid_size // 2, cfg.env.grid_size // 2)
        goal_y = np.random.uniform(-cfg.env.grid_size // 2, cfg.env.grid_size // 2)
        if all(not obstacle.contains(goal_x, goal_y, 0, 0.5) for obstacle in obstacle_list) and \
           np.linalg.norm(np.array([goal_x, goal_y]) - np.array(cfg.objective.start_loc)) > cfg.objective.goal_dist:
            return goal_x, goal_y
    return None, None

def overtake2_goal(cfg: DictConfig, obstacle_list):
    return overtake_follow_goal(cfg, obstacle_list)
    # hardcoded
    top_obstacle, bottom_obstacle = obstacle_list
    max_y = max(top_obstacle.y, bottom_obstacle.y)
    min_y = min(top_obstacle.y, bottom_obstacle.y)
    goal_y = np.random.uniform(min_y, max_y+1e-4)
    return 15, goal_y

def follow_goal(cfg: DictConfig, obstacle_list):
    # Changed
    assert len(obstacle_list) == 1
    obstacle = obstacle_list[0]
    goal_x = 10
    goal_y = np.random.uniform(obstacle.y - 1 if obstacle.y - 1 > -cfg.env.grid_size / 1 else -cfg.env.grid_size / 1, 
                               obstacle.y + 1 if obstacle.y + 1 < cfg.env.grid_size / 1 else cfg.env.grid_size / 1)
    return goal_x, goal_y

def terrain_goal(cfg: DictConfig, obstacle_list, terrain_list):
    goal_x = 10
    goal_y = np.random.uniform(-1, 1)
    # goal_y = np.random.uniform(obstacle.y - 2 if obstacle.y - 2 > -cfg.env.grid_size else -cfg.env.grid_size, 
    #                            obstacle.y + 2 if obstacle.y + 2 < cfg.env.grid_size else cfg.env.grid_size)
    
    # goal_y = np.random.uniform(obstacle.y - 1 if obstacle.y - 1 > -cfg.env.grid_size / 1 else -cfg.env.grid_size / 1, 
    #                            obstacle.y + 1 if obstacle.y + 1 < cfg.env.grid_size / 1 else cfg.env.grid_size / 1)
    return goal_x, goal_y
                   
    return None, None

def yield_goal(cfg: DictConfig, obstacle_list):
    assert len(obstacle_list) == 1
    obstacle = obstacle_list[0]
    x = 10
    y = np.random.uniform(-3, 3)
    # x = 9.5
    # if obstacle.theta > 0:
    #     y = np.random.uniform(2, cfg.env.grid_size * 4 / 10)
    # else:
    #     y = np.random.uniform(-cfg.env.grid_size * 4 / 10, -2)
        
    return x, y

def yield_frontal_goal(cfg: DictConfig, obstacle_list, terrain_list):
    assert len(obstacle_list) == 1
    obstacle = obstacle_list[0]
    if obstacle.y < 0:
        x = 10
        y = obstacle.y - np.random.uniform(1, 3)
    else:
        x = 10
        y = obstacle.y + np.random.uniform(1, 3)
    return x, y

def prefer_goal(cfg: DictConfig, obstacle_list, terrain_list):
    x = 10 
    y = 0
    return x, y

def gen_goal(cfg, obstacle_list, terrain_list, scenario):
    if scenario == "pretrain":
        avoidance_list = obstacle_list + terrain_list
        return random_goal(cfg, avoidance_list)
    elif scenario == "follow":
        return follow_goal(cfg, obstacle_list)
    elif scenario == "overtake" or scenario == "pass":
        return overtake_follow_goal(cfg, obstacle_list)
    elif scenario == "overtake2":
        return overtake2_goal(cfg, obstacle_list)
    elif scenario == "static": 
        return terrain_goal(cfg, obstacle_list, terrain_list)
    elif scenario == "yield":
        return yield_goal(cfg, obstacle_list)
    elif scenario == "prefer":
        return prefer_goal(cfg, obstacle_list, terrain_list)    
    elif scenario == "yield_frontal":
        return yield_frontal_goal(cfg, obstacle_list, terrain_list)
    else:
        raise ValueError(f"Invalid scenario: {scenario}")
    
#################### Plan Generation with RRT subgoals + Hybrid Astar ####################
def gen_rrt_subgoals(cfg: DictConfig, start_loc, goal_loc, obstacles_list) -> list:
    num_subgoals = np.random.randint(cfg.generation.min_subgoals, cfg.generation.max_subgoals + 1)
    rrt_planner = RRTPlanner(cfg, obstacles_list, cfg.env.grid_size)
        
    rrt_start_node = RRTNode(start_loc[0], start_loc[1], 0, 0, 0, None)
    rrt_goal_node = RRTNode(goal_loc[0], goal_loc[1], 0, 0, 0, 0)
    
    max_iter = 10000
    rrt_plans = rrt_planner.plan(rrt_start_node, rrt_goal_node, max_iter, cfg.objective.goal_radius)
    rrt_plan = None
    if len(rrt_plans) == 0:
        return None
    else:
        for plan in rrt_plans:
            plan.reverse()
        rrt_plan = rrt_plans[np.random.randint(len(rrt_plans))] # randomly select one from top k plans
    
    step = len(rrt_plan) // (num_subgoals + 1)
    subgoal_indices = [i * step for i in range(1, num_subgoals)] # evenly spaced subgoals
    subgoal_nodes = []
    for subgoal_idx in subgoal_indices:
        subgoal_node = Node(rrt_plan[subgoal_idx][0], rrt_plan[subgoal_idx][1], 0, 0, 0, 0)
        subgoal_nodes.append(subgoal_node)
    
    subgoal_nodes.append(Node(goal_loc[0], goal_loc[1], 0, 0, 0, 0)) # include the goal node as the last subgoal
    return subgoal_nodes, rrt_plan

def gen_hybrid_astar_path(cfg, planner: AStarPlanner, start_node, goal_node, obstacles_list):
    dt = cfg.env.env_dt
    planned_result = planner.plan(start_node, goal_node, 
                                max_planning_time=cfg.robot.max_planning_time, 
                                goal_radius=cfg.objective.goal_radius, 
                                max_expand_node_num=cfg.generation.max_expand_node_num)
    
    # Plan format is list of (curr_node.x, curr_node.y, curr_node.theta, curr_v, curr_w, curr_node.t_idx)
    if planned_result is not None:
            path, target_max_velocity = planned_result
            if validate_plan(path, obstacles_list, dt, start_node.t_idx) == False:
                # 11/20/2024: this can have a bug but can't figure out what causes it for now
                pass
                # assert len(path) > cfg.robot.max_planning_time + 1 - start_node.t_idx, f"Plan too short: {len(path)}"
            else:
                path = np.array(path)
                return path, target_max_velocity
    return None

def gen_plans_rrt_subgoal(cfg: DictConfig, start_loc, goal_loc, obstacles_list, terrain_list) -> list:    
    outputs = {
        "obstacles_info": [], 
        "terrain_info": [],
        "paths": []
    }
    dt = cfg.env.env_dt
    grid_size = cfg.env.grid_size
    goal_radius = cfg.objective.goal_radius
    grid_resolution = cfg.env.grid_resolution
    angle_resolution = cfg.env.angle_resolution

    # randomly decide whether to use terrain or not
    avoidance_list = obstacles_list + terrain_list
    astar_planner = AStarPlanner(obstacles_list=avoidance_list, grid_size=grid_size, 
                grid_resolution=grid_resolution, angle_resolution=angle_resolution, env_dt=dt, 
                **cfg.robot)
        
    for obstacle in obstacles_list:
        outputs["obstacles_info"].append(obstacle.to_dict())
    
    for terrain in terrain_list:
        outputs["terrain_info"].append(terrain.to_dict())

    for _ in range(cfg.generation.num_plans):
        combined_plan = []

        rrt_result = gen_rrt_subgoals(cfg, start_loc, goal_loc, avoidance_list)
        if rrt_result is None:
            continue 
        
        subgoal_nodes, rrt_plan = rrt_result
        
        valid_plan = True
        start_astar_node = Node(start_loc[0], start_loc[1], 0, 0, 0, 0)
        target_max_velocities = []
        for subgoal_node in subgoal_nodes:
            planned_result = gen_hybrid_astar_path(cfg, astar_planner, start_astar_node, subgoal_node, avoidance_list)
            if planned_result is None:
                valid_plan = False
                break
            tmp_path, target_max_velocity = planned_result
            target_max_velocities.append(target_max_velocity)
            sx, sy, stheta, sv, sw, st_idx = tmp_path[-1]
            start_astar_node = Node(sx, sy, stheta, st_idx, sv, sw)
            tmp_path = np.array(tmp_path)[:,:5]
            combined_plan.extend(tmp_path[:-1])
  
        if not valid_plan or len(combined_plan) == 0 or not validate_plan(combined_plan, avoidance_list, dt):
            continue

        combined_plan.append(tmp_path[-1]) # add the last node
        combined_plan = np.vstack(combined_plan).tolist()
          
        subgoals = [[subgoal_node.x, subgoal_node.y] for subgoal_node in subgoal_nodes]
        tmp_result = {
            "start": start_loc, 
            "goal": goal_loc, 
            "path": combined_plan,
            "info": {
                "subgoals": subgoals,
                "target_max_velocities": target_max_velocities,
                "rrt_plan": rrt_plan,
            }
        }
        outputs["paths"].append(tmp_result)
        ### DEBUG ###
        # print(start_loc, goal_loc, dt, grid_size, goal_radius, subgoals)
        # visualize_rrt_astar([combined_plan], start_loc, goal_loc, avoidance_list, dt, 
        #                     grid_size, goal_radius, subgoals=subgoals, save_name="rrt_astar")
        # input()
    return outputs 
