import numpy as np
from composablenav.misc.common import find_first_waypoint_within_radius, validate_plan

# General success criteria
def collision_criteria(path, goal, goal_radius, obstacles, dt, start_time_idx=0, **kwargs) -> bool:
    first_waypoint_idx = find_first_waypoint_within_radius(path, goal, goal_radius)
    if first_waypoint_idx != -1:
        path = path[:first_waypoint_idx+1]
    valid_plan = validate_plan(path, obstacles, dt, start_time_idx)
    return valid_plan

def goal_reaching_criteria(path, goal, goal_radius, **kwargs) -> bool:
    first_waypoint_idx = find_first_waypoint_within_radius(path, goal, goal_radius)
    return first_waypoint_idx != -1

def follow_n_meters_boolean_criteria(path, obstacles, dt, 
                              follow_width: float, follow_min_dist: float, follow_max_dist: float, goal,
                              n_meters: float, **kwargs) -> bool:
    last_follow_x, last_target_x = follow_criteria(path, obstacles, dt, 
                                   follow_width=follow_width, follow_min_dist=follow_min_dist, goal=goal,
                                   follow_max_dist=follow_max_dist) 
    return bool(last_target_x - last_follow_x >= n_meters)

def pass_boolean_criteria(path, obstacles, dt, overtake_direction: str, **kwargs) -> bool:
    direction = overtake_direction_criteria(path, obstacles, dt)
    if direction != overtake_direction:
        return False
    return True

def yield_boolean_criteria(path, obstacles: list, dt: float, obstacle_width: float, 
                    obstacle_front_min_dist: float, obstacle_front_max_dist: float,  **kwargs) -> bool:
    return yield_criteria(path, obstacles, dt, obstacle_width, 
                          obstacle_front_min_dist, obstacle_front_max_dist) == 1

def yield_frontal_boolean_criteria(path, obstacles: list, dt: float, obstacle_width: float, 
                    obstacle_front_min_dist: float, obstacle_front_max_dist: float,  **kwargs) -> bool:
    return yield_frontal_criteria(path, obstacles, dt, obstacle_width, 
                          obstacle_front_min_dist, obstacle_front_max_dist) == 1

def walk_over_region_percent(path, obstacles, dt, **kwargs):
    assert len(obstacles) == 1
    obstacle = obstacles[0]
    height = obstacle.top - obstacle.bottom
    covered_pos = []
    curr_pos = None 
    for idx, step in enumerate(path):
        t = idx * dt
        x, y = step[0], step[1]
        inside_obstacle = obstacle.dist_to_obstacle(x,y,t) == 0
        if inside_obstacle:
            if curr_pos is None:
                covered_pos.append([x, x])
            else:
                if x > covered_pos[-1][1]:
                    covered_pos[-1][1] = x
            curr_pos = True
        else:
            curr_pos = None
    covered_length = 0
    for start, end in covered_pos:
        covered_length += end - start   
    return covered_length / height

def walk_over_region_percent_boolean_criteria(path, obstacles, dt, percentage, **kwargs):
    return bool(walk_over_region_percent(path, obstacles, dt) >= percentage)

def avoid_region_boolean_criteria(path, obstacles, dt, **kwargs):
    x, y = path[-1][0], path[-1][1]
    for idx, step in enumerate(path):
        x, y = step[0], step[1]
        curr_t = idx * dt
        for terrain in obstacles:
            if terrain.dist_to_obstacle(x,y,curr_t) == 0:
                return False
    return True

def yield_criteria(path, obstacles: list, dt: float, obstacle_width: float, 
                    obstacle_front_min_dist: float, obstacle_front_max_dist: float,  **kwargs):
    assert len(obstacles) == 1
    obstacle = obstacles[0]

    obs_start_x, obs_start_y, _ = obstacle.pos_at_dt(0)
    start_polygon = compute_behind_area([obs_start_x, obs_start_y], obstacle.theta, 
                                  obstacle_width, obstacle_front_min_dist, obstacle_front_max_dist)
    
    has_passed_yielding_region = False # Assume horizontal yielding
    # compute point in front
    for idx, step in enumerate(path):
        t = idx * dt
        x, y = step[0], step[1]
        obs_x, obs_y, obs_yaw = obstacle.pos_at_dt(t)

        # don't cut in front of the obstacle
        polygon = compute_behind_area([obs_x, obs_y], -obstacle.theta, # in front
                                      obstacle_width, obstacle_front_min_dist, obstacle_front_max_dist)

        is_cutting = point_in_polygon((x, y), polygon)
        if is_cutting:
            return -1
            
        # don't go behind the obstacle
        overtake_from_behind = point_in_polygon((x, y), start_polygon)
        if overtake_from_behind:
            return 0
        
        if x > obs_x + obstacle_width:
            has_passed_yielding_region = True
            break
    if has_passed_yielding_region:
        return 1
    else:
        return 0
    
def yield_frontal_criteria(path, obstacles: list, dt: float, obstacle_width: float, 
                    obstacle_front_min_dist: float, obstacle_front_max_dist: float,  **kwargs):
    assert len(obstacles) == 1
    obstacle = obstacles[0]

    # compute point in front
    for idx, step in enumerate(path):
        t = idx * dt
        x, y = step[0], step[1]
        obs_x, obs_y, obs_yaw = obstacle.pos_at_dt(t)
        if x < obs_x:
            if obs_y < 0 and y < obs_y:
                return -1
            if obs_y > 0 and y > obs_y:
                return -1      
    return 1
    
def terrain_criteria(path, obstacles, dt, do_avoid, **kwargs):
    for idx in range(1, len(path)):
        prev_x, prev_y = path[idx-1]
        x, y = path[idx]
        for terrain in obstacles:
            if terrain.collision(prev_x, prev_y, x, y, dt, buffer=0):
                if do_avoid:
                    return -1
    if do_avoid:
        return 1
    else:
        raise NotImplementedError
      

## -------------------------------------------------------------- ##
## ----------------- Critic Functions: Overtake ----------------- ##
## -------------------------------------------------------------- ##
def overtake_consecutive_criteria(path, obstacles: list, dt: float, n_consecutive_overtaken: int):
    assert len(obstacles) == 1
    assert len(path) > 0
    obstacle = obstacles[0]
    current_overtake_count = 0  # counts the current streak of consecutive overtakes
    max_consecutive_overtake_count = 0  # tracks the longest consecutive overtake streak
    extended_path = path + [path[-1]] * n_consecutive_overtaken  # extend path to check full sequence
    
    for idx, step in enumerate(extended_path):
        t = idx * dt
        obs_x, _, _ = obstacle.pos_at_dt(t)
        x = step[0]
        
        if x > obs_x: 
            current_overtake_count += 1
            max_consecutive_overtake_count = max(max_consecutive_overtake_count, current_overtake_count)
        else:
            current_overtake_count = 0  # reset streak if the path is no longer ahead

    return max_consecutive_overtake_count


def overtake_direction_criteria(path, obstacles: list, dt: float):
    assert len(obstacles) == 1
    assert len(path) > 0
    obstacle = obstacles[0]
    direction = None
    for idx, step in enumerate(path):
        t = idx * dt
        obs_x, obs_y, _ = obstacle.pos_at_dt(t)
        x, y = step[0], step[1]
        if direction is None:
            # defined as centerline
            if x > obs_x and y > obs_y:
                direction = "left"
                break
            elif x > obs_x and y < obs_y:
                direction = "right"
                break
                
    return direction 

def stay_in_centerline_criteria(path, obstacles: list, dt: float, max_dist=3):
    assert len(obstacles) == 1
    assert len(path) > 0
    obstacle = obstacles[0]
    cost = 0
    for idx, step in enumerate(path):
        t = idx * dt
        obs_x, obs_y, obs_yaw = obstacle.pos_at_dt(t)
        y = step[1]
        distance = min(np.abs(y - obs_y), max_dist) / max_dist
        cost += distance
    return (-cost / len(path)) / 2

def overtake_reward(path, obstacles, goal, dt: float, goal_radius:float, overtake_direction: str):
    # for PPO, return 1 if path is better, 0 otherwise
    assert overtake_direction in ["left", "right"]
    no_collision = collision_criteria(path, goal, goal_radius, obstacles, dt)
    if not no_collision:
        return -1
    direction = overtake_direction_criteria(path, obstacles, dt)
    cost = stay_in_centerline_criteria(path, obstacles, dt)
    # stay in the center line of the obstacle
    if direction == overtake_direction:
        return 1 + cost
    return 0 + cost

def overtake_reward_random_start(path, obstacles, goal, dt: float, goal_radius:float, overtake_direction: str):
    # for PPO, return 1 if path is better, 0 otherwise
    assert overtake_direction in ["left", "right"]
    no_collision = collision_criteria(path, goal, goal_radius, obstacles, dt)
    if not no_collision:
        return -1
    direction = overtake_direction_criteria(path, obstacles, dt)
    cost = stay_in_centerline_criteria(path, obstacles, dt)
    cost = 0 # V3: remove cost
    # stay in the center line of the obstacle
    if False:
        valid_offset = 0 # V1-3
    elif True:
        valid_offset = 1 # V4
    # temp random start location: maybe removed later
    obs_x, obs_y, _ = obstacles[0].pos_at_dt(0)
    robot_x, robot_y = path[0][0], path[0][1]
    if robot_x > obs_x - valid_offset:
        return 1 + cost
    # temp random start location: maybe removed later
    if direction == overtake_direction:
        return 1 + cost
    return 0 + cost  
## -------------------------------------------------------------- ##
## ------------------ Critic Functions: Follow ------------------ ##
## -------------------------------------------------------------- ##
def compute_behind_area(circle_center, theta, width, min_length, max_length):
    x, y = circle_center
    # Unit vector in the direction of theta
    dx = np.cos(theta)
    dy = np.sin(theta)

    p1 = (x - width * dy, y + width * dx)
    p2 = (x + width * dy, y - width * dx)
    
    # Compute the two points on the back of the circle for the min_length
    p1_min = (p1[0] - min_length * dx, p1[1] - min_length * dy)
    p2_min = (p2[0] - min_length * dx, p2[1] - min_length * dy)
    
    p1_max = (p1[0] - max_length * dx, p1[1] - max_length * dy)
    p2_max = (p2[0] - max_length * dx, p2[1] - max_length * dy)
    
    return [p1_min, p2_min, p2_max, p1_max]

def point_in_polygon(point, polygon: list):
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside

def follow_criteria(path, obstacles: list, dt: float, follow_width: float, 
                    follow_min_dist: float, follow_max_dist: float, goal: float):
    # REDO: has to follow n meters in the end
    assert len(obstacles) == 1
    assert len(path) > 0
    obstacle = obstacles[0]
    # total number of timestep that the path fall within a distance of follow_dist from the obstacle
    consecutive_unfollowing = 0
    last_target_x = None
    last_follow_x = None
    # consecutive steps where path is within follow_dist of obstacle
    for idx, step in enumerate(reversed(path)):
        t = (len(path) - 1 - idx) * dt
        x, y = step[0], step[1]
        obs_x, obs_y, obs_yaw = obstacle.pos_at_dt(t)
        if last_target_x is None:
            last_target_x = obs_x
            last_follow_x = obs_x # a decoy value if not following, then it will be the last follow x
        if obs_x > goal[0]:
            continue
        polygon = compute_behind_area([obs_x, obs_y], obstacle.theta, follow_width, follow_min_dist, follow_max_dist)
        is_following = point_in_polygon((x, y), polygon)
        
        if not is_following:
            consecutive_unfollowing += 1
        else:
            last_follow_x = x
            consecutive_unfollowing = 0
        
        if consecutive_unfollowing > 1:
            break

    return last_follow_x, last_target_x

def follow_criteria_legacy(path, obstacles: list, dt: float, follow_width: float, 
                    follow_min_dist: float, follow_max_dist: float, calc_consecutive=False):
    assert len(obstacles) == 1
    assert len(path) > 0
    obstacle = obstacles[0]
    # total number of timestep that the path fall within a distance of follow_dist from the obstacle
    follow_steps = 0
    max_follow_steps = 0
    current_follow_steps = 0
        
    # consecutive steps where path is within follow_dist of obstacle
    for idx, step in enumerate(path):
        t = idx * dt
        x, y = step[0], step[1]
        obs_x, obs_y, obs_yaw = obstacle.pos_at_dt(t)
        polygon1 = compute_behind_area([obs_x, obs_y], obstacle.theta, follow_width, follow_min_dist, follow_max_dist)
        is_following1 = point_in_polygon((x, y), polygon1)
        
        if is_following1:
            follow_steps += 1
            current_follow_steps += 1
            max_follow_steps = max(max_follow_steps, current_follow_steps)
        else:
            current_follow_steps = 0
    
    if calc_consecutive:
        return max_follow_steps
    return follow_steps

def follow_criteria_for_training(path, obstacles: list, dt: float, follow_width: float, 
                    follow_min_dist: float, follow_max_dist: float, calc_consecutive=False):
    assert len(obstacles) == 1
    assert len(path) > 0
    obstacle = obstacles[0]
    # total number of timestep that the path fall within a distance of follow_dist from the obstacle
    follow_steps = 0
    max_follow_steps = 0
    current_follow_steps = 0
    follow_dist = 1
        
    # consecutive steps where path is within follow_dist of obstacle
    rewards = 0
    radius = 5 # anything within 5 meters will have a reward
    for idx, step in enumerate(path):
        t = idx * dt
        x, y = step[0], step[1]
        obs_x, obs_y, obs_yaw = obstacle.pos_at_dt(t)
        # if abs(y - obs_y) > 0.5 or x > obs_x - 0.5:
        #     continue
        distance = np.sqrt((x - (obs_x - follow_dist))**2 + (y - obs_y)**2)
        rewards += max(0, radius - distance)
    
    # rewards /= len(path)
    return rewards

def follow_criteria_for_training_old(path, obstacles: list, dt: float, follow_width: float, 
                    follow_min_dist: float, follow_max_dist: float, calc_consecutive=False):
    assert len(obstacles) == 1
    assert len(path) > 0
    obstacle = obstacles[0]
    # total number of timestep that the path fall within a distance of follow_dist from the obstacle
    follow_steps = 0
    max_follow_steps = 0
    current_follow_steps = 0
        
    # consecutive steps where path is within follow_dist of obstacle
    for idx, step in enumerate(path):
        t = idx * dt
        x, y = step[0], step[1]
        obs_x, obs_y, obs_yaw = obstacle.pos_at_dt(t)
        # Temporary
        polygon1 = compute_behind_area([obs_x, obs_y], obstacle.theta, follow_width, follow_min_dist, follow_min_dist+1)
        polygon2 = compute_behind_area([obs_x, obs_y], obstacle.theta, follow_width, follow_min_dist, follow_min_dist+2)
        polygon3 = compute_behind_area([obs_x, obs_y], obstacle.theta, follow_width, follow_min_dist, follow_max_dist+3)
        polygon4 = compute_behind_area([obs_x, obs_y], obstacle.theta, follow_width, follow_min_dist, follow_max_dist+4)
        polygon5 = compute_behind_area([obs_x, obs_y], obstacle.theta, follow_width, follow_min_dist, follow_max_dist+5)
        is_following1 = point_in_polygon((x, y), polygon1)
        is_following2 = point_in_polygon((x, y), polygon2)
        is_following3 = point_in_polygon((x, y), polygon3)
        is_following4 = point_in_polygon((x, y), polygon4)
        is_following5 = point_in_polygon((x, y), polygon5)
        
        if is_following1:
            follow_steps += 1
            current_follow_steps += 1
            max_follow_steps = max(max_follow_steps, current_follow_steps)
        elif is_following2:
            follow_steps += 0.5
            current_follow_steps += 0.5
            max_follow_steps = max(max_follow_steps, current_follow_steps)
        elif is_following3:
            follow_steps += 0.2
            current_follow_steps += 0.2
            max_follow_steps = max(max_follow_steps, current_follow_steps)
        elif is_following4:
            follow_steps += 0.1
            current_follow_steps += 0.1
            max_follow_steps = max(max_follow_steps, current_follow_steps)
        elif is_following5:
            follow_steps += 0.05
            current_follow_steps += 0.05
            max_follow_steps = max(max_follow_steps, current_follow_steps)
        else:
            current_follow_steps = 0
    
    if calc_consecutive:
        return max_follow_steps
    return follow_steps

def follow_reward(path, obstacles, goal, dt: float, goal_radius:float, 
                  follow_width: float, follow_min_dist: float, follow_max_dist: float):
    # truncate the path when goal is reached
    first_waypoint_idx = find_first_waypoint_within_radius(path, goal, goal_radius)
    if first_waypoint_idx != -1:
        path = path[:first_waypoint_idx+1]
    no_collision = collision_criteria(path, goal, goal_radius, obstacles, dt)
    if not no_collision:
        return -1
    
    # for PPO, use number of follow steps as rewards
    follow_steps = follow_criteria_for_training_old(path, obstacles, dt, follow_width, follow_min_dist, follow_max_dist)
    # follow_steps = follow_criteria_legacy(path, obstacles, dt, follow_width, follow_min_dist, follow_max_dist)
    return follow_steps


## -------------------------------------------------------------- ##
## ------------------ Critic Functions: Yield ------------------ ##
## -------------------------------------------------------------- ## 
def yield_frontal_reward(path, obstacles, goal, dt: float, goal_radius:float,
                obstacle_width: float, obstacle_front_min_dist: float, obstacle_front_max_dist: float):
    # truncate the path when goal is reached
    first_waypoint_idx = find_first_waypoint_within_radius(path, goal, goal_radius)
    if first_waypoint_idx != -1:
        path = path[:first_waypoint_idx+1]
    no_collision = collision_criteria(path, goal, goal_radius, obstacles, dt)
    reached_goal = goal_reaching_criteria(path, goal, goal_radius)
    if not no_collision:
        return -1
    
    has_yield = yield_frontal_criteria(path, obstacles, dt, obstacle_width, 
                                obstacle_front_min_dist, obstacle_front_max_dist)
    if has_yield == -1:
        return -1
    if reached_goal:
        return 1 + has_yield
    return has_yield

def yield_reward(path, obstacles, goal, dt: float, goal_radius:float,
                obstacle_width: float, obstacle_front_min_dist: float, obstacle_front_max_dist: float):
    # truncate the path when goal is reached
    first_waypoint_idx = find_first_waypoint_within_radius(path, goal, goal_radius)
    if first_waypoint_idx != -1:
        path = path[:first_waypoint_idx+1]
    no_collision = collision_criteria(path, goal, goal_radius, obstacles, dt)
    reached_goal = goal_reaching_criteria(path, goal, goal_radius)
    if not no_collision:
        return -1
    
    has_yield = yield_criteria(path, obstacles, dt, obstacle_width, 
                                obstacle_front_min_dist, obstacle_front_max_dist)
    if has_yield == -1:
        return -1
    if reached_goal:
        return 1 + has_yield
    return has_yield

## -------------------------------------------------------------- ##
## ------------------ Critic Functions: terrain ------------------ ##
## -------------------------------------------------------------- ##
def terrain_reward(path, obstacles, goal, dt: float, goal_radius:float, do_avoid):
    first_waypoint_idx = find_first_waypoint_within_radius(path, goal, goal_radius)
    if first_waypoint_idx != -1:
        path = path[:first_waypoint_idx+1]
    reached_goal = goal_reaching_criteria(path, goal, goal_radius)
    
    terrain_reward = terrain_criteria(path, obstacles, dt=dt, do_avoid=do_avoid) # do avoid is temporary
    if terrain_reward == -1:
        return -1 
    
    return terrain_reward + reached_goal

def avoid_region_reward(path, obstacles, goal, dt: float, goal_radius:float):
    first_waypoint_idx = find_first_waypoint_within_radius(path, goal, goal_radius)
    if first_waypoint_idx != -1:
        path = path[:first_waypoint_idx+1]
    reached_goal = goal_reaching_criteria(path, goal, goal_radius)
    no_collision = collision_criteria(path, goal, goal_radius, obstacles, dt)
    if not no_collision:
        return -1 
    if not reached_goal:
        return -1 
    return 1

def prefer_region_reward(path, obstacles, goal, dt: float, goal_radius:float):
    first_waypoint_idx = find_first_waypoint_within_radius(path, goal, goal_radius)
    if first_waypoint_idx != -1:
        path = path[:first_waypoint_idx+1]
    reached_goal = goal_reaching_criteria(path, goal, goal_radius)
    
    if not reached_goal:
        return 0
    prefer_reward = walk_over_region_percent(path, obstacles, dt)
    
    return prefer_reward
    

## ---------------------------------------------------- ##
## ----------------- Compare Functions ---------------- ##
## ---------------------------------------------------- ##
# return 1 if path1 better, 1 if path2 better, 0 if equal
def overtake_eval(path1, path2, obstacle, overtake_direction: str):
    # implementation only supports obstacle moving in +X direction
    # can adjust coordinates to be in obstacle frame for other directions
    assert overtake_direction in ["left", "right"]

    has_overtaken1, direction1 = overtake_criteria(path1, [obstacle])
    has_overtaken2, direction2 = overtake_criteria(path2, [obstacle])
    if direction1 == direction2:
        if has_overtaken1 != has_overtaken2:
            return 1 if has_overtaken1 else 2
        return 0
    else:
        if direction1 == overtake_direction:
            return 1
        elif direction2 == overtake_direction:
            return 2
        return 0

def follow_eval(path1, path2, obstacle, follow_width, follow_dist):
    steps1 = follow_criteria(path1, [obstacle], follow_width, follow_dist - 2, follow_dist + 4)
    steps2 = follow_criteria(path2, [obstacle], follow_width, follow_dist - 2, follow_dist + 4)
    return 1 if steps1 > steps2 else 2 if steps2 > steps1 else 0

def terrain_eval(path1, path2, terrain_region):
    # path1 and path2 needs to be truncated when reaching goal
    path1_count, path2_count = 0, 0
    for i, step in enumerate(path1):
        if terrain_region.contains(step[0], step[1], i, 0):
            path1_count += 1
    for i, step in enumerate(path2):
        if terrain_region.contains(step[0], step[1], i, 0):
            path2_count += 1
    if path1_count > path2_count:
        return 1
    elif path2_count > path1_count:
        return 2
    return 0

def yield_eval(path1, path2, obstacle, yield_dist):
    path1_pass_front = False
    path2_pass_front = False
    obs_vy = obstacle.speed * np.sin(obstacle.theta)
    for step in path1:
        x, y = step[0], step[1]
        if abs(x - obstacle.x) < obstacle.radius and ((y > obstacle.y and obs_vy > 0) or (y < obstacle.y and obs_vy < 0)):
            path1_pass_front = True
            break
    for step in path2:
        x, y = step[0], step[1]
        if abs(x - obstacle.x) < obstacle.radius and ((y > obstacle.y and obs_vy > 0) or (y < obstacle.y and obs_vy < 0)):
            path2_pass_front = True
            break
    
    if path1_pass_front and not path2_pass_front:
        return 2
    if path2_pass_front and not path1_pass_front:
        return 1
    
    path1_yield = False
    path2_yield = False
    
    if obs_vy > 0:
        for step in path1:
            x, y = step[0], step[1]
            if y >= obstacle.y + obstacle.radius and obstacle.x - x <= yield_dist:
                path1_yield = True
                break
        for step in path2:
            x, y = step[0], step[1]
            if y >= obstacle.y + obstacle.radius and obstacle.x - x <= yield_dist:
                path2_yield = True
                break
    elif obs_vy < 0:
        for step in path1:
            x, y = step[0], step[1]
            if y <= obstacle.y - obstacle.radius and obstacle.x - x <= yield_dist:
                path1_yield = True
                break
        for step in path2:
            x, y = step[0], step[1]
            if y <= obstacle.y - obstacle.radius and obstacle.x - x <= yield_dist:
                path2_yield = True
                break
    
    if path1_yield and not path2_yield:
        return 1
    if path2_yield and not path1_yield:
        return 2
    return 0
    
