import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from composablenav.datasets.obstacles import Obstacle, Circle, Rectangle
import copy

class RRTNode:
    def __init__(self, x, y, t, vx, vy, parent=None):
        self.x = x
        self.y = y
        self.t = t
        self.vx = vx
        self.vy = vy
        self.parent = parent


class RRTPlanner:
    def __init__(self, cfg, obstacle_list, grid_size):
        self.cfg = cfg
        self.grid_size = grid_size
        self.obstacle_list = copy.deepcopy(obstacle_list)
        self.node_list = []
        self.paths = []

    def distance(self, node1, node2):
        return np.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)

    def get_nearest_node_index(self, random_x, random_y):
        distances = [np.sqrt((node.x - random_x)**2 + (node.y - random_y)**2) for node in self.node_list]
        min_index = distances.index(min(distances))
        return min_index

    def steer(self, closest_node, random_x, random_y, max_length):
        theta = np.arctan2(random_y - closest_node.y, random_x - closest_node.x)
        dist = min(max_length, np.sqrt((closest_node.x - random_x)**2 + (closest_node.y - random_y)**2))
        new_node = RRTNode(closest_node.x + dist * np.cos(theta), closest_node.y + dist * np.sin(theta), closest_node.t + 1, 0, 0)
        return new_node
    
    def collision(self, node_1):
        for obstacle in self.obstacle_list:
            if obstacle.contains(node_1.x, node_1.y, node_1.t, 0.5):
                return True
        return False

    def plan(self, start, goal, max_iter, goal_radius):
        self.node_list = []
        self.start = start
        self.goal = goal
        self.node_list.append(start)

        i = 0
        while i < max_iter:
            random_x, random_y = np.random.uniform(-self.grid_size//2, self.grid_size//2), np.random.uniform(-self.grid_size//2, self.grid_size//2)
            nearest_index = self.get_nearest_node_index(random_x, random_y)
            nearest_node = self.node_list[nearest_index]
            new_node = self.steer(nearest_node, random_x, random_y, max_length=1.0)
            new_node.parent = nearest_node
            
            if not self.collision(new_node):
                self.node_list.append(new_node)

                if self.distance(new_node, self.goal) <= goal_radius:
                    self.paths.append(self.generate_final_course(len(self.node_list) - 1))
                    if len(self.paths) >= self.cfg.generation.rrt_paths:
                        return self.paths
            i += 1

        return self.paths
    def generate_final_course(self, goal_index):
        path = [[self.goal.x, self.goal.y]]
        node = self.node_list[goal_index]
        while node.parent is not None:
            path.append([node.x, node.y]) 
            node = node.parent
        path.append([node.x, node.y])
        
        
        
        return path

    def draw_graph(self, obstacle_list, goal_radius, path, robot_loc=None):
        plt.clf()
                
        for node in self.node_list:
            if node.parent:
                plt.plot([node.y, node.parent.y], [node.x, node.parent.x], "-k")

        for obstacle in obstacle_list:
            obstacle.draw()
                    
        if path:
            for i in range(len(path) - 1):
                plt.plot([path[i][1], path[i+1][1]], [path[i][0], path[i+1][0]], "-m") 

        if robot_loc:
            plt.plot(robot_loc[1], robot_loc[0], "xb")
        else:
            plt.plot(self.start.y, self.start.x, "xb")
        plt.plot(self.goal.y, self.goal.x, "xr")
        goal_circle = patches.Circle((self.goal.y, self.goal.x), radius=goal_radius, color='r', fill=False)
        plt.gca().add_patch(goal_circle)
        plt.axis([self.grid_size//2, -self.grid_size//2, -self.grid_size//2, self.grid_size//2])
        plt.grid(True)
        plt.savefig('rrt_path_plot.png')

if __name__ == "__main__":
    obstacle_list = []
    obstacle_list.append(Circle(5, 5, 0, 0, 1))
    obstacle_list.append(Circle(1, 6, 0, 0, 1))
    obstacle_list.append(Circle(2, 0, 0, 0, 1))
    obstacle_list.append(Circle(-2, -3, 0, 0, 1))
    
    rrt = RRTPlanner(obstacle_list=obstacle_list, width=20, height=20)
    start_node = RRTNode(4, 3, 0, 0, 0)
    goal_node = RRTNode(7, 8, 0, 0, 0)
    plan = rrt.plan(start_node, goal_node, max_iter=200, goal_radius=1.0)
    print(plan)
    rrt.draw_graph(goal_radius=1.0, path = plan)