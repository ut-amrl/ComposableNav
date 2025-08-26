from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Point, LineString, Polygon

class Obstacle(ABC):
    def __init__(self, x, y, theta, speed):
        self.x = x
        self.y = y
        self.theta = theta
        self.speed = speed
        
    def pos_at_dt(self, dt):
        return self.x + self.speed * np.cos(self.theta) * dt, self.y + self.speed * np.sin(self.theta) * dt, self.theta

    @abstractmethod
    def contains(self, x, y, dt, buffer):
        pass
    
    @abstractmethod
    def draw(self, t):
        pass
    
    def __str__(self) -> str:
        return f'x: {self.x}, y: {self.y}, theta: {self.theta}, speed: {self.speed}'
    
class Circle(Obstacle):
    def __init__(self, x, y, theta, speed, radius, **kwargs):
        super().__init__(x, y, theta, speed)
        self.radius = radius

    def contains(self, x, y, dt, buffer):
        dt_x, dt_y, _ = self.pos_at_dt(dt)
        return (x - dt_x)**2 + (y - dt_y)**2 <= (self.radius + buffer)**2

    def preference(self, x, y):
        dist = np.sqrt((x - self.x)**2 + (y - self.y)**2)
        preference_value = np.arctan(-(dist - self.radius)) + np.pi/2
        return preference_value
    
    def draw(self, t):
        dt_x, dt_y, _ = self.pos_at_dt(t)
        # circle = plt.Circle((dt_y, dt_x), self.radius, color='g')
        # ax.add_patch()
        circle = patches.Circle((dt_y, dt_x), radius=self.radius, edgecolor='black', facecolor='white')
        plt.gca().add_patch(circle)

        dx = self.speed * np.cos(self.theta)
        dy = self.speed * np.sin(self.theta)
        dt_y += np.sin(self.theta) * self.radius
        dt_x += np.cos(self.theta) * self.radius
        plt.arrow(dt_y, dt_x, dy, dx, head_width=0.2, head_length=0.5, fc='r', ec='r')
    
    def draw_w_names(self, t, name_idx):
        self.draw(t)
        dt_x, dt_y, _ = self.pos_at_dt(t)
        plt.text(dt_y, dt_x, f"Person {name_idx}", fontsize=10, fontweight='bold', color='black', ha='center', va='center')  

    def draw_w_names_color(self, t, name_idx, color):
        dt_x, dt_y, _ = self.pos_at_dt(t)
        if dt_x < -0.5 or dt_y < -6.25 or dt_x > 19.8 or dt_y > 6.25:
            # hardcoded
            return
        # circle = plt.Circle((dt_y, dt_x), self.radius, color='g')
        # ax.add_patch()
        circle = patches.Circle((dt_y, dt_x), radius=self.radius, edgecolor=color, facecolor=color)
        plt.gca().add_patch(circle)

        dx = 0.3 * np.cos(self.theta)
        dy = 0.3 * np.sin(self.theta)
        dt_y += np.sin(self.theta) * self.radius
        dt_x += np.cos(self.theta) * self.radius
        plt.arrow(dt_y, dt_x, dy, dx, head_width=0.5, head_length=0.6, fc=color, ec=color)

        dt_x, dt_y, _ = self.pos_at_dt(t)
        plt.text(dt_y, dt_x, f"{name_idx}", fontsize=10, fontweight='bold', color='white', ha='center', va='center')  

    def grid_cover(self, buffer):
        locs = []
        iter_radius = int(np.ceil(self.radius + buffer))
        min_x = int(np.floor(self.x - self.radius - buffer))
        max_x = int(np.ceil(self.x + self.radius + buffer))
        min_y = int(np.floor(self.y - self.radius - buffer))
        max_y = int(np.ceil(self.y + self.radius + buffer))
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                dist = np.sqrt((x-self.x)**2 + (y-self.y)**2)
                if dist <= self.radius + buffer:
                    locs.append((x, y, self.speed * np.cos(self.theta), self.speed * np.sin(self.theta)))
                    
        return locs
        
    def collision(self, x1, y1, x2, y2, dt, buffer):
        dt_x, dt_y, _ = self.pos_at_dt(dt)
        point = Point(dt_x, dt_y)
        if x1 == x2 and y1 == y2:
            x2 += 1e-9 # avoid warning
        line = LineString([(x1, y1), (x2, y2)])
        collides = point.distance(line) <= self.radius + buffer
        return collides
    
    def collide_with_other(self, other, dt, buffer):
        dt_x, dt_y, _ = self.pos_at_dt(dt)
        other_dt_x, other_dt_y, _ = other.pos_at_dt(dt)
        collided = (dt_x - other_dt_x)**2 + (dt_y - other_dt_y)**2 <= (self.radius + other.radius + buffer)**2
        return collided
    
    def to_dict(self):
        return {'type': 'Circle', 'x': self.x, 'y': self.y, 'theta': self.theta, 'speed': self.speed, 'radius': self.radius}

    def clearance_dist(self, x, y, t):
        obs_x, obs_y, obs_theta = self.pos_at_dt(t)
        dist = np.sqrt((obs_x - x)**2 + (obs_y - y)**2)
        return dist 

    def __str__(self) -> str:
        return f'Circle: x: {self.x}, y: {self.y}, theta: {self.theta}, speed: {self.speed}'
    
    def get_name(self):
        return "Person at location ({}, {})".format(self.x, self.y)
    
class CircleNoisy(Obstacle):
    def __init__(self, x, y, theta, speed, radius, planning_dt, seed, 
                 noise_mean, noise_std, noise_size, **kwargs):
        super().__init__(x, y, theta, speed)
        self.radius = radius
        self.planning_dt = planning_dt
        np.random.seed(seed)
        self.noise = np.random.normal(loc=noise_mean, 
                            scale=noise_std, 
                            size=noise_size)
    def pos_at_dt(self, dt):
        idx = int(round(dt / self.planning_dt))
        nx, ny= self.noise[idx]
        x = self.x + self.speed * np.cos(self.theta) * dt + nx
        y = self.y + self.speed * np.sin(self.theta) * dt + ny
        theta = self.theta
        return x, y, theta
    
    def contains(self, x, y, dt, buffer):
        dt_x, dt_y, _ = self.pos_at_dt(dt)
        return (x - dt_x)**2 + (y - dt_y)**2 <= (self.radius + buffer)**2

    def preference(self, x, y):
        dist = np.sqrt((x - self.x)**2 + (y - self.y)**2)
        preference_value = np.arctan(-(dist - self.radius)) + np.pi/2
        return preference_value
    
    def draw(self, t):
        dt_x, dt_y, _ = self.pos_at_dt(t)
        circle = plt.Circle((dt_y, dt_x), self.radius, color='g')
        plt.gca().add_patch(circle)

        dx = self.speed * np.cos(self.theta)
        dy = self.speed * np.sin(self.theta)
        plt.arrow(dt_y, dt_x, dy, dx, head_width=1, head_length=1, fc='r', ec='r')
        
    def grid_cover(self, buffer):
        locs = []
        iter_radius = int(np.ceil(self.radius + buffer))
        min_x = int(np.floor(self.x - self.radius - buffer))
        max_x = int(np.ceil(self.x + self.radius + buffer))
        min_y = int(np.floor(self.y - self.radius - buffer))
        max_y = int(np.ceil(self.y + self.radius + buffer))
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                dist = np.sqrt((x-self.x)**2 + (y-self.y)**2)
                if dist <= self.radius + buffer:
                    locs.append((x, y, self.speed * np.cos(self.theta), self.speed * np.sin(self.theta)))
                    
        return locs
        
    def collision(self, x1, y1, x2, y2, dt, buffer):
        dt_x, dt_y, _ = self.pos_at_dt(dt)
        point = Point(dt_x, dt_y)
        if x1 == x2 and y1 == y2:
            x2 += 1e-9 # avoid warning
        line = LineString([(x1, y1), (x2, y2)])
        collides = point.distance(line) <= self.radius + buffer
        return collides
    
    def collide_with_other(self, other, dt, buffer):
        dt_x, dt_y, _ = self.pos_at_dt(dt)
        other_dt_x, other_dt_y, _ = other.pos_at_dt(dt)
        collided = (dt_x - other_dt_x)**2 + (dt_y - other_dt_y)**2 <= (self.radius + other.radius + buffer)**2
        return collided
    
    def to_dict(self):
        return {'type': 'Circle', 'x': self.x, 'y': self.y, 'theta': self.theta, 'speed': self.speed, 'radius': self.radius, 
                'planning_dt': self.planning_dt, 'noise': self.noise}
    
    def clearance_dist(self, x, y, t):
        obs_x, obs_y, obs_theta = self.pos_at_dt(t)
        dist = np.sqrt((obs_x - x)**2 + (obs_y - y)**2)
        return dist 
        
class Rectangle(Obstacle):
    def __init__(self, x, y, theta, speed, width, height, **kwargs):
        super().__init__(x, y, theta, speed)
        # this is a static rectangle
        assert theta == 0 and speed == 0, "Only support static rectangle for now"
        self.width = width
        self.height = height
        
        self.upper_left = np.array([x + height/2, y + width/2])
        self.upper_right = np.array([x + height/2, y - width/2])
        self.lower_left = np.array([x - height/2, y + width/2])
        self.lower_right = np.array([x - height/2, y - width/2])
    
    def to_coord(self):
        return self.upper_left, self.upper_right, self.lower_left, self.lower_right

    def contains(self, x, y, dt, buffer):
        # check if x, y in the rectangle
        bottom_x = -self.height/2 - buffer + self.x
        top_x = self.height/2 + buffer + self.x
        right_y = -self.width/2 - buffer + self.y
        left_y = self.width/2 + buffer + self.y

        return bottom_x <= x <= top_x + buffer and right_y <= y <= left_y

    def draw(self, t):
        bottom_x = -self.height/2  + self.x
        right_y = -self.width/2 + self.y # because y is flipped

        rectangle = patches.Rectangle((right_y, bottom_x), 
                                    self.width, self.height, 
                                    angle=0, 
                                    color='gray')

        # Add the rectangle to the current axis
        plt.gca().add_patch(rectangle)
    
    def collision(self, x1, y1, x2, y2, dt, buffer):
        return self.contains(x1, y1, dt, 1) or self.contains(x2, y2, dt, 1) # hardcode buffer to 1
    
    def to_dict(self):
        return {'type': 'Rectangle', 'x': self.x, 'y': self.y, 'theta': self.theta, 'speed': self.speed, 'width': self.width, 'height': self.height}

class RectangleCorner(Obstacle):
    def __init__(self, top, bottom, left, right, **kwargs):
        super().__init__((top-bottom)/2 + bottom, (left-right)/2 + right, 0, 0)
        # this is a static rectangle
        
        self.top = top
        self.bottom = bottom 
        self.left = left 
        self.right = right
        self.width = left - right
        self.height = top - bottom 
        self.upper_left = np.array([top, left])
        self.upper_right = np.array([top, right])
        self.lower_left = np.array([bottom, left])
        self.lower_right = np.array([bottom, right])
        assert self.left > self.right and self.top > self.bottom
    
    def get_repr(self):
        return self.top, self.bottom, self.left, self.right 
    
    def to_coord(self):
        return self.upper_left, self.upper_right, self.lower_left, self.lower_right

    def contains(self, x, y, dt, buffer):
        # check if x, y in the rectangle
        return self.bottom <= x <= self.top + buffer and self.right <= y <= self.left

    # def draw(self, t):
    #     rectangle = patches.Rectangle((self.right, self.bottom), 
    #                                 self.width, self.height, 
    #                                 angle=0, 
    #                                 color='gray')

    #     # Add the rectangle to the current axis
    #     plt.gca().add_patch(rectangle)

    def draw(self, t, offset=0):
        rectangle = patches.Rectangle((self.right+offset, self.bottom+offset), 
                                    self.width-offset*2, self.height-offset*2, 
                                    angle=0, 
                                    color='#222222',
                                    alpha=0.9)

        # Add the rectangle to the current axis
        plt.gca().add_patch(rectangle)   
        
    def draw_w_names(self, t, name_idx, offset=0):
        self.draw(t, offset)
        plt.text(self.y, self.x, f"{name_idx}", fontsize=10, fontweight='bold', color='white', ha='center', va='center')
    
    def collision(self, x1, y1, x2, y2, dt, buffer=1): # hardcode buffer to 1 when used with random but this is changed 12/27/2024
        # Define the rectangle as a polygon
        rectangle = Polygon([
            [self.bottom - buffer, self.right - buffer],
            [self.top + buffer, self.right - buffer],
            [self.top + buffer, self.left + buffer],
            [self.bottom - buffer, self.left + buffer]
        ])
        
        # Define the line segment
        line = LineString([(x1, y1), (x2, y2)])
        
        # Check for intersection
        return line.intersects(rectangle)
    
    def to_dict(self):
        return {'type': 'RectangleCorner', 'top': self.top, 'bottom': self.bottom, 'left': self.left, 'right': self.right}

    def __str__(self):
        return "RectangleCorner: top: {}, bottom: {}, left: {}, right: {}".format(self.top, self.bottom, self.left, self.right)
    
    def clearance_dist(self, x, y, t):
        return 10000
        return self.dist_to_obstacle(x, y, t)
        
    def dist_to_obstacle(self, x, y, t):
        if self.bottom <= x <= self.top and self.right <= y <= self.left:
            return 0
        elif self.right <= y <= self.left:
            return min(
                abs(x - self.bottom),   
                abs(x - self.top)
            )
        elif self.bottom <= x <= self.top:
            return min(
                abs(y - self.right),
                abs(y - self.left)
            )
        
        # If the point is outside the rectangle
        closest_x = max(self.bottom, min(x, self.top))
        closest_y = max(self.right, min(y, self.left))
        return np.sqrt((closest_x - x)**2 + (closest_y - y)**2)
    
    def get_name(self):
        return "Region with position and height width ({}, {}, {}, {})".format(self.x, self.y, self.height, self.width)

class RectangleCornerAvoid(RectangleCorner):
    def draw(self, t, offset=0):
        rectangle = patches.Rectangle((self.right+offset, self.bottom+offset), 
                                    self.width-offset*2, self.height-offset*2, 
                                    angle=0, 
                                    color='#222222',
                                    alpha=0.9)

        # Add the rectangle to the current axis
        plt.gca().add_patch(rectangle)    

    def draw_w_names(self, t, name_idx, offset=0):
        self.draw(t, offset)
        plt.text(self.y, self.x, f"{name_idx}", fontsize=10, fontweight='bold', color='white', ha='center', va='center')
       
    def collision(self, x1, y1, x2, y2, dt, buffer):
        return False
    
    def clearance_dist(self, x, y, t):
        return 1000 # large number (always not avoiding)

class RectangleCornerPrefer(RectangleCorner):
    def draw(self, t, offset=0):
        rectangle = patches.Rectangle((self.right-offset, self.bottom-offset), 
                                    self.width+offset*2, self.height+offset*2, 
                                    angle=0, 
                                    color='#3B7D23',
                                    alpha=0.3)

        # Add the rectangle to the current axis
        plt.gca().add_patch(rectangle)    

    def draw_w_names(self, t, name_idx, offset=0):
        self.draw(t, offset)
        plt.text(self.y, self.x, f"{name_idx}", fontsize=10, fontweight='bold', color='black', ha='center', va='center')
       
    def collision(self, x1, y1, x2, y2, dt, buffer):
        return False
    
    def clearance_dist(self, x, y, t):
        return 1000 # large number (always not avoiding)

class RectangleRotate(Obstacle):
    def __init__(self, x, y, theta, speed, width, height, **kwargs):
        super().__init__(x, y, theta, speed)
        self.width = width
        self.height = height

    def contains(self, x, y, dt, buffer):
        dt_x, dt_y, _ = self.pos_at_dt(dt)
        global_to_rect = np.linalg.inv(np.array([[np.cos(self.theta), -np.sin(self.theta), dt_x], 
                                   [np.sin(self.theta), np.cos(self.theta), dt_y],
                                   [0, 0, 1]]))     
        x, y, _ = global_to_rect @ np.array([x, y, 1])
        return -self.height/2 - buffer <= x <= self.height/2 + buffer and -self.width/2 - buffer <= y <= self.width/2 + buffer

    def preference(self, x, y):
        dist = np.sqrt((x - self.x)**2 + (y - self.y)**2)
        max_size = np.sqrt((self.width/2)**2 + (self.height/2)**2)
        preference_value = np.arctan(-(dist - max_size)) + np.pi/2
        return preference_value

    def draw(self, t):
        dt_x, dt_y, _ = self.pos_at_dt(t)
        # Rotation matrix to apply to the corner offsets
        rect_rot = np.array([[np.cos(self.theta), np.sin(self.theta)], 
                            [-np.sin(self.theta), np.cos(self.theta)]])
        
        # Calculate the offset for the bottom-right corner relative to the center
        bottom_left_offset = rect_rot @ np.array([-self.width / 2, -self.height / 2])
        bottom_left_x = dt_x + bottom_left_offset[1]
        bottom_left_y = dt_y + bottom_left_offset[0]
        # TODO: rotation is not correct
        rectangle = patches.Rectangle((bottom_left_y, bottom_left_x), 
                                    self.width, self.height, 
                                    angle=-self.theta * 180 / np.pi, 
                                    color='gray')

        # Add the rectangle to the current axis
        plt.gca().add_patch(rectangle)

    def grid_cover(self, buffer):
        # TODO implement grid cover for rectangle
        rad = np.sqrt((self.width/2)**2 + (self.height/2)**2) + buffer
        for x in range(int(np.floor(self.x - rad)), int(np.ceil(self.x + rad))):
            for y in range(int(np.floor(self.y - rad)), int(np.ceil(self.y + rad))):
                if self.contains(x, y, 0, buffer):
                    locs.append((x, y, self.speed * np.cos(self.theta), self.speed * np.sin(self.theta)))
        locs = []     
        return locs
    
    def collision(self, x1, y1, x2, y2, dt, buffer):
        return self.contains(x1, y1, dt, buffer) or self.contains(x2, y2, dt, buffer)
        # TODO implement collision check for rectangle
        return False
    
    def to_dict(self):
        return {'type': 'Rectangle', 'x': self.x, 'y': self.y, 'theta': self.theta, 'speed': self.speed, 'width': self.width, 'height': self.height}


class Human(Circle):
    def __init__(self, future_positions, name, radius, dt_interval=0.25, **kwargs):
        super().__init__(x=None, y=None, theta=None, speed=None, radius=radius)
        self.future_positions = future_positions
        self.name = name
        self.dt_interval = dt_interval

    def pos_at_dt(self, dt: float):
        dt = round(dt / self.dt_interval)
        if dt >= len(self.future_positions):
            print(f"Warning: dt is greater than the length of future_positions: {dt} >= {len(self.future_positions)}")
            raise Exception("Warning: dt is greater than the length of future_positions")
        future_pos = self.future_positions[dt]
        return future_pos[0], future_pos[1], self.theta
    
    def contains(self, x, y, dt, buffer):
        dt_x, dt_y, _ = self.pos_at_dt(dt)
        return (x - dt_x)**2 + (y - dt_y)**2 <= (self.radius + buffer)**2
    
    def draw(self, t):
        # not using provided yaw but calculating it from the future positions
        if round(t / self.dt_interval) > len(self.future_positions) - 1:
            return 
        
        dt_x, dt_y, _ = self.pos_at_dt(t)
        circle = plt.Circle((dt_y, dt_x), self.radius, color='g')
        plt.gca().add_patch(circle)
        plt.text(dt_y, dt_x, self.name, fontsize=12, color='black', ha='center', va='center')
        
        if round((t + self.dt_interval)/self.dt_interval) >= len(self.future_positions):
            return
        dt1_x, dt1_y, _ = self.pos_at_dt(t + self.dt_interval)
        dt_theta = np.arctan2(round(dt1_y - dt_y, 5), round(dt1_x - dt_x, 5))

        dx = np.cos(dt_theta)
        dy = np.sin(dt_theta)
        plt.arrow(dt_y, dt_x, dy, dx, head_width=0.1, head_length=0.1, fc='r', ec='r')
        
    def draw_with_offset(self, t, start_x, start_y, start_theta, x_offset, y_offset):
        if t > len(self.future_positions) - 1:
            return 
        
        start_offset = np.array([x_offset, y_offset, 0])
        global_to_start = np.linalg.inv(np.array([[np.cos(start_theta), -np.sin(start_theta), start_x], 
                                    [np.sin(start_theta), np.cos(start_theta), start_y],
                                    [0, 0, 1]]))   
            
        dt_x, dt_y, _ = self.pos_at_dt(t)
        dt_x, dt_y, _ = global_to_start @ np.array([dt_x, dt_y, 1]) + start_offset
        circle = plt.Circle((dt_y, dt_x), self.radius, color='g')
        plt.gca().add_patch(circle)
        plt.text(dt_y, dt_x, self.name, fontsize=12, color='black', ha='center', va='center')
        
        if round((t + self.dt_interval)/self.dt_interval) >= len(self.future_positions):
            return
        dt1_x, dt1_y, _ = self.pos_at_dt(t + self.dt_interval)
        dt1_x, dt1_y, _ = global_to_start @ np.array([dt1_x, dt1_y, 1]) + start_offset
        dt_theta = np.arctan2(round(dt1_y - dt_y, 5), round(dt1_x - dt_x, 5))

        dx = np.cos(dt_theta)
        dy = np.sin(dt_theta)

        # dx = np.cos(dt_theta - start_theta)
        # dy = np.sin(dt_theta - start_theta)
        plt.arrow(dt_y, dt_x, dy, dx, head_width=0.1, head_length=0.1, fc='r', ec='r')

    def __str__(self) -> str:
        return f"Human name {self.name}: num future positions: {len(self.future_positions)} | radius: {self.radius} | dt: {self.dt_interval}"