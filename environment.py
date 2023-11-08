# import models

import pygame
import numpy as np
import math
import random
import time
import gymnasium
import sys
sys.modules["gym"] = gymnasium
from stable_baselines3 import PPO
from collections import OrderedDict


class CarParking(gymnasium.Env):
  
  def __init__(self, fps, step_limit):

#pygame parameters
    self.screen = None #pygame.display.set_mode((area_length, area_width))
    self.clock = None #pygame.time.Clock()

#environment constraints
    self.car_width = 150
    self.car_length = 400
    self.fps = fps
    self.car_hypotenuse = np.linalg.norm([self.car_length/2, self.car_width/2])
    self.hypotenuse_angle = math.acos(self.car_width/(2*self.car_hypotenuse))
    self.car_wheelbase=250
    self.parking_width = 240
    self.timestep = 1
    self.parking_length = 450
    self.road_width = 300
    self.speed_limit = 10
    self.area_width = (2 * self.road_width) + (2 * self.parking_length) #1300
    self.area_length = 8 * self.parking_width #1920
    self.parking_hypotenuse = np.linalg.norm([self.area_width,self.area_length])
    self.max_steering=30
    self.max_velocity=30 #in m/s
    self.step_limit = step_limit
    self.max_acceleration=3.4 #around 12kmh/s
    self.natural_decceleration=1
    self.car_image = pygame.image.load('car_sprite.png')
    self.car_image = pygame.transform.scale(self.car_image, (self.car_width, self.car_length))
    self.action_space=gymnasium.spaces.discrete.Discrete(9)
    # 'velocity','acceleration','angle','pos','steering','wheels_inside', 'distances', 'angular_velocity'
    self.observation_space=gymnasium.spaces.box.Box(
      low = np.array([-self.max_velocity,-self.max_acceleration,0,0,0,-self.max_steering,4,-self.area_length,-self.area_width,-0.06])
      ,high = np.array([self.max_velocity,self.max_acceleration,360,self.area_length,self.area_width,self.max_steering,4,self.area_length,self.area_width,0.06])
      ,dtype=np.int32
    )
    self.set_vars()


# checks if car has collided with border
  def check_if_collision(self, new_pos):
    for wheel in ['lb','rb','lf','rf']:
      # If the wheel has touched the border at all...
      if new_pos[wheel][0] < 0 or new_pos[wheel][0] >= self.area_length or new_pos[wheel][1] < 0 or new_pos[wheel][1] >= self.area_width:
        return True
    return False

  # checks if wheels are inside of the parking lot
  def check_if_inside(self, reward):
    
    goal = True
    wheels_inside = 0
    # for loop to check if individual wheels are inside the coordinates of the parking lot
    # lb - left back
    # rb - right back
    # lf - left front
    # rf - right front
    for wheel in ['lb','rb','lf','rf']:
      if self.goal_number<8:
        if self.wheels[wheel][0] < self.goal[0][0] or self.wheels[wheel][0] > self.goal[1][0] or self.wheels[wheel][1] < self.goal[0][1] or self.wheels[wheel][1] > self.goal[2][1]:
          goal = False
        else:
          wheels_inside += 1

      else:
        if self.wheels[wheel][0] < self.goal[0][0] or self.wheels[wheel][0] > self.goal[1][0] or self.wheels[wheel][1] > self.goal[0][1] or self.wheels[wheel][1] < self.goal[2][1]:
          goal = False
        else:
          wheels_inside += 1
    
    self.state_labelled['wheel_difference'] = wheels_inside - self.state_labelled['wheels_inside']
    self.state_labelled['wheels_inside'] = wheels_inside
    if goal:
      reward = 100
    else:
      reward += self.state_labelled['wheel_difference'] * 50
    return goal, reward

# function to close the pygame
  def close(self):
    if self.screen:
      pygame.quit()
      self.screen=None
      self.clock=None

# process new position after velocity and acceleration are calculated
  def process_new_position(self, reward):
      self.state_labelled['velocity'] += self.state_labelled['acceleration']
      if abs(self.state_labelled['velocity']) > self.speed_limit:
        reward -= 50
        # penalty if car goes over speed limit of 10m/s

      # ensures car doesn't go above max velocity
      self.state_labelled['velocity'] = max(-self.max_velocity, min(self.state_labelled['velocity'], self.max_velocity))
      if abs(self.state_labelled['velocity'])<0.01:
        self.state_labelled['velocity']=0

      if self.state_labelled['steering']:
        turning_radius = self.car_wheelbase / math.sin(math.radians(self.state_labelled['steering']))
        self.state_labelled['angular_velocity'] = self.state_labelled['velocity'] / turning_radius
      else:
        self.state_labelled['angular_velocity'] = 0

      self.state_labelled['pos'] = self.state_labelled['pos'] + np.array([self.state_labelled['velocity'] * math.sin(math.radians(self.state_labelled['angle']))*self.timestep,-self.state_labelled['velocity'] * math.cos(math.radians(self.state_labelled['angle']))*self.timestep])
      self.wheels = {}
      self.wheels['rf'], self.wheels['lf'], self.wheels['rb'], self.wheels['lb'] = self.calculate_four_corners(self.state_labelled['pos'])
      self.state_labelled['angle'] = (self.state_labelled['angle'] + math.degrees(self.state_labelled['angular_velocity'])*self.timestep)%360
      if self.state_labelled['angle'] < 0:
        self.state_labelled['angle']+= 360

      return reward
      
      
# calculate x and y coordinates of 4 wheels baesd on center of car
  def calculate_four_corners(self, pos):
    rb = pos + np.array([self.car_hypotenuse*math.cos(math.radians(self.state_labelled['angle'])+self.hypotenuse_angle),
                                                  self.car_hypotenuse*math.sin(math.radians(self.state_labelled['angle'])+self.hypotenuse_angle)])
    lb = pos + np.array([self.car_hypotenuse*math.cos(math.radians(self.state_labelled['angle'])+math.pi-self.hypotenuse_angle),
                                                  self.car_hypotenuse*math.sin(math.radians(self.state_labelled['angle'])+math.pi-self.hypotenuse_angle)])
    rf = pos + np.array([self.car_hypotenuse*math.cos(math.radians(self.state_labelled['angle'])+2*math.pi-self.hypotenuse_angle),
                                                  self.car_hypotenuse*math.sin(math.radians(self.state_labelled['angle'])+2*math.pi-self.hypotenuse_angle)])
    lf = pos + np.array([self.car_hypotenuse*math.cos(math.radians(self.state_labelled['angle'])+math.pi+self.hypotenuse_angle),
                                                  self.car_hypotenuse*math.sin(math.radians(self.state_labelled['angle'])+math.pi+self.hypotenuse_angle)])
    return rf, lf, rb, lb

# calculates vertical and horizontal distances to the opening of parking lot
  def calculate_wheels_to_t(self):
    distances = []
    for wheel in ['rf','lf','rb','lb']:
      dist_1 = self.goal[2]-self.wheels[wheel]
      dist_2 = self.goal[3]-self.wheels[wheel]
      distances.extend(dist_1.tolist())
      distances.extend(dist_2.tolist())
    return distances

# step function
  def step(self, action):
    if not self.clock:
      self.clock = pygame.time.Clock()
    reward = 0
    opposite = False
    terminated = False
    truncated = False
    # timestep = self.clock.get_time() / 1000

    # the code below transforms a discrete action into a tuple, with the first number relating to steering and the second to driving
    # steering = 0 => gradually resets the steering wheel to neutral
    # steering = 1 => to the right
    # steering = 2 => to the left

    # driving = 0 => gradually loses monentum and comes to halt
    # driving = 1 => accelerate
    # driving = 2 => decelerate
    action = np.array([int((action)/3)+1,((action)%3)+1]) 

    if action[0]==1:
      self.state_labelled['steering']+=self.max_steering*(self.timestep) 
    elif action[0]==2:
      self.state_labelled['steering']-=self.max_steering*(self.timestep)
    else:
      if self.state_labelled['steering'] < 0:
        self.state_labelled['steering'] +=self.max_steering*(self.timestep)
        self.state_labelled['steering'] = min(self.state_labelled['steering'],0)
      elif self.state_labelled['steering'] > 0:
        self.state_labelled['steering'] -= self.max_steering*(self.timestep)
        self.state_labelled['steering'] = max(self.state_labelled['steering'],0)


    if self.state_labelled['steering'] != max(-self.max_steering, min(self.state_labelled['steering'], self.max_steering)):
      self.state_labelled['steering'] = max(-self.max_steering, min(self.state_labelled['steering'], self.max_steering))

    if action[1]==1:
      if self.state_labelled['acceleration']<0:
        self.state_labelled['acceleration'] = 0

      if self.state_labelled['velocity']<0:
        opposite=True
        self.state_labelled['acceleration'] += self.max_acceleration * (self.timestep/3)

      self.state_labelled['acceleration'] += self.max_acceleration * (self.timestep/60) #15 second to reach full acceleration from 0

    elif action[1]==2:
      if self.state_labelled['acceleration']>0:
        self.state_labelled['acceleration'] =0

      if self.state_labelled['velocity']>0:
        opposite=True
        self.state_labelled['acceleration'] -= self.max_acceleration * (self.timestep/3)
 
      self.state_labelled['acceleration'] -= self.max_acceleration * (self.timestep/60)  #15 seconds to reach full acceleration from 0

    else: #action[1]==3:
      if self.state_labelled['velocity']<0:
        self.state_labelled['acceleration'] = min(self.max_acceleration * (self.timestep/10),-self.state_labelled['velocity'])

      elif self.state_labelled['velocity']>0:
        self.state_labelled['acceleration'] = max(-self.max_acceleration * (self.timestep/10),-self.state_labelled['velocity'])
      else:
        self.state_labelled['acceleration']=0
    
    #punish static steering
    if abs(self.state_labelled['velocity'])==0 and action[0]!=0 and action[1]==0:
      reward -= 10

    # punish if car is not vertical
    reward -= min(self.state_labelled['angle']%180,180-self.state_labelled['angle']%180)/10

    if not opposite:
      self.state_labelled['acceleration'] = max(-self.max_acceleration, min(self.state_labelled['acceleration'], self.max_acceleration))
    reward = self.process_new_position(reward)
    self.wheels['rf'], self.wheels['lf'], self.wheels['rb'], self.wheels['lb'] = self.calculate_four_corners(self.state_labelled['pos'])
    distances = self.calculate_distances()
    self.state_labelled['distance_difference'] = np.linalg.norm(distances) - np.linalg.norm(self.state_labelled['distances'])
    self.state_labelled['distances'] = distances
    reward += -self.state_labelled['distance_difference']*5
    
    # penalise for every timestep
    reward -= 1
    
    # check if done
    terminated, reward = self.check_if_inside(reward)
    if self.check_if_collision(self.wheels):
      terminated = True
      reward = -100


    
    info = {}
    self.clock.tick(self.fps)
    self.last_reward = reward
    # change and normalize the labelled dictionary state into a 1 dimensional state to feed into the neural network
    self.state = self.deconstruct_array(self.state_labelled)
    self.current_step += 1
    if self.current_step == self.step_limit:
      truncated = True
    return self.state, reward, terminated, truncated, info
  
  # rendering graphics of the environment
  def render(self):
    if not self.screen:
      pygame.init()
      pygame.display.init()
      if __name__=='__main__':
        self.screen = pygame.display.set_mode((self.area_length, self.area_width), pygame.FULLSCREEN | pygame.SCALED)
      else:
        self.screen = pygame.display.set_mode((self.area_length, self.area_width), pygame.HIDDEN)
      
    self.screen.fill((255, 255, 255))
    if self.goal_number < 8:
      goal = pygame.Rect(self.goal[0][0], self.goal[0][1], self.parking_width, self.parking_length)
    else:
      goal = pygame.Rect(self.goal[2][0], self.goal[2][1], self.parking_width, self.parking_length)
    pygame.draw.rect(self.screen, (0,0,255), goal)
    
    for i in range(7):
      rectangle_top = pygame.Rect(self.parking_width*(i+1)-1, 0, 3, self.parking_length)
      rectangle_bottom = pygame.Rect(self.parking_width*(i+1)-1, self.parking_length + 2 * self.road_width-1, 3, self.parking_length)
      pygame.draw.rect(self.screen, (0,0,0), rectangle_top)
      pygame.draw.rect(self.screen, (0,0,0), rectangle_bottom)


    font = pygame.font.Font(None, 36)
    vel_text = font.render(f"velocity: {self.state_labelled['velocity']}", True, (255, 0, 0)) 
    acc_text = font.render(f"acceleration: {self.state_labelled['acceleration']}", True, (255, 0, 0)) 
    steer_text = font.render(f"Steering angle: {self.state_labelled['steering']}", True, (255, 0, 0)) 
    wheels_text = font.render(f"wheels inside: {self.state_labelled['wheels_inside']}", True, (255, 0, 0)) 
    distance_text = font.render(f"distance: ({round(self.state_labelled['distances'][0],2)},{round(self.state_labelled['distances'][1],2)})", True, (255, 0, 0)) 
    reward_text = font.render(f"last reward: ({self.last_reward})", True, (255, 0, 0)) 
    pos_text = font.render(f"pos: {self.state_labelled['pos']}", True, (255, 0, 0)) 
    ang_vel_text = font.render(f"angular velocity: {self.state_labelled['angular_velocity']}", True, (255,0,0))
    step_text = font.render(f"step: {self.current_step}", True, (255, 0, 0)) 
    # wheel_difference_text = font.render(f"wheel_difference: {self.state_labelled['wheel_difference']}", True, (255, 0, 0)) 
    # distance_difference_text = font.render(f"distance_difference: {self.state_labelled['distance_difference']}", True, (255, 0, 0)) 
    angle_text = font.render(f"angle: {self.state_labelled['angle']}", True, (255, 0, 0)) 
    self.screen.blit(vel_text, (1000, 10))
    self.screen.blit(acc_text, (1000, 100))
    self.screen.blit(steer_text, (1000, 200))
    self.screen.blit(wheels_text, (1000, 300))
    self.screen.blit(distance_text, (1000, 400))
    self.screen.blit(pos_text, (1000, 500))
    self.screen.blit(ang_vel_text, (1000, 600))
    self.screen.blit(reward_text, (1000, 700))
    self.screen.blit(step_text, (1000, 800))
    # self.screen.blit(wheel_difference_text, (1000, 900))
    # self.screen.blit(distance_difference_text, (1000, 1000))
    self.screen.blit(angle_text, (1000, 900))

    
    rotated = pygame.transform.rotate(self.car_image, -self.state_labelled['angle'])
    rotated_rect=rotated.get_rect()
    rotated_rect.center = self.state_labelled['pos']
    self.screen.blit(rotated, rotated_rect)
    pygame.draw.circle(self.screen,(0,255,0),self.state_labelled['pos'],5)
    pygame.draw.circle(self.screen,(0,255,255),self.goal_center,10)
    for part in ['rf','lf','rb','lb']:
      pygame.draw.circle(self.screen, (255,0,0), self.wheels[part], 5)
    pygame.display.update()
    return self.screen


# variable values upon initialization
  def set_vars(self):
    x_noise = random.uniform(-(self.road_width-self.car_width)/2,(self.road_width-self.car_width)/2)
    y_noise = random.uniform(0,50)
    self.current_step = 0
    self.state_labelled = OrderedDict({
        'velocity': 0
        ,'angular_velocity': 0
        ,'acceleration': 0
        ,'angle': 90
        ,'pos': np.array([(self.car_length/2)+10,self.parking_length+(self.road_width/2)-1])
        ,'steering': 0
        ,'wheels_inside': 0
        ,'wheel_difference': 0
        ,'distance_difference': 0})
    # there are 16 parking bays in the environment. Each episode the environment will choose a random bay for the agent to drive into.
    # self.goal_number = random.randint(0,15)
    self.goal_number = 15
    self.wheels = {}
    self.wheels['rf'], self.wheels['lf'], self.wheels['rb'], self.wheels['lb'] = self.calculate_four_corners(self.state_labelled['pos'])
    
    if self.goal_number < 8:
      self.goal = np.array([[(self.goal_number*self.parking_width),0], #top left
                            [((self.goal_number+1)*self.parking_width)-1,0], #top right
                            [(self.goal_number*self.parking_width),self.parking_length-1], #inside left
                            [((self.goal_number+1)*self.parking_width)-1,self.parking_length-1]]) #inside right
    else:
      self.goal = np.array([[((self.goal_number-8)*self.parking_width),self.area_width-1], #bottom left
                            [((self.goal_number-7)*self.parking_width)-1,self.area_width-1], #bottom right
                            [((self.goal_number-8)*self.parking_width),self.parking_length+(2*self.road_width)], #inside left
                            [((self.goal_number-7)*self.parking_width)-1,self.parking_length+(2*self.road_width)]]) #inside right
    # get distance of all wheels to the two outside parking lines
    self.goal_center = np.array(np.mean(self.goal, axis=0))
    self.state_labelled['distances'] = self.calculate_distances()
    self.last_reward = 0
    #get angle
    
    
    # This is the normalisation table
    # it normalises the values by summing the first number (the minimum), and dividing by the second number (range) to fix the range between 0 and 1
    self.normalisation_table = {
      'velocity': [self.max_velocity,self.max_velocity*2], 
      'angular_velocity': [0.06,0.12],
      'acceleration': [self.max_acceleration, self.max_acceleration*2],
      'angle': [0, 360],
      'pos_x': [0, self.area_length],
      'pos_y': [0, self.area_width],
      'steering': [self.max_steering, self.max_steering * 2],
      'wheels_inside': [0, 4],
      'wheel_difference': [2, 2],
      'distances_x': [self.area_length, self.area_length*2],
      'distances_y': [self.area_width, self.area_width*2]
    }
    self.state = self.deconstruct_array(self.state_labelled)


# calculates distance of car centre to parking bay centre
  def calculate_distances(self): #return array of 2 (center of car - center of goal)
    distances = np.zeros(2)
    x = self.goal_center[0]
    y =self.goal_center[1]
    distances[0] = self.state_labelled['pos'][0]-x
    distances[1] = self.state_labelled['pos'][1]-y
    return distances

  def calculate_angle_from_parking(self):
    return math.degrees(np.arctan2(self.goal_center[1] - self.state_labelled['pos'][1], abs(self.goal_center[0] - self.state_labelled['pos'][0])))

# resets variables and state space to initial values
  def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    self.set_vars()
    return self.state, None
  
  # convert tuple action into single number between 0 and 8
  def convert_actions(self, action):
    return action[1] + (action[0]*3)
  
  # turns labelled dictionary into 1 dimensional normalised array
  def deconstruct_array(self,arr):
    llist = []
    #'velocity','acceleration','angle','pos','steering','wheels_inside', 'wheel_difference','distances', 'distance_difference', 'angular_velocity'
    for key in ['velocity','acceleration','angle','pos','steering','wheels_inside', 'distances', 'angular_velocity']:
      if key in ['pos', 'distances']:
        new_arr = arr[key].flatten()
        # new_arr[0] = (new_arr[0]+self.normalisation_table[f'{key}_x'][0])/self.normalisation_table[f'{key}_x'][1]
        # new_arr[1] = (new_arr[1]+self.normalisation_table[f'{key}_y'][0])/self.normalisation_table[f'{key}_y'][1]
        llist.extend(new_arr.tolist())
      elif key == 'distance_difference':
        # llist.append((arr[key]+self.normalisation_table['velocity'][0])/self.normalisation_table['velocity'][1])
        llist.append(arr[key])

      else:
        # llist.append((arr[key]+self.normalisation_table[key][0])/self.normalisation_table[key][1])
        llist.append(arr[key])
        
    return np.array(llist)
  

# to control the environment using WASD keys
if __name__ == '__main__':
  fps = 60
  step_limit = 100_000
  env = CarParking(fps, step_limit)
  state, _ = env.reset()
  score = 0
  start = time.time()
  env.render()
  terminated = False
  truncated = False
  i = 0

  if len(sys.argv) > 1:
    model = PPO.load("./logs/best_model", env=env)
    obs = env.state
    env.render()
    terminated = False
    truncated = False
    action = 1
    while not (terminated or truncated):
        action, _states = model.predict(obs)
        state, reward, terminated, truncated, info = env.step(action)
        score += reward
        env.render()

  else:
    while not (terminated or truncated):
      i += 1
      for event in pygame.event.get():
        if event.type == pygame.QUIT:
          truncated = True
          break

      driving = 2
      steering = 2
      keys = pygame.key.get_pressed()
      if keys[pygame.K_w]:
        driving = 0
      elif keys[pygame.K_s]:
        driving = 1


      if keys[pygame.K_a]:
        steering = 1
      elif keys[pygame.K_d]:
        steering = 0

      action = np.array([steering,driving])
      state, reward, terminated, truncated, info = env.step(env.convert_actions(action))
      env.render()
      score += reward
      # time.sleep(1/fps)
      if i == 2000000:
        done = True
  env.close()
  end = time.time()
  print(f'SCORE: {score}')

