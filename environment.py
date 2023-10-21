
import pygame
import numpy as np
import math
import random
from gym import Env
from collections import OrderedDict
from gym.spaces import Discrete, Box, Dict

class CarParking(Env):
  
  def __init__(self):
    # 1=right, 2=left, 3=stay
    # 1=accelerate, 2=deccelerate, 3=same speed and direction

    self.screen = None #pygame.display.set_mode((area_length, area_width))
    self.clock = None #pygame.time.Clock()
    self.max_iterations = 1000
    self.car_width = 150
    self.car_length = 400
    self.fps = 60
    self.car_hypotenuse = math.sqrt((self.car_length/2)**2 + (self.car_width/2)**2)
    self.hypotenuse_angle = math.acos(self.car_width/(2*self.car_hypotenuse))
    self.car_wheelbase=250
    self.parking_width = 240
    self.parking_length = 450
    self.road_width = 300
    self.area_width = (2 * self.road_width) + (2 * self.parking_length) #1300
    self.area_length = 8 * self.parking_width #1920
    self.timestep=1
    self.max_steering=30
    self.max_velocity=30 #in m/s
    self.speed_limit=10
    self.max_acceleration=3.4 #around 12kmh/s
    self.max_decceleration=6
    self.natural_decceleration=1
    self.car_image = pygame.image.load('car_sprite.png')
    self.car_image = pygame.transform.scale(self.car_image, (self.car_width, self.car_length))
    self.action_space=Box(low=np.array([1,1]),high=np.array([3,3]),dtype=int)
    self.observation_space=Dict({
        'velocity': Box(low=np.array([0]),high=np.array([150]))
        ,'acceleration': Box(low=np.array([0]),high=np.array([150]))
        ,'angle': Box(low=np.array([0]),high=np.array([359])) #angle of car
        ,'pos': Box(low=np.array([0, 0]), high=np.array([self.area_length-1, self.area_width-1])) #calculate 4 corners of car using trigonometry
        ,'steering': Discrete(61,start=-30)#angle of steering wheel
        ,'wheels_inside': Discrete(5,start=0)
        ,'distances': Box(low=np.array([0,0,0,0]),high=np.array([self.area_width]))
        ,'y_dist': Box(low=np.array([0]),high=np.array([self.area_length]))
        })
    self.set_vars()


  def check_if_inside(self):
    goal = True
    wheels_inside = 0
    reward = 0
    for wheel in ['lb','rb','lf','rf']:
      if self.wheels[wheel][0] < 0 or self.wheels[wheel][0] >= self.area_length or self.wheels[wheel][1] < 0 or self.wheels[wheel][1] >= self.area_width:
        return True,-10000
      if self.wheels[wheel][0] < self.goal[0][0] or self.wheels[wheel][0] > self.goal[1][0] or self.wheels[wheel][1] < self.goal[0][1] or self.wheels[wheel][1] > self.goal[2][1]:
        goal = False
      else:
        wheels_inside += 1
        reward += 1
    self.state['wheels_inside'] = wheels_inside
    if goal:
      return True, 5000
    return False, reward

  def close(self):
    if self.screen:

      pygame.quit()
      self.screen=None
      self.clock=None

  def process_new_position(self, reward):
      self.state['velocity'] += self.state['acceleration']
      if abs(self.state['velocity']) > self.speed_limit:
        reward -= 1
      self.state['velocity'] = max(-self.max_velocity, min(self.state['velocity'], self.max_velocity))

      if self.state['steering']:
        turning_radius = self.car_wheelbase / math.sin(math.radians(self.state['steering']))
        angular_velocity = self.state['velocity'] / turning_radius
      else:
        angular_velocity = 0

      self.state['pos'] += np.array([self.state['velocity'] * math.sin(math.radians(self.state['angle']))*self.timestep,-self.state['velocity'] * math.cos(math.radians(self.state['angle']))*self.timestep])
      self.state['angle'] = (self.state['angle'] + math.degrees(angular_velocity) * self.timestep)%360
      if self.state['angle'] < 0:
        self.state['angle']+= 360
      

  def calculate_four_corners(self):
    rb = self.state['pos'] + np.array([self.car_hypotenuse*math.cos(math.radians(self.state['angle'])+self.hypotenuse_angle),
                                                  self.car_hypotenuse*math.sin(math.radians(self.state['angle'])+self.hypotenuse_angle)])
    lb = self.state['pos'] + np.array([self.car_hypotenuse*math.cos(math.radians(self.state['angle'])+math.pi-self.hypotenuse_angle),
                                                  self.car_hypotenuse*math.sin(math.radians(self.state['angle'])+math.pi-self.hypotenuse_angle)])
    rf = self.state['pos'] + np.array([self.car_hypotenuse*math.cos(math.radians(self.state['angle'])+2*math.pi-self.hypotenuse_angle),
                                                  self.car_hypotenuse*math.sin(math.radians(self.state['angle'])+2*math.pi-self.hypotenuse_angle)])
    lf = self.state['pos'] + np.array([self.car_hypotenuse*math.cos(math.radians(self.state['angle'])+math.pi+self.hypotenuse_angle),
                                                  self.car_hypotenuse*math.sin(math.radians(self.state['angle'])+math.pi+self.hypotenuse_angle)])

    return rf, lf, rb, lb
  
  def distance_to_parking(self, car_pos, park_pos):
    return np.linalg.norm([car_pos[0]-park_pos[0], car_pos[1]-park_pos[1]])

  def step(self, action):
    if not self.clock:
      self.clock = pygame.time.Clock()
    reward = 0
    accelerate = True
    # timestep = self.clock.get_time() / 1000
    self.current_iterations += 1
    action = np.array([int((action-1)/3)+1,((action-1)%3)+1]) 
    if action[0]==1:
      
      self.state['steering']+=self.max_steering*self.timestep
    elif action[0]==2:
      self.state['steering']-=self.max_steering*self.timestep
    else:
      pass

    if self.state['steering'] != max(-self.max_steering, min(self.state['steering'], self.max_steering)):
      reward -= 1
      self.state['steering'] = max(-self.max_steering, min(self.state['steering'], self.max_steering))

    if action[1]==1:
      if self.state['velocity']>=0:
        
        self.state['acceleration'] += random.uniform(self.max_acceleration-0.05,self.max_acceleration+0.05) * self.timestep * 0.005
      else:
        accelerate=False
        if self.state['acceleration'] < 0:
          self.state['acceleration'] = 0
        self.state['acceleration'] += random.uniform(self.max_decceleration-0.1,self.max_decceleration+0.1) * self.timestep * 0.005
        if abs(self.state['velocity']) < abs(self.state['acceleration']):
          self.state['acceleration'] = -self.state['velocity']
    elif action[1]==2:
      if self.state['velocity']<=0:
        self.state['acceleration'] -= random.uniform(self.max_acceleration-0.05,self.max_acceleration+0.05) * self.timestep * 0.005
      else:
        accelerate=False
        if self.state['acceleration'] > 0:
          self.state['acceleration'] = 0
        self.state['acceleration'] -= random.uniform(self.max_decceleration-0.1,self.max_decceleration+0.1) * self.timestep * 0.005
        if abs(self.state['velocity']) < abs(self.state['acceleration']):
          self.state['acceleration'] = -self.state['velocity']
    else: #action[1]==3:
      self.state['acceleration']=0
      # accelerate=False
      # if abs(self.state['velocity']) > timestep * natural_decceleration:
      #     self.state['acceleration']= -math.copysign(natural_decceleration, self.state['acceleration'])
      # else:
      #     self.state['acceleration'] = -self.state['velocity']

    if accelerate:
      if self.state['acceleration'] != max(-self.max_acceleration, min(self.state['acceleration'], self.max_acceleration)):
        reward-=1 #unecessary action
      self.state['acceleration'] = max(-self.max_acceleration, min(self.state['acceleration'], self.max_acceleration))
    else:
      if self.state['acceleration'] != max(-self.max_decceleration, min(self.state['acceleration'], self.max_decceleration)):
        reward -= 1
      self.state['acceleration'] != max(-self.max_decceleration, min(self.state['acceleration'], self.max_decceleration))

    self.process_new_position(reward)
    self.wheels['rf'], self.wheels['lf'], self.wheels['rb'], self.wheels['lb'] = self.calculate_four_corners()
    self.state['distances'] = self.calculate_distances()
    self.state['angle_from_parking'] = self.calculate_angle_from_parking()
    # calculate reward
    reward -= 1
    reward += 500/self.distance_to_parking(self.state['pos'], np.array(self.goal_center))

    # check if done
    done, reward_gain = self.check_if_inside()
    if done and reward_gain < -1000:
      reward = reward_gain
    else:
      reward += reward_gain


    
    info = {}
    coords = None
    self.clock.tick(self.fps)
    if self.current_iterations == self.max_iterations:
      done = True

    return self.state, reward, done, info, coords
  
  def render(self):
    if not self.screen:
      pygame.init()
      pygame.display.init()
      self.screen = pygame.display.set_mode((self.area_length, self.area_width), pygame.FULLSCREEN | pygame.SCALED)
      
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

    
    rotated = pygame.transform.rotate(self.car_image, -self.state['angle'])
    rotated_rect=rotated.get_rect()
    rotated_rect.center = self.state['pos']
    self.screen.blit(rotated, rotated_rect)
    pygame.draw.circle(self.screen,(0,255,0),self.state['pos'],5)
    pygame.draw.circle(self.screen,(0,255,255),self.goal_center,10)
    for part in ['rf','lf','rb','lb']:
      pygame.draw.circle(self.screen, (255,0,0), self.wheels[part], 5)
    pygame.display.update()

  def set_vars(self):
    x_noise = random.uniform(-(self.road_width-self.car_width)/2,(self.road_width-self.car_width)/2)
    y_noise = random.uniform(0,50)
    self.state = OrderedDict({
        'velocity': 0
        ,'acceleration': 0
        ,'angle': 90
        ,'pos': np.array([(self.car_length/2),self.parking_length+(self.road_width/2)-1])
        ,'steering': 0
        ,'wheels_inside': 0}) 
    self.goal_number = random.randint(0,15)
    self.wheels = {}
    self.wheels['rf'], self.wheels['lf'], self.wheels['rb'], self.wheels['lb'] = self.calculate_four_corners()
    self.current_iterations = 0
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
    self.state['distances'] = self.calculate_distances()
    #get angle
    self.goal_center = np.array(np.mean(self.goal, axis=0))
    self.state['angle_from_parking'] = self.calculate_angle_from_parking()
    self.normalisation_table = {
      'velocity': [self.max_velocity,self.max_velocity*2], #plus by min, divide by range
      'acceleration': [self.max_acceleration, self.max_acceleration*2],
      'angle': [0, 360],
      'pos_x': [0, self.area_length],
      'pos_y': [0, self.area_width],
      'steering': [self.max_steering, self.max_steering * 2],
      'wheels_inside': [0, 4],
      'distances': [0, np.linalg.norm([self.area_length, self.area_width])],
      'angle_from_parking': [180, 360]
    }

  def calculate_distances(self):
    distances = np.zeros(4)
    for i, wheel in enumerate(['rf','lf','rb','lb']):
      distances[i] = np.linalg.norm(self.wheels[wheel]-np.mean(self.goal[:2],axis=0))
    return np.array(distances)

  def calculate_angle_from_parking(self):
    return math.degrees(np.arctan2(self.goal_center[1] - self.state['pos'][1], abs(self.goal_center[0] - self.state['pos'][0])))


  def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    self.set_vars()
    return self.state
  
  def convert_actions(self, action):
    return action[1] + ((action[0]-1)*3)
  
  def deconstruct_array(self,arr):
    llist = []
    for key in ['velocity','acceleration','angle','pos','steering','wheels_inside','distances','angle_from_parking']:
      if key =='pos':
        new_arr = arr[key].flatten()
        new_arr[0] = (new_arr[0]+self.normalisation_table['pos_x'][0])/self.normalisation_table['pos_x'][1]
        new_arr[1] = (new_arr[1]+self.normalisation_table['pos_y'][0])/self.normalisation_table['pos_y'][1]
        llist.extend(new_arr.tolist())
      elif key == 'distances':
        new_arr =(arr[key].flatten()+self.normalisation_table[key][0])/self.normalisation_table[key][1]
        llist.extend(new_arr.tolist())
      else:
        llist.append((arr[key]+self.normalisation_table[key][0])/self.normalisation_table[key][1])
        
    return np.array(llist)
  
env = CarParking()