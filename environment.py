
import pygame
import numpy as np
import math
import random
import time
from gym import Env
from collections import OrderedDict
from gym.spaces import Discrete, Box, Dict

class CarParking(Env):
  
  def __init__(self, fps):
    # 1=right, 2=left, 3=stay
    # 1=accelerate, 2=deccelerate, 3=same speed and direction

    self.screen = None #pygame.display.set_mode((area_length, area_width))
    self.clock = None #pygame.time.Clock()
    self.car_width = 150
    self.car_length = 400
    self.fps = fps
    self.timestep = 1
    self.car_hypotenuse = np.linalg.norm([self.car_length/2, self.car_width/2])
    self.hypotenuse_angle = math.acos(self.car_width/(2*self.car_hypotenuse))
    self.car_wheelbase=250
    self.parking_width = 240
    self.parking_length = 450
    self.road_width = 300
    self.area_width = (2 * self.road_width) + (2 * self.parking_length) #1300
    self.area_length = 8 * self.parking_width #1920
    self.parking_hypotenuse = np.linalg.norm([self.area_width,self.area_length])

    self.max_steering=30
    self.max_velocity=30 #in m/s
    self.speed_limit=10
    self.max_acceleration=3.4 #around 12kmh/s
    self.natural_decceleration=1
    self.car_image = pygame.image.load('car_sprite.png')
    self.car_image = pygame.transform.scale(self.car_image, (self.car_width, self.car_length))
    self.action_space=Box(low=np.array([1,1]),high=np.array([3,3]),dtype=int)
    # 'velocity','acceleration','angle','pos','steering','wheels_inside','distances','angle_from_parking'
    self.observation_space=Dict({
        'velocity': Box(low=np.array([-self.max_velocity]),high=np.array([self.max_velocity]))
        ,'acceleration': Box(low=np.array([-self.max_acceleration]),high=np.array([self.max_acceleration]))
        ,'angle': Box(low=np.array([0]),high=np.array([359])) #angle of car
        ,'pos': Box(low=np.array([0, 0]), high=np.array([self.area_length-1, self.area_width-1])) #calculate 4 corners of car using trigonometry
        ,'steering': Discrete(61,start=-30)#angle of steering wheel
        ,'wheels_inside': Discrete(5,start=0)
        ,'distances': Box(low=np.array([0,0,0,0]),high=np.array([self.parking_hypotenuse, self.parking_hypotenuse, self.parking_hypotenuse, self.parking_hypotenuse]))
        ,'angle_from_parking': Box(low=np.array([0]),high=np.array([359]))
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
      self.state['velocity'] += self.state['acceleration'] /4
      if abs(self.state['velocity']) > self.speed_limit:
        reward -= 1
      self.state['velocity'] = max(-self.max_velocity, min(self.state['velocity'], self.max_velocity))

      if self.state['steering']:
        turning_radius = self.car_wheelbase / math.sin(math.radians(self.state['steering']))
        angular_velocity = self.state['velocity'] / turning_radius
      else:
        angular_velocity = 0
      self.state['angle'] = (self.state['angle'] + math.degrees(angular_velocity)*self.timestep/2)%360
      if self.state['angle'] < 0:
        self.state['angle']+= 360
      self.state['pos'] += np.array([self.state['velocity'] * math.sin(math.radians(self.state['angle']))*self.timestep/2,-self.state['velocity'] * math.cos(math.radians(self.state['angle']))*self.timestep/2])
      
      

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
    opposite = False
    terminated = False
    # timestep = self.clock.get_time() / 1000
    action = np.array([int((action-1)/3)+1,((action-1)%3)+1]) 

    if action[0]==1:
      self.state['steering']+=self.max_steering*(self.timestep/6) 
    elif action[0]==2:
      self.state['steering']-=self.max_steering*(self.timestep/6)
    else:
      if self.state['steering'] < 0:
        self.state['steering'] +=self.max_steering*(self.timestep/6)
        self.state['steering'] = min(self.state['steering'],0)
      elif self.state['steering'] > 0:
        self.state['steering'] -= self.max_steering*(self.timestep/6)
        self.state['steering'] = max(self.state['steering'],0)


    if self.state['steering'] != max(-self.max_steering, min(self.state['steering'], self.max_steering)):
      reward -= 1
      self.state['steering'] = max(-self.max_steering, min(self.state['steering'], self.max_steering))

    if action[1]==1:
      if self.state['acceleration']<0:
        opposite=True
        self.state['acceleration'] = 0

      self.state['acceleration'] += self.max_acceleration * (self.timestep/6) #15 second to reach full acceleration from 0

    elif action[1]==2:
      if self.state['acceleration']>0:
        opposite=True
        self.state['acceleration'] =0
 
      self.state['acceleration'] -= self.max_acceleration * (self.timestep/6)  #15 seconds to reach full acceleration from 0

    else: #action[1]==3:
      if self.state['velocity']<0:
        self.state['acceleration'] = min(self.max_acceleration * (self.timestep/6),-self.state['velocity'])

      elif self.state['velocity']>0:
        self.state['acceleration'] = max(-self.max_acceleration * (self.timestep/6),-self.state['velocity'])



    if not opposite:
      if self.state['acceleration'] != max(-self.max_acceleration, min(self.state['acceleration'], self.max_acceleration)):
        reward-=1 #unecessary action
      self.state['acceleration'] = max(-self.max_acceleration, min(self.state['acceleration'], self.max_acceleration))

    self.process_new_position(reward)
    self.wheels['rf'], self.wheels['lf'], self.wheels['rb'], self.wheels['lb'] = self.calculate_four_corners()
    self.state['distances'] = self.calculate_distances()
    self.state['angle_from_parking'] = self.calculate_angle_from_parking()
    # calculate reward
    reward -= 1
    reward += 500/self.distance_to_parking(self.state['pos'], np.array(self.goal_center))

    # check if done
    terminated, reward_gain = self.check_if_inside()
    if terminated and reward_gain < -1000:
      reward = reward_gain
    else:
      reward += reward_gain


    
    info = {}
    self.clock.tick(self.fps)

    return self.state, reward, terminated, info
  
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


    font = pygame.font.Font(None, 36)
    vel_text = font.render(str(self.state['velocity']), True, (255, 0, 0)) 
    acc_text = font.render(str(self.state['acceleration']), True, (255, 0, 0)) 
    steer_text = font.render(str(self.state['steering']), True, (255, 0, 0)) 
    self.screen.blit(vel_text, (1000, 10))
    self.screen.blit(acc_text, (1000, 100))
    self.screen.blit(steer_text, (1000, 200))

    
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
  


if __name__ == '__main__':
  fps = 60
  env = CarParking(fps)
  state = env.reset()
  score = 0
  start = time.time()
  # for i in range(10000):
  #     action = env.action_space.sample()
  #     action = np.array([3,1])
  #     if i <1000:
  #         action = np.array([3,1])
  #     state, reward, done, info, coords = env.step(env.convert_actions(action))
  #     env.render()
  #     score += reward
  #     if done:
  #         env.close()
  #         print('CAR CRASH')
  #         break
  env.render()
  done = False
  i = 0
  while not done:
    i += 1
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        done = True
        break

    driving = 3
    steering = 3
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
      driving = 1
    elif keys[pygame.K_s]:
      driving = 2


    if keys[pygame.K_a]:
      steering = 2
    elif keys[pygame.K_d]:
      steering = 1

    action = np.array([steering,driving])
    state, reward, done, info = env.step(env.convert_actions(action))
    env.render()
    score += reward
    if done:
      env.close()
      break
    # time.sleep(1/fps)
    if i == 2000000:
      done = True

  end = time.time()
  print(end-start)

