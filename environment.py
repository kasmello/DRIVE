
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
    # 'velocity','acceleration','angle','pos','steering','wheels_inside','distances'
    # self.observation_space=Dict({
    #     'velocity': Box(low=np.array([-self.max_velocity]),high=np.array([self.max_velocity]))
    #     ,'acceleration': Box(low=np.array([-self.max_acceleration]),high=np.array([self.max_acceleration]))
    #     ,'angle': Box(low=np.array([0]),high=np.array([359])) #angle of car
    #     ,'pos': Box(low=np.array([0, 0]), high=np.array([self.area_length-1, self.area_width-1])) #calculate 4 corners of car using trigonometry
    #     ,'steering': Discrete(61,start=-30)#angle of steering wheel
    #     ,'wheels_inside': Discrete(5,start=0)
    #     ,'distances': Box(low=np.array([0,0]),high=np.array([self.parking_length, self.parking_width]))
    #     })
    self.observation_space=Box(
      low = np.array([0,0,0,0,0,0,0,0,0]),
      high = np.array([1,1,1,1,1,1,1,1,1])
    )
    self.set_vars()


  def check_if_inside(self):
    goal = True
    wheels_inside = 0
    reward = 0
    for wheel in ['lb','rb','lf','rf']:
      if self.wheels[wheel][0] < 0 or self.wheels[wheel][0] >= self.area_length or self.wheels[wheel][1] < 0 or self.wheels[wheel][1] >= self.area_width:
        return True,-10000
      if self.goal_number<8:
        if self.wheels[wheel][0] < self.goal[0][0] or self.wheels[wheel][0] > self.goal[1][0] or self.wheels[wheel][1] < self.goal[0][1] or self.wheels[wheel][1] > self.goal[2][1]:
          goal = False
        else:
          wheels_inside += 1
          reward += 5
      else:
        if self.wheels[wheel][0] < self.goal[0][0] or self.wheels[wheel][0] > self.goal[1][0] or self.wheels[wheel][1] > self.goal[0][1] or self.wheels[wheel][1] < self.goal[2][1]:
          goal = False
        else:
          wheels_inside += 1
          reward += 5

    if self.state_labelled['velocity'] != 0:
      goal = False

    self.state_labelled['wheels_inside'] = wheels_inside
    if goal:
      return True, 10000
    return False, reward

  def close(self):
    if self.screen:

      pygame.quit()
      self.screen=None
      self.clock=None

  def process_new_position(self, reward):
      self.state_labelled['velocity'] += self.state_labelled['acceleration'] /10
      if abs(self.state_labelled['velocity']) > self.speed_limit:
        reward -= 1
      self.state_labelled['velocity'] = max(-self.max_velocity, min(self.state_labelled['velocity'], self.max_velocity))
      if abs(self.state_labelled['velocity'])<0.01:
        self.state_labelled['velocity']=0

      if self.state_labelled['steering']:
        turning_radius = self.car_wheelbase / math.sin(math.radians(self.state_labelled['steering']))
        angular_velocity = self.state_labelled['velocity'] / turning_radius
      else:
        angular_velocity = 0
      self.state_labelled['angle'] = (self.state_labelled['angle'] + math.degrees(angular_velocity)*self.timestep)%360
      if self.state_labelled['angle'] < 0:
        self.state_labelled['angle']+= 360
      self.state_labelled['pos'] += np.array([self.state_labelled['velocity'] * math.sin(math.radians(self.state_labelled['angle']))*self.timestep,-self.state_labelled['velocity'] * math.cos(math.radians(self.state_labelled['angle']))*self.timestep])
      return reward
      
      

  def calculate_four_corners(self):
    rb = self.state_labelled['pos'] + np.array([self.car_hypotenuse*math.cos(math.radians(self.state_labelled['angle'])+self.hypotenuse_angle),
                                                  self.car_hypotenuse*math.sin(math.radians(self.state_labelled['angle'])+self.hypotenuse_angle)])
    lb = self.state_labelled['pos'] + np.array([self.car_hypotenuse*math.cos(math.radians(self.state_labelled['angle'])+math.pi-self.hypotenuse_angle),
                                                  self.car_hypotenuse*math.sin(math.radians(self.state_labelled['angle'])+math.pi-self.hypotenuse_angle)])
    rf = self.state_labelled['pos'] + np.array([self.car_hypotenuse*math.cos(math.radians(self.state_labelled['angle'])+2*math.pi-self.hypotenuse_angle),
                                                  self.car_hypotenuse*math.sin(math.radians(self.state_labelled['angle'])+2*math.pi-self.hypotenuse_angle)])
    lf = self.state_labelled['pos'] + np.array([self.car_hypotenuse*math.cos(math.radians(self.state_labelled['angle'])+math.pi+self.hypotenuse_angle),
                                                  self.car_hypotenuse*math.sin(math.radians(self.state_labelled['angle'])+math.pi+self.hypotenuse_angle)])

    return rf, lf, rb, lb
  

  def step(self, action):
    if not self.clock:
      self.clock = pygame.time.Clock()
    reward = 0
    opposite = False
    terminated = False
    # timestep = self.clock.get_time() / 1000
    action = np.array([int((action)/3)+1,((action)%3)+1]) 

    if action[0]==1:
      self.state_labelled['steering']+=self.max_steering*(self.timestep/6) 
    elif action[0]==2:
      self.state_labelled['steering']-=self.max_steering*(self.timestep/6)
    else:
      if self.state_labelled['steering'] < 0:
        self.state_labelled['steering'] +=self.max_steering*(self.timestep/6)
        self.state_labelled['steering'] = min(self.state_labelled['steering'],0)
      elif self.state_labelled['steering'] > 0:
        self.state_labelled['steering'] -= self.max_steering*(self.timestep/6)
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
    
    if not opposite:
      self.state_labelled['acceleration'] = max(-self.max_acceleration, min(self.state_labelled['acceleration'], self.max_acceleration))
    reward = self.process_new_position(reward)
    self.wheels['rf'], self.wheels['lf'], self.wheels['rb'], self.wheels['lb'] = self.calculate_four_corners()
    self.state_labelled['distances'] = self.calculate_distances()
    # calculate reward
    reward -= 2
    euc_dis = np.linalg.norm(self.state_labelled['distances'])
    if euc_dis < 500 and abs(self.state_labelled['velocity'])>0:
      velocity_factor = 100*abs(self.state_labelled['velocity'])
      reward += 800/(euc_dis+velocity_factor)
    else:
      
      reward += 500/(euc_dis)


    # check if done
    terminated, reward_gain = self.check_if_inside()
    if terminated and abs(reward_gain) > 1000:
      reward = reward_gain
    else:
      reward += reward_gain

    
    info = {}
    self.clock.tick(self.fps)
    self.last_reward = reward
    self.state = self.deconstruct_array(self.state_labelled)
    return self.state, reward, terminated, info
  
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
    self.screen.blit(vel_text, (1000, 10))
    self.screen.blit(acc_text, (1000, 100))
    self.screen.blit(steer_text, (1000, 200))
    self.screen.blit(wheels_text, (1000, 300))
    self.screen.blit(distance_text, (1000, 400))
    self.screen.blit(pos_text, (1000, 500))
    self.screen.blit(reward_text, (1000, 600))

    
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

  def set_vars(self):
    x_noise = random.uniform(-(self.road_width-self.car_width)/2,(self.road_width-self.car_width)/2)
    y_noise = random.uniform(0,50)
    self.state_labelled = OrderedDict({
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
    self.goal_center = np.array(np.mean(self.goal, axis=0))
    self.state_labelled['distances'] = self.calculate_distances()
    self.last_reward = 0
    #get angle
    
    
    self.normalisation_table = {
      'velocity': [self.max_velocity,self.max_velocity*2], #plus by min, divide by range
      'acceleration': [self.max_acceleration, self.max_acceleration*2],
      'angle': [0, 360],
      'pos_x': [0, self.area_length],
      'pos_y': [0, self.area_width],
      'steering': [self.max_steering, self.max_steering * 2],
      'wheels_inside': [0, 4],
      'distances_x': [0, self.area_length],
      'distances_y': [0, self.area_width]
    }

  def calculate_distances(self): #return array of 2 (center of car - center of goal)
    distances = np.zeros(2)
    x = self.goal_center[0]
    y =self.goal_center[1]
    distances[0] = self.state_labelled['pos'][0]-x
    distances[1] = self.state_labelled['pos'][1]-y
    return distances

  def calculate_angle_from_parking(self):
    return math.degrees(np.arctan2(self.goal_center[1] - self.state_labelled['pos'][1], abs(self.goal_center[0] - self.state_labelled['pos'][0])))


  def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    self.set_vars()
    return self.state_labelled
  
  def convert_actions(self, action):
    return action[1] + (action[0]*3)
  
  def deconstruct_array(self,arr):
    llist = []
    for key in ['velocity','acceleration','angle','pos','steering','wheels_inside','distances']:
      if key in ['pos', 'distances']:
        new_arr = arr[key].flatten()
        new_arr[0] = (new_arr[0]+self.normalisation_table[f'{key}_x'][0])/self.normalisation_table[f'{key}_x'][1]
        new_arr[1] = (new_arr[1]+self.normalisation_table[f'{key}_y'][0])/self.normalisation_table[f'{key}_y'][1]
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

