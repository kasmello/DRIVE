import numpy as np
import time
import random
import os
from datetime import datetime
from agent import ActorCritic
from environment import CarParking

# Making folder for end frames:

# Get the current timestamp
current_time = datetime.now()
# Format the timestamp as a string (e.g., "2023-10-24_15-30-45")
timestamp_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
# Specify the path where you want to create the folder
folder_path = os.path.join(os.getcwd(), timestamp_str)
# Create the folder
os.makedirs(folder_path)
print(f"Created folder: {folder_path}")

env = CarParking(60)

gamma = 0.99
hidden_sizes = (64, 64)
lr_policy = 0.001
lr_value = 0.005
max_episodes = 10000
max_steps = 10000
criterion_episodes = 5

agent = ActorCritic(env, gamma=gamma, hidden_sizes=hidden_sizes, lr_policy=lr_policy, lr_value=lr_value)

agent.train(max_episodes, 100, folder_path)

state = env.reset()
done = False
steps = 0
total_reward = 0



while not (done or steps > max_steps):
    # take action based on policy
    action = agent.policy(state, stochastic=False)

    # environment receives the action and returns:
    # next observation, reward, terminated, truncated, and additional information (if applicable)
    state, reward, done, info = env.step(action)
    env.render()
    
    total_reward += reward
    steps += 1

print(f'Reward: {total_reward}')

# store RGB frames for the entire episode

# close the environment
env.close()

# create and play video clip using the frames and given fps
