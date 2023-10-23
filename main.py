import numpy as np
import time
import random
from agent import PolicyNetwork, ValueNetwork, ActorCritic
from environment import CarParking

env = CarParking(60)

gamma = 0.99
hidden_sizes = (64, 64)
lr_policy = 0.001
lr_value = 0.005
max_episodes = 1000
max_steps = 10000
criterion_episodes = 5

agent = ActorCritic(env, gamma=gamma, hidden_sizes=hidden_sizes, lr_policy=lr_policy, lr_value=lr_value)

agent.train(max_episodes, 100)

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
    
    total_reward += reward
    steps += 1

print(f'Reward: {total_reward}')

# store RGB frames for the entire episode
frames = env.render()

# close the environment
env.close()

# create and play video clip using the frames and given fps
