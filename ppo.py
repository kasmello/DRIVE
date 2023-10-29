import gymnasium as gym

import os
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
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

env = CarParking(60, 1_000)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000)
model.save("ppo_drive")


model = PPO.load("ppo_drive")

obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    state, reward, done, info = env.step(action)
    env.render()