import gymnasium as gym

import os
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
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

stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=4, min_evals=100_000, verbose=1)
eval_callback = EvalCallback(env, best_model_save_path="./logs/",
                             log_path="./logs/", eval_freq=5_000,
                             deterministic=True, render=False, verbose=1)

model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.001)
model.learn(total_timesteps=2_000_000, progress_bar=True, callback=eval_callback)
model.save("ppo_drive")



model = PPO.load("ppo_drive", env=env)

obs, _states = env.reset()
terminated = False
truncated = False
while not (terminated or truncated):

    action, _states = model.predict(obs)

    state, reward, terminated, truncated, info = env.step(action)
    env.render()