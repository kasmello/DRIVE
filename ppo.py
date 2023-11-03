
import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from environment import CarParking

env = CarParking(60, 1_000)

stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=4, min_evals=100, verbose=1)
eval_callback = EvalCallback(env, best_model_save_path="./logs/",
                             log_path="./logs/", eval_freq=5_000,
                             deterministic=True, render=False, verbose=1, callback_after_eval=stop_train_callback)

model = PPO("MlpPolicy", env, verbose=1, learning_rate=1e-3, batch_size=64, policy_kwargs=dict(net_arch=[64, 64]))
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