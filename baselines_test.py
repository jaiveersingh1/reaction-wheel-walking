import os
import time

import pybullet_envs
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

ENV_NAME = "HopperBulletEnv-v0"
log_dir = "/tmp/"
stats_path = os.path.join(log_dir, ENV_NAME + "vec_normalize.pkl")

training = True
loadPrevious = True
if training:
    env = make_vec_env(ENV_NAME, n_envs=1)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = (
        PPO.load(log_dir + "ppo_" + ENV_NAME)
        if loadPrevious
        else PPO("MlpPolicy", env, verbose=1)
    )
    model.learn(total_timesteps=50000)

    # Don't forget to save the VecNormalize statistics when saving the agent
    model.save(log_dir + "ppo_" + ENV_NAME)
    env.save(stats_path)

# Load the agent
model = PPO.load(log_dir + "ppo_" + ENV_NAME)

# Load the saved statistics
env = make_vec_env(ENV_NAME, n_envs=1, env_kwargs={"render": True})
env = VecNormalize.load(stats_path, env)
#  do not update them at test time
env.training = False
# reward normalization is not needed at test time
env.norm_reward = False

obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    time.sleep(1 / 24)
    if done:
        print("Done now!")
        obs = env.reset()

from stable_baselines3.common.evaluation import evaluate_policy

mean_reward, std_reward = evaluate_policy(model, env)

print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")