import os
import time
import argparse

import pybullet_envs
from stable_baselines3 import PPO, DDPG, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

_model = PPO

# ENV_NAME = "CartPole-v1"
ENV_NAME = "HalfCheetahBulletEnv-v0"
log_dir = "/tmp/"
stats_path = os.path.join(log_dir, ENV_NAME + "vec_normalize.pkl")

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--training', default=False, action='store_true', help='Whether or not you want to do training')
parser.add_argument('-l', '--load_previous', default=False, action='store_true', help='Whether or not you want to resume training from previous save')
parser.add_argument('-T', '--train_steps', default=1000000, type=int)
parser.add_argument('-n', '--num_envs', default=1, type=int)

args = parser.parse_args()

training = args.training
loadPrevious = args.load_previous
print(args.load_previous)
if training:
    env = make_vec_env(ENV_NAME, n_envs=args.num_envs)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = (
        _model.load(log_dir + str(_model.__module__) + ENV_NAME, env)
        if loadPrevious
        else _model("MlpPolicy", env, verbose=1)
    )
    model.learn(total_timesteps=args.train_steps)

    # Don't forget to save the VecNormalize statistics when saving the agent
    model.save(log_dir + str(_model.__module__) + ENV_NAME)
    env.save(stats_path)

# Load the agent
model = _model.load(log_dir + str(_model.__module__) + ENV_NAME)

# Load the saved statistics
env = make_vec_env(ENV_NAME, n_envs=1, env_kwargs={"render": True})
# env = make_vec_env(ENV_NAME, n_envs=1)
env = VecNormalize.load(stats_path, env)
#  do not update them at test time
env.training = False
# reward normalization is not needed at test time
env.norm_reward = False
env.render()
obs = env.reset()
for i in range(1000):
    # action, _state = model.predict(obs, deterministic=False)
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    # print(action, _state)
    time.sleep(1 / 24)
    # env.render()
    if done:
        print("Done now!")
        obs = env.reset()

from stable_baselines3.common.evaluation import evaluate_policy

mean_reward, std_reward = evaluate_policy(model, env)

print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")