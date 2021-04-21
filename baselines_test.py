import os
import time
import argparse

import pybullet_envs
from stable_baselines3 import PPO, DDPG, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

ENV_NAME = "HopperBulletEnv-v0"
MODEL_PREFIX = "a2c"
_model = A2C

stats_path = f"./{MODEL_PREFIX}_{ENV_NAME}_stats.pkl"

parser = argparse.ArgumentParser()
parser.add_argument(
    "-t",
    "--training",
    default=False,
    action="store_true",
    help="Whether or not you want to do training",
)
parser.add_argument(
    "-l",
    "--load_previous",
    default=False,
    action="store_true",
    help="Whether or not you want to resume training from previous save",
)
parser.add_argument("-s", "--save_freq", default=1000, type=int)
parser.add_argument("-T", "--train_steps", default=50000, type=int)
parser.add_argument("-n", "--num_envs", default=1, type=int)

args = parser.parse_args()

training = args.training
loadPrevious = args.load_previous

if training:
    env = make_vec_env(ENV_NAME, n_envs=args.num_envs)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = (
        _model.load(f"./{MODEL_PREFIX}_{ENV_NAME}_model", env)
        if loadPrevious
        else _model(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=f"./{MODEL_PREFIX}_{ENV_NAME}_tensorboard/",
        )
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=f"./{MODEL_PREFIX}_{ENV_NAME}_checkpoints/",
        name_prefix=f"{MODEL_PREFIX}_{ENV_NAME}",
    )
    model.learn(total_timesteps=args.train_steps, callback=checkpoint_callback)

    # Don't forget to save the VecNormalize statistics when saving the agent
    model.save(f"./{MODEL_PREFIX}_{ENV_NAME}_model")
    env.save(stats_path)
    exit()


# Load the agent
model = _model.load(f"./{MODEL_PREFIX}_{ENV_NAME}_model")

# Load the saved statistics
env = make_vec_env(ENV_NAME, n_envs=1, env_kwargs={"render": True})
env = VecNormalize.load(stats_path, env)
#  do not update them at test time
env.training = False
# reward normalization is not needed at test time
env.norm_reward = False
obs = env.reset()
env.render()

for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    time.sleep(1 / 60)
    if done:
        obs = env.reset()

from stable_baselines3.common.evaluation import evaluate_policy

mean_reward, std_reward = evaluate_policy(model, env)

print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")