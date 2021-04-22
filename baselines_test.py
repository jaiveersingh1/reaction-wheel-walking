import os
import time
import argparse
from gym.envs.registration import register

import pybulletgym
import pybullet_envs
from stable_baselines3 import PPO, DDPG, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# register(
# 	id='rwrl-v0', 
# 	entry_point='pybulletgym.envs.roboschool.envs.locomotion.rwrl_env:RWRLEnv',
# 	max_episode_steps=1000,
# 	reward_threshold=3000.0
# )

# ENV_NAME = "CartPole-v1"
# ENV_NAME = "HalfCheetahBulletEnv-v0"
ENV_NAME = "rwrl-v0"
_model = PPO
MODEL_PREFIX = "PPO"
STATS_PATH = f"./{MODEL_PREFIX}_{ENV_NAME}_stats.pkl"
BEST_MODEL_PATH = f"./{MODEL_PREFIX}_{ENV_NAME}_best"

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
parser.add_argument("-T", "--train_steps", default=100000, type=int)
parser.add_argument("-n", "--num_envs", default=1, type=int)

args = parser.parse_args()

training = args.training
loadPrevious = args.load_previous

if training:
    env = make_vec_env(ENV_NAME, n_envs=args.num_envs)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = make_vec_env(ENV_NAME, n_envs=args.num_envs)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = (
        _model.load(BEST_MODEL_PATH + "/best_model", env)
        if loadPrevious
        else _model(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=f"./{MODEL_PREFIX}_{ENV_NAME}_tensorboard/",
        )
    )

    # Use deterministic actions for evaluation
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_PATH,
        eval_freq=args.save_freq,
        deterministic=True,
        render=False,
    )
    model.learn(total_timesteps=args.train_steps, callback=eval_callback)

    # Don't forget to save the VecNormalize statistics when saving the agent
    env.save(STATS_PATH)
    exit()


# Load the agent
model = _model.load(BEST_MODEL_PATH + "/best_model")

# Load the saved statistics
env = make_vec_env(ENV_NAME, n_envs=1, env_kwargs={"render": True})
env = VecNormalize.load(STATS_PATH, env)
#  do not update them at test time
env.training = False
# reward normalization is not needed at test time
env.norm_reward = False
obs = env.reset()
total_reward = 0
for i in range(1000):
    # action, _state = model.predict(obs, deterministic=False)
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    total_reward += reward
    env.render()
    time.sleep(1 / 60)
    if done:
        obs = env.reset()
        print("Total reward:", total_reward)
        total_reward = 0

from stable_baselines3.common.evaluation import evaluate_policy

mean_reward, std_reward = evaluate_policy(model, env)
print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")