import os
import time

import pybullet_envs
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

ENV_NAME = "HopperBulletEnv-v0"
MODEL_PREFIX = "a2c"
_model = A2C

stats_path = f"./{MODEL_PREFIX}_{ENV_NAME}_stats.pkl"

training = False
loadPrevious = False
if training:
    env = make_vec_env(ENV_NAME, n_envs=1)
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
        save_freq=1000,
        save_path=f"./{MODEL_PREFIX}_{ENV_NAME}_checkpoints/",
        name_prefix=f"{MODEL_PREFIX}_{ENV_NAME}",
    )
    model.learn(total_timesteps=50000, callback=checkpoint_callback)

    # Don't forget to save the VecNormalize statistics when saving the agent
    model.save(f"./{MODEL_PREFIX}_{ENV_NAME}_model")
    env.save(stats_path)

# Load the agent
model = _model.load(f"./{MODEL_PREFIX}_{ENV_NAME}_model")

# Load the saved statistics
env = make_vec_env(ENV_NAME, n_envs=1, env_kwargs={"render": True})
env = VecNormalize.load(stats_path, env)
#  do not update them at test time
env.training = False
# reward normalization is not needed at test time
env.norm_reward = False
env.render()

obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    time.sleep(1 / 60)
    if done:
        obs = env.reset()

from stable_baselines3.common.evaluation import evaluate_policy

mean_reward, std_reward = evaluate_policy(model, env)

print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")