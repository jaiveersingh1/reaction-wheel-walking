import os
import time
import argparse

from stable_baselines3 import PPO, DDPG, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from legmo.envs.legmo_env import LegMoEnv

ENV_NAME = "legmo-v0"
MODEL_TYPE = PPO
MODEL_PREFIX = "PPO"

DATA_FOLDER = f"./data/{ENV_NAME}/{MODEL_PREFIX}"
STATS_PATH = f"{DATA_FOLDER}/stats.pkl"
BEST_MODEL_LOAD_PATH = f"{DATA_FOLDER}/best_model.zip"
TENSORBOARD_PATH = f"{DATA_FOLDER}/tensorboard/"


def main(args):

    # Only train model if requested to do so
    if args.training:
        # Create the environment
        env = VecNormalize(
            make_vec_env(LegMoEnv, args.num_envs, env_kwargs={"render": False}),
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
        )

        model = (
            MODEL_TYPE.load(BEST_MODEL_LOAD_PATH, env)
            if args.load_previous
            else MODEL_TYPE(
                "MlpPolicy",
                env,
                n_steps=256,
                verbose=1,
                tensorboard_log=TENSORBOARD_PATH,
            )
        )

        # Periodically evaluate the agent's performance with
        # deterministic actions, and save best agent
        eval_callback = EvalCallback(
            env,
            best_model_save_path=DATA_FOLDER,
            eval_freq=args.save_freq,
            deterministic=True,
            render=False,
        )

        # Train the model for the desired steps, evaluating periodically
        model.learn(total_timesteps=args.train_steps, callback=eval_callback)

        # Save normalized statistics from environment as well
        env.save(STATS_PATH)

        # Disconnect from GUI server
        env.close()

    # After training (or if training is skipped), test best agent
    test_env = VecNormalize.load(
        STATS_PATH,
        make_vec_env(LegMoEnv, n_envs=1, env_kwargs={"render": True}),
    )
    test_model = MODEL_TYPE.load(BEST_MODEL_LOAD_PATH, test_env)

    test_env.training = False
    test_env.norm_reward = False

    total_reward = 0
    obs = test_env.reset()
    for i in range(args.test_steps):
        # Predict action given observation
        action, _state = test_model.predict(obs, deterministic=True)

        # Receive observation, reward, and metadata by stepping forward
        obs, reward, done, info = test_env.step(action)
        test_env.render()

        total_reward += reward

        # If done before timesteps expire, reset and continue
        if done:
            print("Total reward:", total_reward)
            time.sleep(1)
            obs = test_env.reset()
            total_reward = 0

    # Evaluate using SB3 evaluation function
    mean_reward, std_reward = evaluate_policy(test_model, test_env)
    print(f"Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")


if __name__ == "__main__":
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
    parser.add_argument("-T", "--train_steps", default=10000, type=int)
    parser.add_argument("-E", "--test_steps", default=500, type=int)
    parser.add_argument("-n", "--num_envs", default=1, type=int)

    args = parser.parse_args()

    main(args)