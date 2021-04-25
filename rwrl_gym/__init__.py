from gym.envs.registration import register
from rwrl_gym.rwrl import RWRL

import os 
from shutil import copyfile

model_xml = "rwrl.xml"
try:
	copyfile(os.path.join(os.path.dirname(__file__), model_xml), \
		os.path.join(os.path.dirname(__file__), "..", "pybullet-gym", "pybulletgym", "envs", "assets", "mjcf", model_xml))
except Exception as e:
	print(e)

register(
	id='rwrl-v0', 
	entry_point='rwrl_gym.rwrl_env:RWRLEnv',
	max_episode_steps=1000,
	reward_threshold=3000.0
)