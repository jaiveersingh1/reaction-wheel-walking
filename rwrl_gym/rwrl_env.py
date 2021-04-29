from pybulletgym.envs.roboschool.envs.locomotion.walker_base_env import WalkerBaseBulletEnv
from rwrl_gym.rwrl import RWRL


class RWRLEnv(WalkerBaseBulletEnv):
    def __init__(self, render=False):
        self.robot = RWRL()
        WalkerBaseBulletEnv.__init__(self, self.robot, render)
