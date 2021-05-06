from pybulletgym.envs.roboschool.robots.locomotors.walker_base import WalkerBase
from pybulletgym.envs.roboschool.robots.robot_bases import MJCFBasedRobot
import numpy as np


class RWRL(WalkerBase, MJCFBasedRobot):
    foot_list = ["left_leg", "right_leg", "torso"]  # track these contacts with ground

    def __init__(self):
        WalkerBase.__init__(self, power=0.10)
        MJCFBasedRobot.__init__(self, "rwrl.xml", "torso", action_dim=5, obs_dim=21)

    def alive_bonus(self, z, pitch):
        # Use contact other than feet to terminate episode: due to a lot of strange walks using knees
        # return +1 if np.abs(pitch) < 1 and not self.feet_contact[2] and (self.feet_contact[0] or self.feet_contact[1]) and np.abs(z) < .35 else -1
        # return +1 if np.abs(pitch) < 1 and not self.feet_contact[2] and np.abs(z) < .35 else -1 # its swimming butterfly !
        # print(z)
        # print(pitch, z)
        return +1 if -1.3 < pitch < 1.5 and not self.feet_contact[2] and -.03 < z < .15 else -1

    def robot_specific_reset(self, bullet_client):
        WalkerBase.robot_specific_reset(self, bullet_client)
        self.jdict["x_joint"].power_coef = 200.0
        self.jdict["y_joint"].power_coef  = 200.0
        self.jdict["z_joint"].power_coef  = 200.0
        self.jdict["left_joint"].power_coef = 100
        self.jdict["right_joint"].power_coef  = 100