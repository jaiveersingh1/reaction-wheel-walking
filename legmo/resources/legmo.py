import pybullet as p
import numpy as np
import os


class LegMo:
    def __init__(self, client):
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__), "legmo.urdf")
        self.robot = p.loadURDF(
            fileName=f_name, basePosition=[0, 0, 0.125], physicsClientId=client
        )

        # Joint indices for right and left legs, respectively
        self.leg_joints = [0, 1]

        # Joint indices for roll, pitch, and yaw wheels, respectively
        self.rw_joints = [2, 3, 4]

    def get_ids(self):
        return self.client, self.robot

    def apply_action(self, action):
        # Expects R, P, Y, Right, Left motor values
        roll, pitch, yaw, right, left = action

        # Set the leg servomotors to appropriate positions
        p.setJointMotorControlArray(
            self.robot,
            self.leg_joints,
            controlMode=p.POSITION_CONTROL,
            targetPositions=[right, left],
            physicsClientId=self.client,
        )

        # Set the reaction wheel BLDC motors to appropriate velocities
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot,
            jointIndices=self.rw_joints,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=[roll, pitch, yaw],
            forces=[5000] * 3,  # TODO(JS): Verify that this force makes sense
            physicsClientId=self.client,
        )

    def get_observation(self):
        # Return global orientation of robot (as would be measured by IMU)
        pos, ori = p.getBasePositionAndOrientation(self.robot, self.client)
        observation = ori
        return observation