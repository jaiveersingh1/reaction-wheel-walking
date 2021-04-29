import gym
import numpy as np
import pybullet as p


class LegMoEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        """
        Action space has:
        - Target Reaction Wheel Velocities  (R, P, Y)
        - Target Servomotors Positions (R, L)
        """
        RW_VELOCITY_MIN = -1.0
        RW_VELOCITY_MAX = 1.0
        SERVO_POSITION_MIN = np.deg2rad(-60)
        SERVO_POSITION_MAX = np.deg2rad(60)
        self.action_space = gym.spaces.box.Box(
            low=np.array(
                [
                    RW_VELOCITY_MIN,  # Roll RW
                    RW_VELOCITY_MIN,  # Pitch RW
                    RW_VELOCITY_MIN,  # Yaw RW
                    SERVO_POSITION_MIN,  # Right leg servo
                    SERVO_POSITION_MIN,  # Left leg servo
                ]
            ),
            high=np.array(
                [
                    RW_VELOCITY_MAX,  # Roll RW
                    RW_VELOCITY_MAX,  # Pitch RW
                    RW_VELOCITY_MAX,  # Yaw RW
                    SERVO_POSITION_MAX,  # Right leg servo
                    SERVO_POSITION_MAX,  # Left leg servo
                ]
            ),
        )

        """
        Observation space has:
        - Distance to goal (X, Y, Theta)
        - Current Global Orientation (X, Y, Z, W)
        - Current Reaction Wheel Velocities (R, P, Y)
        - Current Servomotor Positions (R, L)
        """
        WORLD_X_MIN = -10.0
        WORLD_X_MAX = 10.0
        WORLD_Y_MIN = -10.0
        WORLD_Y_MAX = 10.0

        self.observation_space = gym.spaces.box.Box(
            low=np.array(
                [
                    WORLD_X_MIN,  # Distance to goal X
                    WORLD_Y_MIN,  # Distance to goal Y
                    np.deg2rad(-180),  # Distance to goal Theta
                    -1.0,  # Current Quaternion X
                    -1.0,  # Current Quaternion Y
                    -1.0,  # Current Quaternion Z
                    -1.0,  # Current Quaternion W
                    RW_VELOCITY_MIN,  # Roll RW
                    RW_VELOCITY_MIN,  # Pitch RW
                    RW_VELOCITY_MIN,  # Yaw RW
                    SERVO_ANGLE_MIN,  # Right leg servo
                    SERVO_ANGLE_MIN,  # Left leg servo
                ]
            ),
            high=np.array(
                [
                    WORLD_X_MAX,  # Distance to goal X
                    WORLD_Y_MAX,  # Distance to goal Y
                    np.deg2rad(180),  # Distance to goal Theta
                    1.0,  # Current Quaternion X
                    1.0,  # Current Quaternion Y
                    1.0,  # Current Quaternion Z
                    1.0,  # Current Quaternion W
                    RW_VELOCITY_MAX,  # Roll RW
                    RW_VELOCITY_MAX,  # Pitch RW
                    RW_VELOCITY_MAX,  # Yaw RW
                    SERVO_ANGLE_MAX,  # Right leg servo
                    SERVO_ANGLE_MAX,  # Left leg servo
                ]
            ),
        )
        self.np_random, _ = gym.utils.seeding.np_random()

        self.client = p.connect(p.DIRECT)

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self):
        pass

    def close(self):
        p.disconnect(self.client)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]