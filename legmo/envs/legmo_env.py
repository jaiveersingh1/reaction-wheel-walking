import gym
import numpy as np
import pybullet as p
from time import sleep

from legmo.resources.legmo import LegMo
from legmo.resources.plane import Plane


class LegMoEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, render=False):
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
        - Current Global Orientation (X, Y, Z, W)
        """
        self.observation_space = gym.spaces.box.Box(
            low=np.array(
                [
                    -1.0,  # Current Quaternion X
                    -1.0,  # Current Quaternion Y
                    -1.0,  # Current Quaternion Z
                    -1.0,  # Current Quaternion W
                    -1.0,  # Goal Quaternion X
                    -1.0,  # Goal Quaternion Y
                    -1.0,  # Goal Quaternion Z
                    -1.0,  # Goal Quaternion W
                ]
            ),
            high=np.array(
                [
                    1.0,  # Current Quaternion X
                    1.0,  # Current Quaternion Y
                    1.0,  # Current Quaternion Z
                    1.0,  # Current Quaternion W
                    1.0,  # Goal Quaternion X
                    1.0,  # Goal Quaternion Y
                    1.0,  # Goal Quaternion Z
                    1.0,  # Goal Quaternion W
                ]
            ),
        )
        self.np_random, _ = gym.utils.seeding.np_random()

        self.should_render = render
        self.client = p.connect(p.GUI if self.should_render else p.DIRECT)

        self.reset()

    def reset(self):
        # Reset basic PyBullet properties
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.8)

        # Re-initialize plane and LegMo robot
        self.plane = Plane(self.client)
        self.legmo = LegMo(self.client)

        # Set the goal to a random target
        self.goal = self.generateGoal()
        self.done = False
        self.num_steps = 0

        # Get observation to return
        legmo_obs = self.legmo.get_observation()

        self.prev_dist_to_goal = np.arccos(
            np.abs(np.dot(np.array(legmo_obs), np.array(self.goal)))
        )
        return np.array(legmo_obs + self.goal, dtype=np.float32)

    def generateGoal(self, drawGoal=True):
        # Generate a goal orientation that is near the "North Pole" of
        # the unit sphere (ie, close-to-vertical orientation)
        phi = np.deg2rad(self.np_random.uniform(0, 20))
        theta = np.deg2rad(self.np_random.uniform(0, 360))

        # Spin around by theta
        theta_transform = p.getQuaternionFromEuler([0, 0, theta])

        # Pitch down by phi
        phi_transform = p.getQuaternionFromEuler([0, phi, 0])

        _, goal_ori = p.multiplyTransforms(
            [0, 0, 0], theta_transform, [0, 0, 0], phi_transform
        )
        return goal_ori

    def step(self, action):
        GOAL_THRESHOLD = np.deg2rad(5)
        LIVING_REWARD = -0.5
        MAX_STEPS = 200

        # Feed action to the robot and get observation of robot's state
        self.legmo.apply_action(action)
        p.stepSimulation()
        legmo_obs = self.legmo.get_observation()

        # Compute reward as change in angle between two orientations
        dist_to_goal = np.arccos(
            np.abs(np.dot(np.array(legmo_obs), np.array(self.goal)))
        )
        reward = max(self.prev_dist_to_goal - dist_to_goal, 0) + LIVING_REWARD
        self.prev_dist_to_goal = dist_to_goal

        self.num_steps += 1

        message = ""
        succeeded = False
        # Done by timeout
        if self.num_steps > MAX_STEPS:
            message = "Timeout!"
            self.done = True
            reward = -5
        # Done by colliding with ground
        if (
            len(
                p.getContactPoints(
                    bodyA=self.legmo.robot, bodyB=self.plane.plane, linkIndexA=-1
                )
            )
            > 0
        ):
            message = "Collision!"
            self.done = True
            reward = -50
        # Done by reaching goal
        elif dist_to_goal < GOAL_THRESHOLD:
            message = "Goal!"
            self.done = True
            reward = 100

            succeeded = True

        observation = np.array(legmo_obs + self.goal, dtype=np.float32)

        if self.should_render:
            if self.done:
                p.addUserDebugText(
                    text=message,
                    textPosition=[0.1, 0, 0],
                    textColorRGB=[0, 1, 0] if succeeded else [1, 0, 0],
                )
                sleep(0.25)

            # self.render()
        return observation, reward, self.done, dict()

    def render(self, mode="human"):
        p.removeAllUserDebugItems()

        legmoPos, _ = p.getBasePositionAndOrientation(self.legmo.robot)

        vector = np.array(p.getMatrixFromQuaternion(self.goal)).reshape(
            3, 3
        ) @ np.array([0, 0, 1])

        p.addUserDebugLine(
            lineFromXYZ=legmoPos,
            lineToXYZ=(np.array(legmoPos) + np.array(vector)),
            lineColorRGB=[1, 0, 1],
            lineWidth=30,
        )

    def close(self):
        p.disconnect(self.client)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]