import gym
import numpy as np
import pybullet as p

from legmo.resources.legmo import LegMo
from legmo.resources.plane import Plane


class LegMoEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    RW_VELOCITY_MIN = -5000.0
    RW_VELOCITY_MAX = 5000.0
    SERVO_POSITION_MIN = np.deg2rad(-60)
    SERVO_POSITION_MAX = np.deg2rad(60)

    MAX_STEPS = 1000

    ORIENTATION_WEIGHT = 0.1
    LEGS_WEIGHT = 0.0
    DIST_WEIGHT = 1000
    GOAL_THRESHOLD = ORIENTATION_WEIGHT * np.deg2rad(5) + LEGS_WEIGHT * np.deg2rad(5)

    LIVING_REWARD = 0.1

    def __init__(self, render=False, hardness=0):
        """
        Action space has:
        - Target Reaction Wheel Velocities  (R, P, Y)
        - Target Servomotors Positions (R, L)
        """

        self.action_space = gym.spaces.box.Box(
            low=np.array(
                [
                    self.RW_VELOCITY_MIN,  # Roll RW
                    self.RW_VELOCITY_MIN,  # Pitch RW
                    self.RW_VELOCITY_MIN,  # Yaw RW
                    self.SERVO_POSITION_MIN,  # Right leg servo
                    self.SERVO_POSITION_MIN,  # Left leg servo
                ]
            ),
            high=np.array(
                [
                    self.RW_VELOCITY_MAX,  # Roll RW
                    self.RW_VELOCITY_MAX,  # Pitch RW
                    self.RW_VELOCITY_MAX,  # Yaw RW
                    self.SERVO_POSITION_MAX,  # Right leg servo
                    self.SERVO_POSITION_MAX,  # Left leg servo
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
                    self.SERVO_POSITION_MIN,  # Current right leg servo
                    self.SERVO_POSITION_MIN,  # Current left leg servo
                    -1.0,  # Goal Quaternion X
                    -1.0,  # Goal Quaternion Y
                    -1.0,  # Goal Quaternion Z
                    -1.0,  # Goal Quaternion W
                    self.SERVO_POSITION_MIN,  # Goal right leg servo
                    self.SERVO_POSITION_MIN,  # Goal left leg servo
                    -np.pi,  # Goal theta (walking) direction
                ]
            ),
            high=np.array(
                [
                    1.0,  # Current Quaternion X
                    1.0,  # Current Quaternion Y
                    1.0,  # Current Quaternion Z
                    1.0,  # Current Quaternion W,
                    self.SERVO_POSITION_MAX,  # Current right leg servo
                    self.SERVO_POSITION_MAX,  # Current left leg servo
                    1.0,  # Goal Quaternion X
                    1.0,  # Goal Quaternion Y
                    1.0,  # Goal Quaternion Z
                    1.0,  # Goal Quaternion W
                    self.SERVO_POSITION_MAX,  # Goal right leg servo
                    self.SERVO_POSITION_MAX,  # Goal left leg servo
                    np.pi,  # Goal theta (walking) direction
                ]
            ),
        )
        self.np_random, _ = gym.utils.seeding.np_random()

        self.client = p.connect(p.GUI if render else p.DIRECT)
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
        self.message = ""
        self.num_steps = 0

        # Get observation to return
        legmo_obs, self.pos = self.legmo.get_observation()

        self.prev_dist_to_goal = self.distanceToGoal(legmo_obs)

        return np.array(legmo_obs + self.goal, dtype=np.float32)

    def generateGoal(self):
        # Generate goal leg positions
        # left = self.np_random.uniform(-30, 30)
        # right = self.np_random.uniform(-30, 30)
        left = 0
        right = 0

        # Generate a goal movement direction
        heading = self.np_random.uniform(-np.pi/8, np.pi/8)

        # Generate a goal orientation that is near the "North Pole" of
        # the unit sphere (ie, close-to-vertical orientation)
        # sphere_phi = np.deg2rad(self.np_random.uniform(-30, 30))
        # sphere_theta = np.deg2rad(self.np_random.uniform(-20, 20))
        sphere_phi = 0
        sphere_theta = 0

        _, goal_ori = p.multiplyTransforms(
            [0, 0, 0],
            p.getQuaternionFromEuler([0, 0, sphere_theta]),
            [0, 0, 0],
            p.getQuaternionFromEuler([0, sphere_phi, 0]),
        )

        goal = goal_ori + (right, left, heading)
        return goal

    def distanceToGoal(self, current_obs):

        current_ori, current_legs = current_obs[:4], current_obs[4:6]
        goal_ori, goal_legs = self.goal[:4], self.goal[4:6]
        goal_angle = self.goal[6]

        return (
            np.arccos(np.abs(np.dot(np.array(current_ori), np.array(goal_ori))))
            * self.ORIENTATION_WEIGHT
            + np.linalg.norm(np.array(current_legs) - np.array(goal_legs))
            * self.LEGS_WEIGHT
            + np.dot(self.pos, [np.sin(goal_angle), np.cos(goal_angle), 0])
            * -self.DIST_WEIGHT
        )

    def step(self, action):

        # Feed action to the robot and get observation of robot's state
        self.legmo.apply_action(action)
        p.stepSimulation()
        legmo_obs, pos = self.legmo.get_observation()
        self.pos = pos

        # Compute reward as reduction in distance to goal
        dist_to_goal = self.distanceToGoal(legmo_obs)
        reward = max(self.prev_dist_to_goal - dist_to_goal, 0) + self.LIVING_REWARD
        self.prev_dist_to_goal = dist_to_goal

        self.num_steps += 1

        # Done by timeout
        if self.num_steps > self.MAX_STEPS:
            self.message = "Timeout!"
            self.done = True
            # reward = -5
        # Done by colliding with ground
        elif (
            len(
                p.getContactPoints(
                    bodyA=self.legmo.robot, bodyB=self.plane.plane, linkIndexA=-1
                )
            )
            > 0
        ):
            self.message = "Collision!"
            self.done = True
            reward -= 50
        # Done by reaching goal
        # elif dist_to_goal < self.GOAL_THRESHOLD:
        #     self.message = "Goal!"
        #     self.done = True
        #     self.succeeded = True
        #     reward = 100

        observation = np.array(legmo_obs + self.goal, dtype=np.float32)

        return observation, reward, self.done, dict()

    def render(self, mode="human"):
        p.removeAllUserDebugItems()

        legmoPos, _ = p.getBasePositionAndOrientation(self.legmo.robot)

        goal_ori = self.goal[:4]
        vector = np.array(p.getMatrixFromQuaternion(goal_ori)).reshape(3, 3) @ np.array(
            [0, 0, 1]
        )

        p.addUserDebugLine(
            lineFromXYZ=legmoPos,
            lineToXYZ=(np.array(legmoPos) + np.array(vector)),
            lineColorRGB=[1, 0, 1],
            lineWidth=30,
        )
        if self.done:
            p.addUserDebugText(
                text=self.message, textPosition=[0.1, 0, 0.1], textColorRGB=[0, 1, 0]
            )

    def close(self):
        p.disconnect(self.client)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]