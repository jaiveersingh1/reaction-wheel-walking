import pybullet as p
import time
import pybullet_data
import numpy as np 

class Robot:
    def __init__(self, urdf, startPos, startOrientation):
        self.id = p.loadURDF(urdf, startPos, startOrientation)
    
    def simulate(self, time_steps):
        for i in range(time_steps):
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

    def getPose(self):
        # Returns a tuple of (position, quaternion)
        return p.getBasePositionAndOrientation(self.id)

    def getEulerOrientation(self):
        return p.getEulerFromQuaternion(self.getPose()[1])



