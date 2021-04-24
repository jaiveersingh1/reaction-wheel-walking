import pybullet as p
import time
import pybullet_data
import numpy as np 

class BoxBot:
    def __init__(self, startPos, startOrientation):
        self.boxId = p.loadURDF("boxbot.urdf", startPos, startOrientation)
        
    def simulate(self, time_steps):
        print(self.getPose())
        for i in range(time_steps):
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
    
    def getPose(self):
        return p.getBasePositionAndOrientation(self.boxId)

def main():
    print("STARTING SIM")
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    planeId = p.loadURDF("plane.urdf")  # Loads the floor
    startPos = [0, 0, 1]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])
    box = BoxBot(startPos, startOrientation)

    box.simulate(10000)

    p.disconnect()

if __name__ == "__main__":
    main()
