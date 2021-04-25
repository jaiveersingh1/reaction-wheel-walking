import pybullet as p
import time
import pybullet_data
import numpy as np 

class BoxBot:
    def __init__(self, startPos, startOrientation):
        self.id = p.loadURDF("boxbot.urdf", startPos, startOrientation)
        self.desiredPosEuler = [np.pi/4, 0, 0]
        
    def simulate(self, time_steps):
        for i in range(time_steps):
            print(i)
            if i > 500:
                self.controlTorque()
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
    
    def getPose(self):
        # Returns a tuple of (position, quaternion)
        return p.getBasePositionAndOrientation(self.id)

    def getEulerOrientation(self):
        return p.getEulerFromQuaternion(self.getPose()[1])
    
    def setTorque(self, torque):
        p.setJointMotorControl2(bodyIndex=self.id,
                                        jointIndex=0,
                                        controlMode=p.TORQUE_CONTROL,
                                        force=torque)

    def setVelocity(self, velocity, maxForce=500):
        p.setJointMotorControl2(bodyIndex=self.id,
                                jointIndex=0,
                                controlMode=p.TORQUE_CONTROL,
                                targetVelocity=velocity,
                                force=maxForce)
    
    def controlTorque(self):
        orientation = self.getEulerOrientation()
        if orientation[0] < self.desiredPosEuler[0]:
            self.setTorque(-1)
        elif orientation[0] > self.desiredPosEuler[0]:
            self.setTorque(1)

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
