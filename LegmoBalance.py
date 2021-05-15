import pybullet as p
import time
import pybullet_data
import numpy as np 

class Legmo:
    def __init__(self, startPos, startOrientation):
        self.id = p.loadURDF("legmo.urdf", startPos, startOrientation)
        self.desiredPosEuler = np.array([np.pi/4, 0, 0])
        self.lastPos = np.array([0, 0, 0])
        self.currPos = np.array([0, 0, 0])

        self.kp = np.array([5.5, 0, 0])
        self.kd = np.array([20, 0, 0])
        
    def simulate(self, time_steps):
        for i in range(time_steps):
            print(i)
            self.lastPos = self.currPos
            self.currPos = np.array(self.getEulerOrientation())
            if i > 1000:
                print(self.getEulerOrientation())
                self.pdControl(self.kp, self.kd)
            p.stepSimulation()
            time.sleep(1.0 / 240.0)
    
    def getPose(self):
        # Returns a tuple of (position, quaternion)
        return p.getBasePositionAndOrientation(self.id)

    def getEulerOrientation(self):
        return p.getEulerFromQuaternion(self.getPose()[1])
    
    def setTorque(self, torque, joint=0):
        p.setJointMotorControl2(bodyIndex=self.id,
                                        jointIndex=joint,
                                        controlMode=p.TORQUE_CONTROL,
                                        force=torque)

    def setVelocity(self, velocity, maxForce=500):
        p.setJointMotorControl2(bodyIndex=self.id,
                                jointIndex=0,
                                controlMode=p.TORQUE_CONTROL,
                                targetVelocity=velocity,
                                force=maxForce)
    
    def pdControl(self, kp, kd):
        velocity = self.currPos - self.lastPos
        e = self.desiredPosEuler - self.currPos
        e_dot = -velocity
        
        controlInput = -kp * e - kd * e_dot
        self.setTorque(controlInput[0], 2)
        self.setTorque(controlInput[1], 3)
        self.setTorque(controlInput[2], 4)



    def controlTorque(self):
        orientation = self.getEulerOrientation()
        if orientation[0] < self.desiredPosEuler[0]:
            self.setTorque(-1)
        elif orientation[0] > self.desiredPosEuler[0]:
            self.setTorque(1)
    
    def render(self, mode="human"):
        p.removeAllUserDebugItems()

        legmoPos, _ = p.getBasePositionAndOrientation(self.id)

        goal_ori = p.getQuaternionFromEuler(self.desiredPosEuler)
        vector = np.array(p.getMatrixFromQuaternion(goal_ori)).reshape(3, 3) @ np.array(
            [0, 0, 1]
        )

        p.addUserDebugLine(
            lineFromXYZ=legmoPos,
            lineToXYZ=(np.array(legmoPos) + np.array(vector)),
            lineColorRGB=[1, 0, 1],
            lineWidth=30,
        )


def main():
    print("STARTING SIM")
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    planeId = p.loadURDF("plane.urdf")  # Loads the floor
    startPos = [0, 0, .2]
    startOrientation = p.getQuaternionFromEuler([0, 0, 0])
    legmo = Legmo(startPos, startOrientation)

    legmo.simulate(10000)

    p.disconnect()

 

if __name__ == "__main__":
    main()
