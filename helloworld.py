import pybullet as p
import time
import pybullet_data
import numpy as np

physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")
startPos = [0, 0, 1]
startOrientation = p.getQuaternionFromEuler([0, np.pi / 2, 0])
boxId = p.loadURDF("boxmodel.urdf", startPos, startOrientation)
# set the center of mass frame (loadURDF sets base link frame)

for i in range(10000):
    print(p.getBasePositionAndOrientation(boxId))
    p.stepSimulation()
    if i == 1000:
        p.setJointMotorControl2(bodyIndex=boxId,
                                jointIndex=0,
                                controlMode=p.VELOCITY_CONTROL,
                                targetVelocity=100,
                                force=500)
    time.sleep(1.0 / 240.0)
cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
print(cubePos, cubeOrn)
p.disconnect()
