import pybullet as p
import time
import pybullet_data
import numpy as np

physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
# p.setAdditionalSearchPath("./")
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
# p.setGravity(0, 0, -9.8)
p.setGravity(0, 0, 0)
planeId = p.loadURDF("plane.urdf")
startPos = [0, 0, 1]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
boxId = p.loadURDF("rwrl.urdf", startPos, startOrientation)
# boxId = p.loadURDF("r2d2.urdf", startPos, startOrientation)
n = p.getNumJoints(boxId)
for i in range(n):
    print("Joint", i, p.getJointInfo(boxId, i))

# p.setJointMotorControlArray(boxId, [3, 5], p.POSITION_CONTROL, [.5236, -.5236])

# set the center of mass frame (loadURDF sets base link frame)
for i in range(10000):
    # if i < 240*5:
    #     p.setJointMotorControlArray(boxId, [3, 5], p.VELOCITY_CONTROL, targetVelocities=[.5, -.5])
    if i == 240 * 3:
    	print("start leg")
    if 240*3 < i < 240*5:
    	p.setJointMotorControlArray(boxId, [3, 5], p.VELOCITY_CONTROL, targetVelocities=[1, -1])
        # p.setJointMotorControlArray(boxId, [0], p.VELOCITY_CONTROL, targetVelocities=[115])
    if i == 240*5:
    	print("end leg")
    p.stepSimulation()
    time.sleep(1.0 / 240.0)
cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
print(cubePos, cubeOrn)
p.disconnect()
