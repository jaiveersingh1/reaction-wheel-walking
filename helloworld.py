import pybullet as p
import time
import pybullet_data
import numpy as np

physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
p.setGravity(0, 0, -1)
planeId = p.loadURDF("plane.urdf")
startPos = [0, 0, 1]
startOrientation = p.getQuaternionFromEuler([0, np.pi / 2, 0])
boxId = p.loadURDF("r2d2.urdf", startPos, startOrientation)


# set the center of mass frame (loadURDF sets base link frame)
for i in range(10000):

    phi = np.deg2rad(30)
    theta = np.deg2rad(80)

    # Spin around by theta
    theta_transform = p.getQuaternionFromEuler([0, 0, theta])

    # Pitch down by phi
    phi_transform = p.getQuaternionFromEuler([0, phi, 0])

    _, goal_ori = p.multiplyTransforms(
        [0, 0, 0], theta_transform, [0, 0, 0], phi_transform
    )

    if True:
        p.removeAllUserDebugItems()
        legmoPos, _ = p.getBasePositionAndOrientation(boxId)

        vector = np.array(p.getMatrixFromQuaternion(goal_ori)).reshape(3, 3) @ np.array(
            [0, 0, 1]
        )

        p.addUserDebugLine(
            lineFromXYZ=legmoPos,
            lineToXYZ=(np.array(legmoPos) + np.array(vector)),
            lineColorRGB=[1, 0, 1],
            lineWidth=30,
        )

    p.stepSimulation()
    time.sleep(1.0 / 240.0)
cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
print(cubePos, cubeOrn)
p.disconnect()
