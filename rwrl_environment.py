import pybullet as p
import pybullet_data

client = p.connect(p.GUI)

p.setGravity(0, 0, -10, physicsClientId=client)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")