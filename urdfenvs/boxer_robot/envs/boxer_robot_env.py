from urdfenvs.boxer_robot.resources.boxer_robot import BoxerRobot
from urdfenvs.urdfCommon.urdf_env import UrdfEnv


class BoxerRobotEnv(UrdfEnv):
    def __init__(self, **kwargs):
        super().__init__(BoxerRobot(), **kwargs)
        self.reset()
