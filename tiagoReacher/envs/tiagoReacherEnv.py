import numpy as np
import pybullet as p
from abc import abstractmethod

from tiagoReacher.resources.tiagoRobot import TiagoRobot
from albertReacher.resources.plane import Plane
from urdfCommon.urdfEnv import UrdfEnv


class TiagoReacherEnv(UrdfEnv):

    def __init__(self, render=False, dt=0.01, n=19):
        super().__init__(TiagoRobot(), render=render, dt=dt)
        self._n = self.robot.n()
        self.setSpaces()

    def n(self):
        return self._n

    @abstractmethod
    def setSpaces(self):
        pass

    @abstractmethod
    def applyAction(self, action):
        pass

    def reset(self, initialSet=False, pos=None, vel=None):
        if not isinstance(pos, np.ndarray) or not pos.size == self._n + 1:
            pos = np.zeros(self._n+1)
        if not isinstance(vel, np.ndarray) or not vel.size == self._n:
            vel = np.zeros(self._n)
        if not initialSet:
            print("Run " + str(self._nSteps) + " steps in this run")
            self._nSteps = 0
            p.resetSimulation()
        p.setPhysicsEngineParameter(
            fixedTimeStep=self._dt, numSubSteps=self._numSubSteps
        )
        self.plane = Plane()
        self.robot.reset(pos=pos, vel=vel)
        p.setGravity(0, 0, -10)
        p.stepSimulation()

        # Get observation to return
        return self.robot.get_observation()
