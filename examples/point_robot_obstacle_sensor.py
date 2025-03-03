import gym
import urdfenvs.point_robot_urdf # pylint: disable=unused-import
from urdfenvs.sensors.obstacle_sensor import ObstacleSensor
from examples.scene_objects.obstacles import (
    sphere_obst_1,
    urdf_obst_1,
    dynamic_sphere_obst_3,
)
import numpy as np


def main():
    env = gym.make("pointRobotUrdf-vel-v0", dt=0.05, render=True)

    default_action = np.array([0.1, 0.0, 0.0])
    n_episodes = 1
    n_steps = 100000
    pos0 = np.array([1.0, 0.1, 0.0])
    vel0 = np.array([1.0, 0.0, 0.0])
    ob = env.reset(pos=pos0, vel=vel0)
    print(f"Initial observation : {ob}")

    # add obstacles
    env.add_obstacle(sphere_obst_1)
    env.add_obstacle(urdf_obst_1)
    env.add_obstacle(dynamic_sphere_obst_3)

    # add sensor
    sensor = ObstacleSensor()
    env.add_sensor(sensor)

    for _ in range(n_episodes):

        print("Starting episode")
        t = 0
        for _ in range(n_steps):
            t += env.dt()
            action =default_action
            ob, _, _, _ = env.step(action)
            print(ob["obstacleSensor"])


if __name__ == "__main__":
    main()
