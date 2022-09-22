import os
import math
import gym
import numpy as np
import urdfenvs.point_robot_urdf
import pytest

from motion_planning_env.sphere_obstacle import SphereObstacle
from motion_planning_env.cylinder_obstacle import CylinderObstacle
from motion_planning_env.box_obstacle import BoxObstacle
from motion_planning_env.urdf_obstacle import UrdfObstacle

sphere_dict = {
    "position": [2, 1, 1],
    "geometry": {"radius": 0.6},
}
sphere = SphereObstacle(name="simple_sphere", content_dict=sphere_dict)

cylinder_dict = {
    "position": [2, 1, 1],
    "geometry": {"radius": 0.6, "height": 0.6},
}
cylinder = CylinderObstacle(name="simple_cylinder", content_dict=cylinder_dict)

box_dict = {
    "position": [2, 1, 1],
    "geometry": {"length": 0.6, "width": 0.5, "height": 0.6},
}
box = BoxObstacle(name="simple_box", content_dict=box_dict)

duck_small_dict= {
    "position": [2, -1, 0.25],
    "orientation": [math.pi/2, 0, 0],
    "geometry": {
        "urdf": os.path.join(os.path.dirname(__file__),
            "obstacle_data/duck/duck.urdf"),
    }
}
duck = UrdfObstacle(name="duck_urdf", content_dict=duck_small_dict)

def test_add_bullet_id_to_obst():
    init_pos = np.array([0.0, -1.0, 0.0])
    init_vel = np.array([-1.0, 0.0, 0.0])
    env = gym.make("pointRobotUrdf-acc-v0", render=False, dt=0.01)
    _ = env.reset(pos=init_pos, vel=init_vel)

    sphere_bullet_id = env.add_obstacle(sphere)
    cylinder_bullet_id = env.add_obstacle(cylinder)
    box_bullet_id = env.add_obstacle(box)
    duck_bullet_id = env.add_obstacle(duck)

    assert env.get_bullet_id_to_obst()[sphere_bullet_id] == sphere.name()
    assert env.get_bullet_id_to_obst()[cylinder_bullet_id] == cylinder.name()
    assert env.get_bullet_id_to_obst()[box_bullet_id] == box.name()
    assert env.get_bullet_id_to_obst()[duck_bullet_id] == duck.name()

def test_add_bullet_id_to_obst_non_unique_key():
    init_pos = np.array([0.0, -1.0, 0.0])
    init_vel = np.array([-1.0, 0.0, 0.0])
    env = gym.make("pointRobotUrdf-acc-v0", render=False, dt=0.01)
    _ = env.reset(pos=init_pos, vel=init_vel)

    with pytest.raises(KeyError):
        env.add_bullet_id_to_obst(1, "obst1")
        env.add_bullet_id_to_obst(1, "obstacle1")

def test_add_bullet_id_to_obst_non_unique_value():
    init_pos = np.array([0.0, -1.0, 0.0])
    init_vel = np.array([-1.0, 0.0, 0.0])
    env = gym.make("pointRobotUrdf-acc-v0", render=False, dt=0.01)
    _ = env.reset(pos=init_pos, vel=init_vel)

    with pytest.raises(ValueError):
        env.add_bullet_id_to_obst(1, "obst1")
        env.add_bullet_id_to_obst(2, "obst1")
