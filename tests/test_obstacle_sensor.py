import gym
import numpy as np
import pytest

from urdfenvs.sensors.obstacle_sensor import ObstacleSensor
import urdfenvs.point_robot_urdf

from motion_planning_env.sphere_obstacle import SphereObstacle
from motion_planning_env.urdf_obstacle import UrdfObstacle
from motion_planning_env.dynamic_sphere_obstacle import DynamicSphereObstacle


obst_1_dict = {
    "type": "sphere",
    "position": [2.0, 2.0, 1.0],
    "geometry": {"radius": 1.0},
}
sphere_obst_1 = SphereObstacle(name="simple_sphere", content_dict=obst_1_dict)

spline_dict = {"degree": 2,
    "controlPoints": [[0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0]],
    "duration": 10}

dynamic_obst_3_dict = {
    "type": "splineSphere",
    "geometry": {"trajectory": spline_dict, "radius": 0.2},
}
dynamic_sphere_obst_3 = DynamicSphereObstacle(name="dyn_simple_sphere_3", content_dict=dynamic_obst_3_dict)


@pytest.fixture
def point_robot_env():
    import urdfenvs.point_robot_urdf

    env = gym.make("pointRobotUrdf-vel-v0", render=False, dt=0.01)
    _ = env.reset()
    return env


def test_static_obstacle(point_robot_env):
    point_robot_env.add_obstacle(sphere_obst_1)

    # add sensor
    sensor = ObstacleSensor()
    sensor.set_bullet_id_to_obst(point_robot_env.get_bullet_id_to_obst())
    point_robot_env.add_sensor(sensor)
    action = np.random.random(point_robot_env.n())
    ob, _, _, _ = point_robot_env.step(action)
    assert "obstacleSensor" in ob
    assert "simple_sphere" in ob["obstacleSensor"]
    assert isinstance(ob["obstacleSensor"]["simple_sphere"]["pose"]["position"], np.ndarray)
    assert isinstance(ob["obstacleSensor"]["simple_sphere"]["twist"]["linear"], np.ndarray)
    assert isinstance(ob["obstacleSensor"]["simple_sphere"]["pose"]["orientation"], np.ndarray)
    assert isinstance(ob["obstacleSensor"]["simple_sphere"]["twist"]["angular"], np.ndarray)
    np.testing.assert_array_almost_equal(
        ob["obstacleSensor"]["simple_sphere"]["pose"]["position"],
        sphere_obst_1.position(),
        decimal=2,
    )


def test_dynamic_obstacle(point_robot_env):
    point_robot_env.add_obstacle(dynamic_sphere_obst_3)

    # add sensor
    sensor = ObstacleSensor()
    sensor.set_bullet_id_to_obst(point_robot_env.get_bullet_id_to_obst())
    point_robot_env.add_sensor(sensor)
    action = np.random.random(point_robot_env.n())
    ob, _, _, _ = point_robot_env.step(action)
    assert "obstacleSensor" in ob
    assert "dyn_simple_sphere_3" in ob["obstacleSensor"]
    assert isinstance(ob["obstacleSensor"]["dyn_simple_sphere_3"]["pose"]["position"], np.ndarray)
    assert isinstance(ob["obstacleSensor"]["dyn_simple_sphere_3"]["twist"]["linear"], np.ndarray)
    assert isinstance(ob["obstacleSensor"]["dyn_simple_sphere_3"]["pose"]["orientation"], np.ndarray)
    assert isinstance(ob["obstacleSensor"]["dyn_simple_sphere_3"]["twist"]["angular"], np.ndarray)
    np.testing.assert_array_almost_equal(
        ob["obstacleSensor"]["dyn_simple_sphere_3"]["pose"]["position"],
        dynamic_sphere_obst_3.position(t=point_robot_env.t()),
        decimal=2,
    )


def test_shape_observation_space(point_robot_env):
    # add obstacle and sensor
    point_robot_env.add_obstacle(sphere_obst_1)
    sensor = ObstacleSensor()
    sensor.set_bullet_id_to_obst(point_robot_env.get_bullet_id_to_obst())
    point_robot_env.add_sensor(sensor)
    action = np.random.random(point_robot_env.n())
    ob, _, _, _ = point_robot_env.step(action)

    assert ob["obstacleSensor"]["simple_sphere"]["pose"]["position"].shape == (3, )
    assert ob["obstacleSensor"]["simple_sphere"]["pose"]["orientation"].shape == (4, )
    assert ob["obstacleSensor"]["simple_sphere"]["twist"]["linear"].shape == (3, )
    assert ob["obstacleSensor"]["simple_sphere"]["twist"]["angular"].shape == (3, )


@pytest.mark.skip(
    reason="Fails due to different position in pybullet and "\
            "obstacle from motion planning scene"
)

def test_urdf_obstacle(point_robot_env):
    urdf_obst_1_dict = {
        "type": "urdf",
        "position": [1.5, 0.0, 0.05],
        "geometry": {
        "urdf": os.path.join(os.path.dirname(__file__), "obstacle_data/duck.urdf"),
            },
    }
    urdf_obst_1 = UrdfObstacle(name="duckUrdf", content_dict=urdf_obst_1_dict)

    point_robot_env.add_obstacle(urdf_obst_1)

    sensor = ObstacleSensor()
    sensor.set_bullet_id_to_obst(point_robot_env.get_bullet_id_to_obst())
    point_robot_env.add_sensor(sensor)
    action = np.random.random(point_robot_env.n())
    ob, _, _, _ = point_robot_env.step(action)
    assert "obstacleSensor" in ob
    assert "duckUrdf" in ob["obstacleSensor"]
    assert isinstance(ob["obstacleSensor"]["duckUrdf"]["pose"]["position"], np.ndarray)
    assert isinstance(ob["obstacleSensor"]["duckUrdf"]["twist"]["linear"], np.ndarray)
    assert isinstance(ob["obstacleSensor"]["duckUrdf"]["pose"]["orientation"], np.ndarray)
    assert isinstance(ob["obstacleSensor"]["duckUrdf"]["twist"]["angular"], np.ndarray)
    np.testing.assert_array_almost_equal(
        ob["obstacleSensor"]["pose"]["position"],
        dynamic_sphere_obst_3.position(t=point_robot_env.t()),
        decimal=2,
    )
