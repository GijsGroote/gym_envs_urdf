from motion_planning_env.sphere_obstacle import SphereObstacle
from motion_planning_env.urdf_obstacle import UrdfObstacle
from motion_planning_env.dynamic_sphere_obstacle import DynamicSphereObstacle

import os

obst_1_dict = {
    "type": "sphere",
    "position": [2.0, 2.0, 1.0],
    "geometry": {"radius": 1.0},
}
sphere_obst_1 = SphereObstacle(name="simpleSphere", content_dict=obst_1_dict)

obst_2_dict = {
    "type": "sphere",
    "movable": True,
    "position": [2.0, -0.0, 0.5],
    "geometry": {"radius": 0.2},
}
sphere_obst_2 = SphereObstacle(name="simpleSphere", content_dict=obst_2_dict)

urdf_obst_1_dict = {
    "type": "urdf",
    "position": [1.5, 0.0, 0.05],
    "geometry": {
    "urdf": os.path.join(os.path.dirname(__file__), "obstacle_data/duck.urdf"),
        },
}
urdf_obst_1 = UrdfObstacle(name="duckUrdf", content_dict=urdf_obst_1_dict)

dynamic_obst_1_dict = {
    "type": "sphere",
    "geometry": {"trajectory": ["2.0 - 0.1 * t", "-0.0", "0.1"], "radius": 0.2},
}
dynamic_sphere_obst_1 = DynamicSphereObstacle(
    name="simpleSphere", content_dict=dynamic_obst_1_dict)

dynamic_obst_2_dict = {
    "type": "analyticSphere",
    "geometry": {"trajectory": ["0.6", "0.5 - 0.1 * t", "0.8"], "radius": 0.2},
}
dynamic_sphere_obs_2 = DynamicSphereObstacle(
    name="simpleSphere", content_dict=dynamic_obst_2_dict)

spline_dict = {"degree": 2,
    "controlPoints": [[0.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0]],
    "duration": 10}
dynamic_obst_3_dict = {
    "type": "splineSphere",
    "geometry": {"trajectory": spline_dict, "radius": 0.2},
}
dynamic_sphere_obst_3 = DynamicSphereObstacle(name="simpleSphere",
    content_dict=dynamic_obst_3_dict)

