from gym.envs.registration import register
register(
    id='pointRobot-vel-v7',
    entry_point='urdfenvs.point_robot_urdf.envs:PointRobotVelEnv'
)
register(
    id='pointRobot-acc-v7',
    entry_point='urdfenvs.point_robot_urdf.envs:PointRobotAccEnv'
)
