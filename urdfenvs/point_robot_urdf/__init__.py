from gym.envs.registration import register
register(
    id='point_robot-vel',
    entry_point='urdfenvs.point_robot_urdf.envs:PointRobotVelEnv'
)


register(
    id='point_robot-acc',
    entry_point='urdfenvs.point_robot_urdf.envs:PointRobotAccEnv'
)
