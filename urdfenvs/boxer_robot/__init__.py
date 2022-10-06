from gym.envs.registration import register
register(
    id='boxer-robot-vel',
    entry_point='urdfenvs.boxer_robot.envs:BoxerRobotVelEnv'
)
register(
    id='boxer-robot-acc',
    entry_point='urdfenvs.boxer_robot.envs:BoxerRobotAccEnv'
)
