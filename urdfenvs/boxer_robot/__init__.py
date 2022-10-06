from gym.envs.registration import register
register(
    id='boxerRobot-vel-v7',
    entry_point='urdfenvs.boxer_robot.envs:BoxerRobotVelEnv'
)
register(
    id='boxerRobot-acc-v7',
    entry_point='urdfenvs.boxer_robot.envs:BoxerRobotAccEnv'
)
