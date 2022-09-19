from motion_planning_goal.static_subgoal import StaticSubgoal
from motion_planning_goal.dynamic_subgoal import DynamicSubgoal

goal_1_dict = {
    "m": 3, "w": 1.0, "prime": True,
    "indices": [0, 1, 2], "parent_link": 0, "child_link": 3,
    "desired_position": [1, 0, 0.1],
    "epsilon": 0.02, "type": "static_subgoal",
}

goal1 = StaticSubgoal(name="goal1", content_dict=goal_1_dict)
dynamic_goal_dict = {
    "m": 3, "w": 1.0, "prime": True, "indices": [0, 1, 2],
    "parent_link": 0, "child_link": 3,
    "trajectory": ["0.5", "0.2 + 0.2 * ca.sin(0.3 * t)", "0.4"],
    "epsilon": 0.08, "type": "analytic_subgoal",
}
dynamic_goal = DynamicSubgoal(name="goal2", content_dict=dynamic_goal_dict)
splineDict = {"degree": 2,
        "controlPoints": [[0.0, -0.0, 0.2], [3.0, 0.0, 2.2], [3.0, 3.0, 1.2]],
        "duration": 10}
spline_goal_dict = {
    "m": 3, "w": 1.0, "prime": True,
    "indices": [0, 1, 2], "parent_link": 0, "child_link": 3,
    "trajectory": splineDict, "epsilon": 0.08,
    "type": "spline_subgoal",
}
spline_goal = DynamicSubgoal(name="goal3", content_dict=spline_goal_dict)
