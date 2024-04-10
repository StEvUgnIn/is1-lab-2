from environment import env, run_experiment
from problem_solving_agent import (
    TreeSearchAgent,
    UniformCostSearchAgent
)


env.action_space.seed(0)
_goal = 15
observation, info = env.reset()
informed_agent = TreeSearchAgent(observation, _goal)
run_experiment(informed_agent)
print()

uninformed_agent = UniformCostSearchAgent(observation, _goal)
run_experiment(uninformed_agent)
print()
env.close()
