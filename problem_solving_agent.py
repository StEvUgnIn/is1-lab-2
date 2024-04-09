"""Problem Solving Agent

Returns:
    int: Action

Yields:
    iterable[int]: Range of actions
"""

from collections import defaultdict

from search import tree_search, uniform_cost_search
from environment import env


class Agent():
    """Dummy Agent
    """


class SimpleProblemSolvingAgent(Agent):
    """A problem solving agent is a goal-based agent consider future actions and desirability for their actions based on
    atomic representation"""

    def __init__(self, state, goal = None):
        """persistent:
        seq, an action sequence (iterable), initially empty
        state, some description of the current world state
        goal, a goal, initially None
        problem, a problem formulation

        Args:
            initial_state (int): the initial state of the agent
            goal (int): the goal of the agent
        """
        self.state = None
        self.goal = goal
        self.problem = None
        self.seq = []
        self.planning = None
        self.reset(state)

    def select_action(self, state):
        """returns an action"""
        self.update_state(state=state)
        if not self.seq:
            self.problem = self.formulate_problem(self.state, self.goal, self.planning)
            self.seq = self.search(self.problem)
        return self.seq.pop(0)

    def update_state(self, **percept):
        """Update state

        Args:
            percept (dict): object containing the current state
        """
        self.state = percept['state']

    @classmethod
    def _action_range(cls, state):
        """Range of actions

        Args:
            state (int): a state of the agent
        """
        if state - 4 in env.observation_space:
            yield 0
        if state + 1 in env.observation_space:
            yield 1
        if state - 1 in env.observation_space:
            yield 2
        if state + 4 in env.observation_space:
            yield 3

    @classmethod
    def formulate_problem(cls, state, goal, planning):
        """Formulate the problem

        Args:
            state (int): a state of the agent
            goal (int): a goal
            planning (defaultdict): the planning of the agent

        Returns:
            dict: a problem formulation
        """
        return {
            'initial_state': state,
            'actions': cls._action_range,
            'goal_test': lambda state: state == goal,
            'state_space': env.observation_space,
            'planning': planning,
        }

    @classmethod
    def search(cls, problem):
        """Solve the problem

        Args:
            problem (dict): a problem formulation

        Returns:
            NotImplemented: not implemented (abstract)
        """
        return NotImplemented

    def update(self, old_state, action, cost, new_state):
        """Do the learning (e.g. update the Q-values, or collect data to do off-policy learning)

        Args:
            old_state (int): the former state
            action (int): an action
            cost (float): the cost (reward)
            new_state (int): the new state
        """
        if not self.planning[old_state].get(action):
            self.planning[old_state][action] = (-cost, new_state)
        else:
            self.planning[old_state][action] = (self.planning[old_state][action][0] - cost, new_state)
        self.max_reward -= cost

    def cheat_code(self):
        """Cheat code
        """
        self.planning[8+1][0] = (100, 4+1)
        self.planning[4+0][1] = (100, 4+1)
        self.planning[0+1][2] = (100, 4+1)
        self.planning[4+2][3] = (100, 4+1)
        self.planning[4+2][1] = (100, 4+3)
        self.planning[0+3][2] = (100, 4+3)
        self.planning[8+2][1] = (100, 8+3)
        self.planning[8+0][2] = (100, 12+0)
        self.planning[12+1][3] = (100, 12+0)

    def reset(self, state, cheater = False):
        """reset the agent to its initial state after an epoch. Can also be used to perform learning after an epoch)

        Args:
            state (int, optional): the state of the agent. Defaults to None.
        """
        self.state = state
        self.planning = defaultdict(dict)
        self.problem = self.formulate_problem(state, self.goal, self.planning)
        self.seq = self.search(self.problem)
        if not self.seq:
            raise IndexError("empty sequence")
        self.max_reward = 0
        if cheater:
            self.cheat_code()


class UniformCostSearchAgent(SimpleProblemSolvingAgent):
    """A UCS agent is an uninformed problem solving agent using the UCS algorithm to learn the policy of an environment."""

    @classmethod
    def search(cls, problem):
        """Solve the problem

        Args:
            problem (dict): a problem formulation

        Returns:
            list[int]: actions sequence
        """
        return uniform_cost_search(problem)


class TreeSearchAgent(SimpleProblemSolvingAgent):
    """A tree-search agent is an informed problem solving agent using the A* algorithm to learn the policy of an environment.
    """

    @classmethod
    def search(cls, problem):
        """Solve the problem

        Args:
            problem (dict): a problem formulation

        Returns:
            list[int]: actions sequence
        """
        return tree_search(problem)
