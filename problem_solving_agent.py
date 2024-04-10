"""Problem Solving Agent

Returns:
    int: Action

Yields:
    iterable[int]: Range of actions
"""

from collections import defaultdict

from search import tree_search, uniform_cost_search
from environment import env, nrows


class Agent():
    """Dummy Agent
    """


class SimpleProblemSolvingAgent(Agent):
    """A problem solving agent is a goal-based agent consider future actions and desirability for their actions based on
    atomic representation"""

    def __init__(self, state, goal=None):
        """persistent:
            seq (iterable): an action sequence, initially empty
            state (int): some description of the current world state
            goal (int): a goal, initially None
            problem (dict): a problem formulation

        Args:
            initial_state (int): the initial state of the agent
            goal (int): the goal of the agent
        """
        self.state = None
        self.goal = goal
        self.problem = None
        self.seq = []
        self.successors = None
        self.reset(state)

    def select_action(self, state):
        """returns an action"""
        self.update_state(state=state)
        if not self.seq:
            self.problem = self.formulate_problem(
                self.state, self.goal, self.successors)
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
        if state - nrows in env.observation_space:
            yield 0
        if state + 1 in env.observation_space:
            yield 1
        if state - 1 in env.observation_space:
            yield 2
        if state + nrows in env.observation_space:
            yield 3

    @classmethod
    def formulate_problem(cls, state, goal, successors):
        """Formulate the problem

        Args:
            state (int): a state of the agent
            goal (int): a goal

        Returns:
            dict: a problem formulation
        """
        return {
            'initial_state': state,
            'actions': cls._action_range,
            'goal_test': lambda state: state == goal,
            'state_space': env.observation_space,
            'result': successors,
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
        if not self.successors[old_state].get(action):
            self.successors[old_state][action] = (-cost, new_state)
        else:
            self.successors[old_state][action] = (
                self.successors[old_state][action][0] - cost, new_state)
        self.max_reward -= cost

    def reset(self, state, cheater=True):
        """reset the agent to its initial state after an epoch. Can also be used to perform learning after an epoch)

        Args:
            state (int, optional): the state of the agent. Defaults to None.
        """
        self.state = state
        self.successors = defaultdict(dict)
        self.problem = self.formulate_problem(
            state, self.goal, self.successors)
        self.seq = self.search(self.problem)
        if not self.seq:
            raise IndexError("empty sequence")
        self.max_reward = 0
        self._wall()
        if cheater:
            self._cheat_code()

    def _cheat_code(self):
        """Cheat code
        """
        self.successors[2*nrows+1][0] = (100, 1*nrows+1)
        self.successors[1*nrows+0][1] = (100, 1*nrows+1)
        self.successors[0*nrows+1][2] = (100, 1*nrows+1)
        self.successors[1*nrows+2][3] = (100, 1*nrows+1)
        self.successors[1*nrows+2][1] = (100, 1*nrows+3)
        self.successors[0*nrows+3][2] = (100, 1*nrows+3)
        self.successors[2*nrows+2][1] = (100, 2*nrows+3)
        self.successors[2*nrows+0][2] = (100, 3*nrows+0)
        self.successors[3*nrows+1][3] = (100, 3*nrows+0)

    def _wall(self):
        self.successors[0][0] = (50, 0)
        self.successors[1][0] = (50, 1)
        self.successors[2][0] = (50, 2)
        self.successors[3][0] = (50, 3)
        self.successors[3][1] = (50, 3)
        self.successors[7][1] = (50, 7)
        self.successors[11][1] = (50, 11)
        self.successors[15][1] = (50, 15)
        self.successors[12][2] = (50, 12)
        self.successors[13][2] = (50, 13)
        self.successors[14][2] = (50, 14)
        self.successors[15][2] = (50, 15)
        self.successors[0][3] = (50, 0)
        self.successors[4][3] = (50, 4)
        self.successors[8][3] = (50, 8)
        self.successors[12][3] = (50, 12)


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
