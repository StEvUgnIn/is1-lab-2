"""Search algorithms

Raises:
    RecursionError: frontier is empty
    RecursionError: frontier is empty

Returns:
    list[int]: an action sequence

Yields:
    iterable[int]: set of actions
"""

from functools import reduce
from collections import namedtuple
from queue import PriorityQueue

from environment import nrows

Node = namedtuple('Node', ['cost', 'state', 'parent', 'action'])
_maxsize = 140


def solution(node):
    """Returns the action sequence (list)

    Args:
        node (Node): a node

    Yields:
        iterable[int]: an action sequence
    """
    while node.parent:
        yield node.action
        node = node.parent


def child_node(problem, node, action):
    """Child node

    Args:
        problem (dict): formulation of an agent problem
        node (Node): a node is described as a (cost, state) pair
        action (int): action of an agent

    Returns:
        int: Action of the agent
    """
    assert isinstance(problem, dict)
    assert isinstance(node, Node)
    assert isinstance(action, int)
    children = {
        0: Node(*problem['result'][node.state].get(0, (1, node.state - nrows)), node, 0),
        1: Node(*problem['result'][node.state].get(1, (1, node.state + 1)), node, 1),
        2: Node(*problem['result'][node.state].get(2, (1, node.state - 1)), node, 2),
        3: Node(*problem['result'][node.state].get(3, (1, node.state + nrows)), node, 3),
    }
    assert action in children.keys(), "action needs to be in action space"
    assert children[action].state in problem['state_space']
    return children[action]


def tree_search(problem):
    """Returns a solution, or failure

    Args:
        problem (dict): formulation of an agent problem

    Raises:
        RecursionError: frontier is empty

    Returns:
        list[int]: an action sequence
    """
    # initialize the frontier using the initial state of problem
    node = Node(1, problem['initial_state'], None, -1)
    frontier = [node]
    explored = set()
    while True:
        if not frontier:
            raise RecursionError("frontier is empty")
        # choose a leaf node and remove it from the frontier
        node = reduce(
            lambda n1, n2: n1 if n1.cost < n2.cost else n2,
            frontier
        )
        frontier.remove(node)
        # if the node contains a goal state then return the corresponding solution
        if problem['goal_test'](node.state):
            return list(solution(node))
        # expand the chosen node, adding the resulting nodes to the frontier
        explored.add(node.state)
        for action in problem['actions'](node.state):
            child = child_node(problem, node, action)
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier and node.cost > child.cost:
                # replace that frontier node with child
                idx = frontier.index(node)
                frontier[idx] = child


def uniform_cost_search(problem):
    """Returns a solution, or failure

    Args:
        problem (dict): formulation of an agent problem

    Raises:
        RecursionError: frontier is empty

    Returns:
        list[int]: an action sequence
    """
    # node = a node with STATE = problem.START, PATH-COST = 0
    node = Node(1, problem['initial_state'], None, -1)
    # frontier = a priority queue ordered by PATH-COST, with node as the only element
    frontier = PriorityQueue(_maxsize)
    frontier.put(node, block=False)
    explored = set()
    while True:
        if frontier.empty():
            raise RecursionError("frontier is empty")
        # chooses the lowest-cost node in frontier
        node = frontier.get(block=False)
        if problem['goal_test'](node.state):
            return list(solution(node))
        explored.add(node.state)
        for action in problem['actions'](node.state):
            child = child_node(problem, node, action)
            if child.state not in explored and child not in frontier.queue:
                frontier.put(child, block=False)
            elif child in frontier.queue and node.cost > child.cost:
                # replace that frontier node with child
                frontier.queue.remove(node)
                frontier.put(child, block=False)


if __name__ == '__main__':
    from collections import defaultdict
    from environment import env

    def action_range(state):
        """Range of actions

        Args:
            state (int): state of agent
        """
        if state - nrows in env.observation_space:
            yield 0
        if state + 1 in env.observation_space:
            yield 1
        if state - 1 in env.observation_space:
            yield 2
        if state + nrows in env.observation_space:
            yield 3

    result = uniform_cost_search(dict(initial_state=0, actions=action_range, goal_test=lambda state: state ==
                                 15, state_space=env.observation_space, result=defaultdict(dict)))
    print(result)
    result = tree_search(dict(initial_state=0, actions=action_range, goal_test=lambda state: state ==
                         15, state_space=env.observation_space, result=defaultdict(dict)))
    print(result)
