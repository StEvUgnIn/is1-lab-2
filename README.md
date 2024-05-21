# Intelligent Systems-1 assignment #2

## A simple problem-solving agent

We take Gymnasium [Ciff Walking](https://gymnasium.farama.org/environments/toy_text/cliff_walking/) game as the environment setup.

The description of the environment is the following:

### States

The observation is a value representing the player’s current position as current_row * nrows + current_col (where both the row and col start at 0).

For example, the stating position can be calculated as follows: 3 * 12 + 0 = 36.

The observation is represented by type `int`.

### Initial state

The episode starts with the player in state `[36]` (location [3, 0]).

### Actions

The action shape is `(1,)` in the range `{0, 3}` indicating which direction to move the player.

- 0: Move up
- 1: Move right
- 2: Move down
- 3: Move left

### Transition model

— 

### Goal test

Check whether a state is the state`[47]` (location [3, 11]).

### Path cost

Sum of the weights of the graph edges
