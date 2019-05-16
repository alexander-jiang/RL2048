# RL2048

Reinforcement learning model to play the 2048 game.

- States: the board state
  - features to represent state?
    - total number of tiles for each row & column (8 features, each ranging from 0 to 4)
    - number of rows & columns that will merge (2 features, each ranging from 0 to 4)
- Reward: the points earned by combining blocks
- Actions: the 4 directions

TODO:
- build naive bots (e.g. random moves, or top-left biased random moves) for comparison/prototyping
  - how many turns to get to first 64-tile? to first 128-tile? etc.
  - average number of tiles per turn? average tile value per turn?
  - average weight of duplicated tiles per turn?
  - score vs. turn number
  - distribution of actions taken per game
- implement RL model (Q-learning)
