# RL2048

Reinforcement learning model to play the 2048 game.

- States: the board state
  - features to represent state?
    - total number of tiles for each row & column (8 features, each ranging from 0 to 4)
    - number of rows & columns that will merge (2 features, each ranging from 0 to 4)
- Reward: the points earned by combining blocks
- Actions: the 4 directions

TODO:
- finish simulator
  - implement random seed in simulator so that added tiles can be deterministic (for comparison)
- implement recorder to document each session
  - save game state by encoding each turn as a comma-separated line?
- build naive bots (e.g. random moves, or top-left biased random moves) for comparison/prototyping
- implement RL model (Q-learning)
