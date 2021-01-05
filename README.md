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

### Instructions

Python 3.7
```
conda create --name RL2048 python=3.7
pip install -r requirements.txt
```

Set up Weights & Biases (wandb): (be sure to set environment variables first, like WANDB_API_KEY)
```
wandb login
```

save model config, then build dataset, then fit the model
```
python save_model_config.py fc_deep_rl_model_config.json
python build_rl_dataset_v2.py fc_deep_rl_model_config.json
python fit_rl_model_fully_connected.py value_model_fc.h5
```
