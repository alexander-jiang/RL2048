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


### Deep Q-learning (DQN)
starting with a fully connected NN as the value function approximator

state representation as features:
272 inputs (16 * 17), one 17-length bit array for each
of the 16 cells. The first bit in each bitarray is set if the corresponding cell has a 2-tile,
the second bit if the cell has a 4-tile, and so on (the 17th bit is set if the corresponding cell
has a 2^17-tile, the maximum possible tile value on a 4x4 board which spawns 4-tiles at most)

NN architecture: densely connected (for now)
hidden layer 1: 64 units
hidden layer 2: 4 units

approach:
offline/batch RL, where we play games, then compile training data for the NN
(using the predictions of the previous iteration of the NN as the training labels)
and update the NN weights

TODO:
- implement deep Q-learning with experience replay algorithm from Mnih et al., 2015 (see Algorithm 1)
  - implement custom loss function in Keras to compute loss of DQN over a minibatch of (s,a,s',r) tuples?
  - mini batch size? (i.e. sampling randomly from the original training set)
how to do this in keras?
SGD (stochastic gradient descent) optimizer lets you set a batch_size hyperparam
but RMSprop is mentioned in the Google Atari Deep RL paper (Mnih et al., 2015)

- increase the dataset size? (from just 100 games split 80/20 in train/val to something like 1000 or 10000?)
may need to put this on cloud compute to get this done reasonably quickly
how many games needed to play to make a good training set?



### Instructions

Python 3.7
```
conda create --name RL2048 python=3.7
conda activate RL2048
pip install -r requirements.txt
```

Set up Weights & Biases (wandb): (be sure to set environment variables first, like WANDB_API_KEY)
```
wandb login
```

save model config, generate initial weights, copy weights to a target model,
then build dataset and fit the model
```
python reinforcement_learning/save_model_config.py reinforcement_learning/fc_deep_rl_model_config.json --save-model q_models/Q_model_0.h5
cp q_models/Q_model_0.h5 q_models/target_Q_model.h5
python reinforcement_learning/deep_q_learning_experience_replay.py q_models/Q_model_0.h5 q_models/target_Q_model.h5
python reinforcement_learning/fit_rl_model_fully_connected.py q_models/Q_model_0.h5
```


```
python3 -m reinforcement_learning.save_model_config reinforcement_learning/fc_deep_rl_model_config.json --save-model q_models/Q_model_0.h5
```