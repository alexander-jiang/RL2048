from __future__ import absolute_import, division, print_function, unicode_literals

from game_engine import GameState, Game
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# print("tensorflow version =", tf.version.VERSION)
# print("tf.keras version =", tf.keras.__version__)

value_model = tf.keras.Sequential()
value_model.add(layers.Dense(32, activation='relu', input_shape=(16,))) # flatten the tiles array
value_model.add(layers.Dense(8, activation='relu'))
value_model.add(layers.Dense(1, activation='relu'))


value_model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                    loss='mse',
                    metrics=['mae'])

data = []
labels = []


game = Game()
game.new_game(random_seed=131)
np.random.seed(131)
TWO_TILE_PROB = 0.9

# build training dataset from one complete game
while True:
    print("current state:", game.state.tiles)
    # take current state, choose action, compute successors using value_model
    data.append(np.ravel(game.state.tiles))

    if game.state.game_over:
        labels.append(0) # true value of terminal state is 0
        break

    # TODO better way to choose the next action (epsilon-greedy?)
    possible_moves = sorted(game.state.moves_available()) # sort to allow reproducibility with the same random seed
    # print("possible moves:", possible_moves)
    action = np.random.choice(possible_moves)
    print("chosen action (randomly):", action)

    successors = game.state.successor_states(action, prob_two_tile=TWO_TILE_PROB)
    label = 0
    for (prob, successor, reward) in successors:
        # print(f"successor (prob {prob}): {successor.tiles}")
        network_input = np.expand_dims(np.ravel(successor.tiles), axis=0)
        network_output = value_model.predict(network_input)[0][0]
        # print("network output:", network_output)
        label += prob * (reward + network_output)

    print("label:", label)
    labels.append(label)

    # update current state using the chosen action
    game.move(action)

data = np.asarray(data)
labels = np.asarray(labels)
print("shape of data:", data.shape)
print("shape of labels:", labels.shape)

value_model.fit(data, labels, epochs=10, batch_size=32)

"""
# to scale up, use the dataset API
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32).repeat()

value_model.fit(dataset, epochs=10, steps_per_epoch=30)
"""

# print("model weights:")
# print(value_model.weights)
