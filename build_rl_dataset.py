from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
from game_engine import GameState, Game
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Build training/validation dataset by playing N games.')
    parser.add_argument('-p', '--two_tile_prob', type=float, default=0.9,
        help='probability of spawning a 2-tile (instead of a 4-tile) after a successful move')
    parser.add_argument('-r', '--random_seed', type=int, default=131,
        help='random seed (for reproducibility)')
    parser.add_argument('-n', '--num_games', type=int, default=100,
        help='number of total games (training and validation)')
    parser.add_argument('-s', '--val_split', type=float, default=0.2,
        help='what fraction of the games should be used for validation')
    parser.add_argument('-t', '--train_game_dir', type=str, default="deep_rl_training_games",
        help='directory where training games and data are saved to')
    parser.add_argument('-v', '--val_game_dir', type=str, default="deep_rl_validation_games",
        help='directory where validation games and data are saved to')
    args = parser.parse_args()

    # build training dataset by playing N complete games
    data_train = []
    labels_train = []
    data_val = []
    labels_val = []
    TWO_TILE_PROB = args.two_tile_prob
    RANDOM_SEED = args.random_seed
    NUM_GAMES = args.num_games
    VAL_SPLIT = args.val_split
    NUM_TRAIN_GAMES = int(NUM_GAMES * (1 - VAL_SPLIT))
    TRAIN_GAME_FILES_DIR = args.train_game_dir
    VAL_GAME_FILES_DIR = args.val_game_dir

    for i in range(NUM_GAMES):
        if i < NUM_TRAIN_GAMES:
            game_dir = TRAIN_GAME_FILES_DIR
        else:
            game_dir = VAL_GAME_FILES_DIR
        print(f"Game {i + 1} of {NUM_GAMES}: {NUM_TRAIN_GAMES} training games")
        game = Game()
        game.new_game(random_seed=RANDOM_SEED + i, game_dir=game_dir)
        np.random.seed(RANDOM_SEED + i)

        while True:
            # training data is a list of game states
            if i < NUM_TRAIN_GAMES:
                data_train.append(np.ravel(game.state.tiles))
                if game.state.game_over:
                    labels_train.append(0) # true value of terminal state is 0 (no further rewards possible)
                    break
            else:
                data_val.append(np.ravel(game.state.tiles))
                if game.state.game_over:
                    labels_val.append(0) # true value of terminal state is 0 (no further rewards possible)
                    break
            # print("current state:", game.state.tiles)

            # choose an action
            # TODO better way to choose the next action (epsilon-greedy?)
            possible_moves = sorted(game.state.moves_available()) # sort to allow reproducibility with the same random seed
            # print("possible moves:", possible_moves)
            action = np.random.choice(possible_moves)
            # print("chosen action (randomly):", action)

            # compute training label based on value of successors (approximated using current model weights)
            successors = game.state.successor_states(action, prob_two_tile=TWO_TILE_PROB)
            label = 0
            for (prob, successor, reward) in successors:
                # print(f"successor (prob {prob}): {successor.tiles}")
                network_input = np.expand_dims(np.ravel(successor.tiles), axis=0)
                network_output = value_model.predict(network_input)[0][0]
                # print("network output:", network_output)
                label += prob * (reward + network_output)

            if i < NUM_TRAIN_GAMES:
                labels_train.append(label)
            else:
                labels_val.append(label)
            # print("label:", label)

            # update current state using the chosen action
            game.move(action)

    # Cast data and labels to numpy arrays
    # TODO use tensorflow Dataset instead?
    data_train = np.asarray(data_train)
    labels_train = np.asarray(labels_train)
    data_val = np.asarray(data_val)
    labels_val = np.asarray(labels_val)
    print("shape of data_train:", data_train.shape)
    # print("dtype of data_train:", data_train.dtype)
    print("shape of labels_train:", labels_train.shape)
    # print("dtype of labels_train:", labels_train.dtype)
    print("shape of data_val:", data_val.shape)
    # print("dtype of data_val:", data_val.dtype)
    print("shape of labels_val:", labels_val.shape)
    # print("dtype of labels_val:", labels_val.dtype)

    # Save training data and labels to csv
    # TODO save to HDF5 or use tensorflow Saver instead?
    print("==== Saving data and labels to csv files ====")
    np.savetxt(f"{TRAIN_GAME_FILES_DIR}/data_train.csv", data_train, delimiter=',')
    np.savetxt(f"{TRAIN_GAME_FILES_DIR}/labels_train.csv", labels_train, delimiter=',')
    np.savetxt(f"{VAL_GAME_FILES_DIR}/data_val.csv", data_val, delimiter=',')
    np.savetxt(f"{VAL_GAME_FILES_DIR}/labels_val.csv", labels_val, delimiter=',')

if __name__ == "__main__": main()
