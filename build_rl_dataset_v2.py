from __future__ import absolute_import, division, print_function, unicode_literals

import click
from game_engine import GameState, Game
from keras.models import model_from_json
import numpy as np
import wandb


def convert_tiles_to_bitarray(tiles) -> np.ndarray:
    """
    Convert from a 4x4 array, where each cell is the log base 2 value of the tile,
    into a flattened bitarray representation, where each of the 16 cells is represented by 17 bits,
    with the first bit set if the tile value is 2, the second bit set in the tile value is 4,
    and so on up to 2^17 (the maximum possible tile value on a 4x4 board with 4-tiles being
    the maximum possible spawned tile).
    """
    flat_tiles = np.ravel(tiles)
    bitarray_input = np.zeros((16, 17))
    for i in range(16):
        if flat_tiles[i] != 0:
            bitarray_input_idx = flat_tiles[i] - 1
            bitarray_input[i,bitarray_input_idx] = 1
    return np.ravel(bitarray_input)

@click.command()
@click.argument("model_config", type=str)
@click.option("-p", "--two-tile-prob", type=float, default=0.9,
    help='probability of spawning a 2-tile (instead of a 4-tile) after a successful move')
@click.option("-r", "--random-seed", type=int, default=131,
    help="random seed (for reproducibility)")
@click.option("-n", "--num_games", type=int, default=100,
    help="number of total games (training and validation)")
@click.option("-s", "--val_split", type=float, default=0.2,
    help="what fraction of the games should be used for validation")
@click.option("-t", "--train_game_dir", type=str, default="deep_rl_training_games_v2",
    help="directory where training games and data are saved to")
@click.option("-v", "--val_game_dir", type=str, default="deep_rl_validation_games_v2",
    help="directory where validation games and data are saved to")
def main(
    model_config: str,
    two_tile_prob: float,
    random_seed: int,
    num_games: int,
    val_split: float,
    train_game_dir: str,
    val_game_dir: str,
):
    """
    Build training/validation dataset by playing N games.
    model_config - path to a model config JSON file (generated using to_json) to use when generating labels
    """

    # build training dataset by playing N complete games
    data_train = []
    labels_train = []
    data_val = []
    labels_val = []
    TWO_TILE_PROB = two_tile_prob
    RANDOM_SEED = random_seed
    NUM_GAMES = num_games
    VAL_SPLIT = val_split
    NUM_TRAIN_GAMES = int(NUM_GAMES * (1 - VAL_SPLIT))
    TRAIN_GAME_FILES_DIR = train_game_dir
    VAL_GAME_FILES_DIR = val_game_dir

    with open(model_config, 'r') as model_config_file:
        value_model = model_from_json(model_config_file.read())

    # Weights & Biases
    wandb.init(project="2048-deep-rl")

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
                data_train.append(convert_tiles_to_bitarray(game.state.tiles))
                if game.state.game_over:
                    labels_train.append(0) # true value of terminal state is 0 (no further rewards possible)
                    break
            else:
                data_val.append(convert_tiles_to_bitarray(game.state.tiles))
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
                network_input = np.expand_dims(convert_tiles_to_bitarray(successor.tiles), axis=0)
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
