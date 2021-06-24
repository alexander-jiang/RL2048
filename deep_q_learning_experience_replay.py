from __future__ import absolute_import, division, print_function, unicode_literals

import click
from game_engine import GameState, Game
from keras.models import load_model
import numpy as np
import wandb
import random
import os


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

def parse_flattened_experience_tuple(flat_tuple: np.ndarray):
    assert flat_tuple.shape == (2 * 16 * 17 + 2,)

    current_state_bitarray = flat_tuple[0:(16 * 17)]
    action = flat_tuple[16 * 17]
    new_state_bitarray = flat_tuple[(16 * 17 + 1):(2 * 16 * 17 + 1)]
    reward = flat_tuple[(2 * 16 * 17 + 1)]
    return (current_state_bitarray, action, new_state_bitarray, reward)

@click.command()
@click.argument("model_h5_file", type=str)
@click.argument("target_h5_file", type=str)
@click.option("-p", "--two-tile-prob", type=float, default=0.9,
    help='probability of spawning a 2-tile (instead of a 4-tile) after a successful move')
@click.option("-r", "--random-seed", type=int, default=131,
    help="random seed (for reproducibility)")
# @click.option("-n", "--num-games", type=int, default=100,
#     help="number of total games (training and validation)")
# @click.option("-s", "--val-split", type=float, default=0.2,
#     help="what fraction of the games should be used for validation")
@click.option("-t", "--train_game_dir", type=str, default="deep_rl_training_games_v2",
    help="directory where training games and data are saved to")
# @click.option("-v", "--val_game_dir", type=str, default="deep_rl_validation_games_v2",
#     help="directory where validation games and data are saved to")
def main(
    model_h5_file: str,
    target_h5_file: str,
    two_tile_prob: float,
    random_seed: int,
    # num_games: int,
    # val_split: float,
    train_game_dir: str,
    # val_game_dir: str,
):
    """
    Build training/validation dataset by playing N games.
    model_h5_file - path to a model saved in .h5 file to use when selecting
    actions (via epsilon-greedy method), etc.
    target_h5_file - path to a model saved in .h5 file to use when computing
    labels for experience tuples
    """

    # build training dataset by playing N complete games
    data_train = []
    labels_train = []
    data_val = []
    labels_val = []
    TWO_TILE_PROB = two_tile_prob
    RANDOM_SEED = random_seed
    # NUM_GAMES = num_games
    # VAL_SPLIT = val_split
    # NUM_TRAIN_GAMES = int(NUM_GAMES * (1 - VAL_SPLIT))
    TRAIN_GAME_FILES_DIR = train_game_dir
    # VAL_GAME_FILES_DIR = val_game_dir

    print(f"==== Loading model from {model_h5_file} ====")
    value_model = load_model(model_h5_file)

    print(f"==== Loading target model from {target_h5_file} ====")
    target_model = load_model(target_h5_file)

    # Weights & Biases
    wandb.init(project="2048-deep-rl")

    epsilon = 0.05 # for epsilon-greedy action selection
    # TODO set initial epsilon to 1, and then linearly anneal to a lower value like 0.05 or 0.1 over some number of timesteps

    # discount factor
    gamma = 0.99

    # hold the target Q-model fixed for this many timesteps before updating with minibatch
    target_update_delay = 10_000

    # save value model
    value_model_save_period = 1_000

    # Each SGD update is calculated over this many experience tuples (sampled randomly from the replay memory)
    minibatch_size = 32

    # SGD updates are sampled from this number of the most recent experience tuples
    replay_memory_capacity = 10_000

    # how many timesteps of learning per episode
    timesteps_per_episode = 100_000

    # populate replay memory by using a uniform random policy for this many timesteps before learning starts
    burnin_period = 1_000

    # action is encoded as an int in 0..3
    ACTIONS = ["Up", "Down", "Left", "Right"]

    value_model.compile(optimizer="sgd", loss="mean_squared_error")

    # Generate the experience tuples to fill replay memory for one episode of training
    #
    # replay memory format:
    # - state,
    # - action,
    # - reward,
    # - max Q-value of successor state based on target network,
    # - Q(state,action) based on current network
    replay_memory = []
    game = Game()
    game.new_game(random_seed=RANDOM_SEED, game_dir=TRAIN_GAME_FILES_DIR)
    np.random.seed(RANDOM_SEED)
    print(f"New game (random seed = {RANDOM_SEED})")

    for t in range(burnin_period):
        if game.state.game_over:
            RANDOM_SEED = random.randrange(100_000)
            game = Game()
            game.new_game(random_seed=RANDOM_SEED, game_dir=TRAIN_GAME_FILES_DIR)
            np.random.seed(RANDOM_SEED)
            print(f"New game (random seed = {RANDOM_SEED})")

        current_state = game.state.copy()
        # print("current state:", current_state.tiles)

        # choose an action uniformly at random during the burn-in period (to initially populate the replay memory)
        action = np.random.choice(np.arange(4))

        # update current state using the chosen action
        game.move(ACTIONS[action])
        new_state = game.state.copy()
        # print("new state:", new_state.tiles)
        reward = new_state.score - current_state.score

        # save the (s,a,s',r) experience tuple (flattened) to replay memory
        experience_tuple = []
        experience_tuple.extend(convert_tiles_to_bitarray(current_state.tiles))
        experience_tuple.append(action)
        experience_tuple.extend(convert_tiles_to_bitarray(current_state.tiles))
        experience_tuple.append(reward)
        # print(f"experience tuple: {experience_tuple}")

        replay_memory.append(experience_tuple)

    replay_memory_ndarray = np.asarray(replay_memory)
    print("replay_memory shape:", replay_memory_ndarray.shape)
    assert len(replay_memory) == burnin_period

    timesteps_since_last_update = 0
    for t in range(timesteps_per_episode):
        print(f"timestep = {t}")
        if game.state.game_over:
            RANDOM_SEED = random.randrange(100_000)
            game = Game()
            game.new_game(random_seed=RANDOM_SEED, game_dir=TRAIN_GAME_FILES_DIR)
            np.random.seed(RANDOM_SEED)
            print(f"New game (random seed = {RANDOM_SEED})")

        current_state = game.state.copy()
        # print("current state:", current_state.tiles)

        # choose an action (epsilon-greedy)
        epsilon_greedy_roll = np.random.random_sample()
        if epsilon_greedy_roll < epsilon:
            action = np.random.choice(np.arange(4))
            print("chosen action (randomly):", ACTIONS[action])
        else:
            # choose the "best" action based on current model weights -> Q values
            network_input = np.expand_dims(convert_tiles_to_bitarray(current_state.tiles), axis=0)
            network_output = value_model.predict(network_input)[0]
            assert len(network_output) == 4
            # print(f"network output: {network_output}")
            action = np.argmax(network_output)
            # print("chosen action (best):", ACTIONS[action])

        # update current state using the chosen action
        game.move(ACTIONS[action])
        new_state = game.state.copy()
        # print("new state:", new_state.tiles)
        reward = new_state.score - current_state.score

        # save the (s,a,s',r) experience tuple (flattened) to replay memory
        experience_tuple = []
        experience_tuple.extend(convert_tiles_to_bitarray(current_state.tiles))
        experience_tuple.append(action)
        experience_tuple.extend(convert_tiles_to_bitarray(new_state.tiles))
        experience_tuple.append(reward)
        # print(f"experience tuple: {experience_tuple}")

        replay_memory.append(experience_tuple)

        # Constrain replay memory capacity
        if len(replay_memory) > replay_memory_capacity:
            shift = len(replay_memory) - replay_memory_capacity
            replay_memory = replay_memory[shift:]
        assert len(replay_memory) <= replay_memory_capacity

        # Sample a minibatch of experience tuples from replay memory
        replay_memory_ndarray = np.asarray(replay_memory)
        # TODO is the minibatch sampled without replacement?
        minibatch_indices = np.random.choice(replay_memory_ndarray.shape[0], minibatch_size, replace=False)
        minibatch = replay_memory_ndarray[minibatch_indices]
        # print(f"minibatch shape: ", minibatch.shape)
        assert minibatch.shape == (minibatch_size, replay_memory_ndarray.shape[1])

        # Compute the labels for the minibatch based on target Q model
        # TODO vectorize this calculation?
        labels = np.zeros((minibatch_size,))
        # print(f"labels shape: ", labels.shape)
        for j in range(minibatch_size):
            # Parse out (s, a, s', r) from the (flattened) experience tuple
            _, _, new_state_bitarray, reward = parse_flattened_experience_tuple(minibatch[j])
            target_input = np.expand_dims(new_state_bitarray, axis=0)
            target_output = target_model.predict(target_input)[0]
            best_q_value = np.max(target_output)
            labels[j] = reward + gamma * best_q_value
        # print(f"labels: ", labels)

        # Perform SGD update on current Q model weights based on minibatch & labels
        minibatch_x = minibatch[:, :(16 * 17)]
        _first_record = minibatch_x[0].reshape((4, 4, 17))
        # print(f"minibatch_x shape = {minibatch_x.shape}, first record = {_first_record}")
        value_model.fit(x=minibatch_x, y=labels, batch_size=minibatch_size, verbose=1)

        model_h5_filename = os.path.splitext(model_h5_file)[0]
        model_h5_out = f"{model_h5_filename}_{t}.h5"
        if t % value_model_save_period == 0 and t > 0:
            print(f"==== Saving value model to {model_h5_out} ====")
            value_model.save(model_h5_out)

        # Only update the target model to match the current Q model every C timesteps
        timesteps_since_last_update += 1
        if timesteps_since_last_update >= target_update_delay:
            timesteps_since_last_update = 0

            # update the target model
            model_h5_out = f"{model_h5_filename}_{t}.h5"
            value_model.save(model_h5_out)
            target_model = load_model(model_h5_out)
            target_h5_filename = os.path.splitext(target_h5_file)[0]
            target_h5_out = f"{target_h5_filename}_{t}.h5"
            print(f"==== Saving target model to {target_h5_out} ====")
            target_model.save(target_h5_out)

    # # Cast data and labels to numpy arrays
    # # TODO use tensorflow Dataset instead?
    # data_train = np.asarray(data_train)
    # labels_train = np.asarray(labels_train)
    # data_val = np.asarray(data_val)
    # labels_val = np.asarray(labels_val)
    # print("shape of data_train:", data_train.shape)
    # # print("dtype of data_train:", data_train.dtype)
    # print("shape of labels_train:", labels_train.shape)
    # # print("dtype of labels_train:", labels_train.dtype)
    # print("shape of data_val:", data_val.shape)
    # # print("dtype of data_val:", data_val.dtype)
    # print("shape of labels_val:", labels_val.shape)
    # # print("dtype of labels_val:", labels_val.dtype)
    #
    # # Save training data and labels to csv
    # # TODO save to HDF5 or use tensorflow Saver instead?
    # print("==== Saving data and labels to csv files ====")
    # np.savetxt(f"{TRAIN_GAME_FILES_DIR}/data_train.csv", data_train, delimiter=',')
    # np.savetxt(f"{TRAIN_GAME_FILES_DIR}/labels_train.csv", labels_train, delimiter=',')
    # np.savetxt(f"{VAL_GAME_FILES_DIR}/data_val.csv", data_val, delimiter=',')
    # np.savetxt(f"{VAL_GAME_FILES_DIR}/labels_val.csv", labels_val, delimiter=',')

if __name__ == "__main__": main()
