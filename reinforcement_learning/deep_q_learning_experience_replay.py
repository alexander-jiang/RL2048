from __future__ import absolute_import, division, print_function, unicode_literals

import click
from game_engine import GameState, Game
from keras.models import load_model
import numpy as np
import wandb
import random
import os
import time

from .experience_replay_utils import (
    convert_tiles_to_bitarray,
    ExperienceReplay,
    tiles_repr,
)


def linear_anneal_parameter(
    initial_value, final_value, anneal_start_t, anneal_end_t, current_t
):
    if current_t <= anneal_start_t:
        return initial_value
    elif current_t >= anneal_end_t:
        return final_value
    fraction = (current_t - anneal_start_t) / (anneal_end_t - anneal_start_t)
    return fraction * (final_value - initial_value) + initial_value


@click.command()
@click.argument("model_h5_file", type=str)
@click.argument("target_h5_file", type=str)
@click.option(
    "-p",
    "--two-tile-prob",
    type=float,
    default=0.9,
    help="probability of spawning a 2-tile (instead of a 4-tile) after a successful move",
)
@click.option(
    "-r",
    "--random-seed",
    type=int,
    default=131,
    help="random seed (for reproducibility)",
)
# @click.option("-n", "--num-games", type=int, default=100,
#     help="number of total games (training and validation)")
# @click.option("-s", "--val-split", type=float, default=0.2,
#     help="what fraction of the games should be used for validation")
@click.option(
    "-t",
    "--train_game_dir",
    type=str,
    default="deep_rl_training_games_v2",
    help="directory where training games and data are saved to",
)
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

    # for epsilon-greedy action selection
    # set initial epsilon to 1, and then linearly anneal to a lower value
    epsilon_start = 1.0
    epsilon_final = 0.05
    epsilon_anneal_start_t = 1
    epsilon_anneal_end_t = 50_000

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
    burnin_period = 10_000

    # action is encoded as an int in 0..3
    # TODO refactor this to be a global defined in a different file, maybe experience_replay_utils.py?
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

    # TODO abstract away the generation of experience tuples into a generator class?

    replay_memory_ndarray = np.zeros((replay_memory_capacity, 2 * 16 * 17 + 2))
    replay_memory_idx = 0
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
        exp = ExperienceReplay(current_state.tiles, action, new_state.tiles, reward)
        replay_memory_ndarray[replay_memory_idx] = exp.flatten()
        replay_memory_idx = replay_memory_idx + 1
        if replay_memory_idx == replay_memory_capacity:
            replay_memory_idx = 0
        # if reward > 0:
        #     print(f"experience tuple with reward: {exp}")

    # replay_memory_ndarray = np.asarray(replay_memory)
    print("replay_memory shape:", replay_memory_ndarray.shape)
    # assert len(replay_memory) == burnin_period
    # print("writing replay memory to file:")
    # with open("replay_memory_burnin.txt", "w") as burn_in_file:
    #     for i in range(len(replay_memory)):
    #         exp = ExperienceReplay.from_flattened(replay_memory[i])
    #         burn_in_file.write(repr(exp) + "\n\n")

    timesteps_since_last_update = 0
    last_time_check = time.perf_counter()
    for t in range(timesteps_per_episode):
        epsilon = linear_anneal_parameter(
            epsilon_start,
            epsilon_final,
            epsilon_anneal_start_t,
            epsilon_anneal_end_t,
            t,
        )
        fit_verbose = 0
        if (t + 1) % 500 == 0:
            print(f"timestep = {t}, epsilon = {epsilon}")
            new_time = time.perf_counter()
            print(f"avg time per step: {(new_time - last_time_check) / t}")
            fit_verbose = 1

        if game.state.game_over:
            RANDOM_SEED = random.randrange(100_000)
            game = Game()
            game.new_game(random_seed=RANDOM_SEED, game_dir=TRAIN_GAME_FILES_DIR)
            np.random.seed(RANDOM_SEED)
            # print(f"New game (random seed = {RANDOM_SEED})")

        current_state = game.state.copy()
        # print("current state:", current_state.tiles)

        # choose an action (epsilon-greedy)
        epsilon_greedy_roll = np.random.random_sample()
        if epsilon_greedy_roll < epsilon:
            action = np.random.choice(np.arange(4))
            # print("chosen action (randomly):", ACTIONS[action])
        else:
            # choose the "best" action based on current model weights -> Q values
            network_input = np.expand_dims(
                convert_tiles_to_bitarray(current_state.tiles), axis=0
            )
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
        exp = ExperienceReplay(current_state.tiles, action, new_state.tiles, reward)
        replay_memory_ndarray[replay_memory_idx] = exp.flatten()
        replay_memory_idx = replay_memory_idx + 1
        # if reward > 0:
        #     print(f"experience tuple with reward: {exp}")

        # Constrain replay memory capacity
        if replay_memory_idx == replay_memory_capacity:
            replay_memory_idx = 0
        # if len(replay_memory) > replay_memory_capacity:
        #     shift = len(replay_memory) - replay_memory_capacity
        #     replay_memory = replay_memory[shift:]
        # assert len(replay_memory) <= replay_memory_capacity

        # Sample a minibatch of experience tuples from replay memory
        # replay_memory_ndarray = np.asarray(replay_memory)
        # TODO is the minibatch sampled without replacement?
        minibatch_indices = np.random.choice(
            replay_memory_ndarray.shape[0], minibatch_size, replace=False
        )
        minibatch = replay_memory_ndarray[minibatch_indices]
        # print(f"minibatch shape: ", minibatch.shape)
        assert minibatch.shape == (minibatch_size, replay_memory_ndarray.shape[1])

        # Compute the labels for the minibatch based on target Q model (vectorized)
        minibatch_succs = replay_memory_ndarray[
            minibatch_indices, (16 * 17 + 1) : (2 * 16 * 17 + 1)
        ]
        minibatch_rewards = replay_memory_ndarray[minibatch_indices, (2 * 16 * 17 + 1)]
        target_output = target_model.predict(minibatch_succs)
        best_q_values = np.max(target_output, axis=1)
        labels = minibatch_rewards + gamma * best_q_values

        # # Compute the labels for the minibatch based on target Q model
        # labels = np.zeros((minibatch_size,))
        # # print(f"labels shape: ", labels.shape)
        # for j in range(minibatch_size):
        #     # Parse out (s, a, s', r) from the experience tuple
        #     # minibatch_exp = ExperienceReplay.from_flattened(minibatch[j])
        #     successor_bitarray = minibatch[j, (16 * 17 + 1) : (2 * 16 * 17 + 1)]
        #     reward = minibatch[j, (2 * 16 * 17 + 1)]
        #     target_input = np.expand_dims(successor_bitarray, axis=0)
        #     target_output = target_model.predict(target_input)[0]
        #     best_q_value = np.max(target_output)
        #     # TODO check if the successor state is a terminal state: if so, then the label is just the reward
        #     labels[j] = reward + gamma * best_q_value
        # # print(f"labels: ", labels)

        # Perform SGD update on current Q model weights based on minibatch & labels
        minibatch_x = minibatch[:, : (16 * 17)]
        _first_record = minibatch_x[0].reshape((4, 4, 17))
        # print(f"minibatch_x shape = {minibatch_x.shape}, first record = {_first_record}")
        value_model.fit(
            x=minibatch_x, y=labels, batch_size=minibatch_size, verbose=fit_verbose
        )

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


if __name__ == "__main__":
    main()
