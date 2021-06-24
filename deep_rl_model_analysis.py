import click
from keras.models import load_model
import numpy as np

from game_engine import GameState, Game
from deep_q_learning_experience_replay import convert_tiles_to_bitarray

ACTIONS = ["Up", "Down", "Left", "Right"]

def pretty_print_tiles(game_state):
    output = ""
    for row_idx in range(len(game_state.tiles)):
        row = game_state.tiles[row_idx]
        row_output = ""
        for idx in range(len(row)):
            value = row[idx]
            if idx == len(row) - 1:
                row_output += f"{value:2d}"
            else:
                row_output += f"{value:2d} "
        if row_idx < len(game_state.tiles) - 1:
            row_output += "\n"
        output += row_output
    return output

@click.command()
@click.argument("model_h5_file", type=str)
@click.argument("game_file", type=str)
def main(model_h5_file: str, game_file: str):
    print(f"==== Loading model from {model_h5_file} ====")
    value_model = load_model(model_h5_file)

    with open(game_file, 'r') as f:
        lines = f.readlines()
        game_states = [GameState.from_csv_line(line) for line in lines]

    for current_state in game_states:
        network_input = np.expand_dims(convert_tiles_to_bitarray(current_state.tiles), axis=0)
        network_output = value_model.predict(network_input)[0]
        assert len(network_output) == 4
        # print(f"network output: {network_output}")
        action = np.argmax(network_output)

        print(pretty_print_tiles(current_state))
        print("selected action:", ACTIONS[action])
        print("---")

if __name__ == "__main__":
    main()
