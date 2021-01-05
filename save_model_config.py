import argparse
import os
import json

from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Input, Conv1D, Concatenate, Reshape
import wandb

from deep_rl_models.fully_connected_nn import FullyConnectedNNModelFactory


def main():
    parser = argparse.ArgumentParser(description='Fit a deep RL model from a training and validation dataset.')
    parser.add_argument('model_config', type=str,
        help='path to save model configuration')
    args = parser.parse_args()

    # Weights & Biases
    wandb.init(project="2048-deep-rl")
    config = wandb.config

    value_model = FullyConnectedNNModelFactory.get_model(config)

    # Serialize a model to JSON string
    model_json_str = value_model.to_json()
    print("model summary")
    print(value_model.summary())

    # save the JSON to a file
    with open(args.model_config, 'w') as local_file:
        local_file.write(model_json_str)
        # json.dump(model_json_str, local_file)

    # TODO why does W&B always exit with code 1?
    with open(os.path.join(wandb.run.dir, args.model_config), 'w') as wandb_file:
        wandb_file.write(model_json_str)
        # json.dump(model_json_str, wandb_file)

    # import pdb; pdb.set_trace()

if __name__ == "__main__": main()
exit(0)
