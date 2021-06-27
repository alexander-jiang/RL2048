import click
import os
import json

from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Input, Conv1D, Concatenate, Reshape
import wandb

from deep_rl_models.fully_connected_nn import FullyConnectedNNModelFactory


@click.command()
@click.argument("model_config", type=str)
@click.option(
    "--save-model",
    type=str,
    default="",
    help="if not empty, path to save full model (as .h5 file)",
)
def main(model_config: str, save_model: str):
    """
    Save the initialized model configuration as a JSON file, and optionally
    save the full model as a .h5 file.

    model_config - the path to save model configuration (as JSON file)
    """
    # Weights & Biases
    wandb.init(project="2048-deep-rl")
    config = wandb.config

    value_model = FullyConnectedNNModelFactory.get_model(config)

    # Serialize a model to JSON string
    model_json_str = value_model.to_json()
    print("model summary")
    print(value_model.summary())

    # save the JSON to a file
    with open(model_config, "w") as local_file:
        local_file.write(model_json_str)
        # json.dump(model_json_str, local_file)

    wandb.save(model_config)

    # Save entire model (weights, model config, optimizer config) to a HDF5 file
    if len(save_model) > 0:
        print("==== Saving model to HDF5 file ====")
        value_model.save(save_model)
        value_model.save(os.path.join(wandb.run.dir, save_model))


if __name__ == "__main__":
    main()
