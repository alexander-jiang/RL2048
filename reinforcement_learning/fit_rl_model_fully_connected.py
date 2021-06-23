import os
import click

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Input, Conv1D, Concatenate, Reshape
from keras import optimizers
from keras import backend
import numpy as np
import wandb
from wandb.keras import WandbCallback

from deep_rl_models.fully_connected_nn import FullyConnectedNNModelFactory


@click.command()
@click.option("-t", "--train-game-dir", type=str, default="deep_rl_training_games_v2",
    help='directory to load training games from')
@click.option("-v", "--val-game-dir", type=str, default="deep_rl_validation_games_v2",
    help='directory to load validation games from')
@click.argument("save_model", type=str)
def main(train_game_dir: str, val_game_dir: str, save_model: str):
    """
    Fit a deep RL model from a training and validation dataset.

    save_model (str) - path to save model
    """
    TRAIN_GAME_FILES_DIR = train_game_dir
    VAL_GAME_FILES_DIR = val_game_dir

    # Weights & Biases
    wandb.init(project="2048-deep-rl")
    config = wandb.config

    # optimizer settings
    config.learn_rate = 0.001
    config.rho = 0.9
    config.momentum = 0.0
    config.epsilon = 1e-07

    # training settings
    config.epochs = 100
    config.batch_size = 32

    value_model = FullyConnectedNNModelFactory.get_model(config)

    # debugging model
    print(value_model.summary())
    # print("model weights:")
    # print(value_model.weights)

    # Configure the model learning process, loss function, and other metrics to monitor during training
    optimizer = optimizers.RMSprop(learning_rate=config.learn_rate, rho=config.rho, momentum=config.momentum, epsilon=config.epsilon)
    value_model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    # load saved data and labels
    data_train = np.loadtxt(f"{TRAIN_GAME_FILES_DIR}/data_train.csv", delimiter=',', dtype=np.int32)
    labels_train = np.loadtxt(f"{TRAIN_GAME_FILES_DIR}/labels_train.csv", delimiter=',', dtype=np.float64)
    data_val = np.loadtxt(f"{VAL_GAME_FILES_DIR}/data_val.csv", delimiter=',', dtype=np.int32)
    labels_val = np.loadtxt(f"{VAL_GAME_FILES_DIR}/labels_val.csv", delimiter=',', dtype=np.float64)

    # testing
    print("testing")
    test_input = np.reshape(data_train[0], (16, 17))
    print("testing with input:", test_input)
    get_output = backend.function([value_model.layers[0].input], [value_model.layers[2].output])
    print("test output:")
    wrapped = np.expand_dims(data_train[0], axis=0)
    print(get_output([wrapped])[0])

    # update model weights using the training data and labels
    value_model.fit(data_train, labels_train, epochs=config.epochs, batch_size=config.batch_size,
                    validation_data=(data_val, labels_val), callbacks=[WandbCallback()])

    # Save entire model (weights, model config, optimizer config) to a HDF5 file
    print("==== Saving model to HDF5 file ====")
    value_model.save(save_model)
    value_model.save(os.path.join(wandb.run.dir, save_model))

    # Restore the saved model, allowing you to resume training from the state where you left off.
    value_model = tf.keras.models.load_model('value_model.h5')

    """
    # to scale up, use the dataset API
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.batch(32).repeat()

    value_model.fit(dataset, epochs=10, steps_per_epoch=30)
    """

if __name__ == "__main__": main()
