import argparse
import os

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import optimizers
import numpy as np
import wandb
from wandb.keras import WandbCallback


def main():
    parser = argparse.ArgumentParser(description='Fit a deep RL model from a training and validation dataset.')
    parser.add_argument('-t', '--train_game_dir', type=str, default="deep_rl_training_games",
        help='directory to load training games from')
    parser.add_argument('-v', '--val_game_dir', type=str, default="deep_rl_validation_games",
        help='directory to load validation games from')
    parser.add_argument('save_model', type=str,
        help='path to save model')
    args = parser.parse_args()

    TRAIN_GAME_FILES_DIR = args.train_game_dir
    VAL_GAME_FILES_DIR = args.val_game_dir

    # Weights & Biases
    wandb.init(project="2048-deep-rl")
    config = wandb.config

    # model architecture
    config.hidden_layer_size = 32
    config.layer_1_size = 8

    # optimizer settings
    config.learn_rate = 0.001
    config.beta_1 = 0.9
    config.beta_2 = 0.999
    config.epsilon = None
    config.decay = 0.0

    # training settings
    config.epochs = 50
    config.batch_size = 32

    # print("tensorflow version =", tf.version.VERSION)
    # print("tf.keras version =", tf.keras.__version__)

    # Set up model architecture/configuration
    value_model = Sequential()
    value_model.add(Dense(config.hidden_layer_size, activation='relu', input_shape=(16,))) # flatten the tiles array
    value_model.add(Dense(config.layer_1_size, activation='relu'))
    value_model.add(Dense(1, activation='relu'))

    # Configure the model learning process, loss function, and other metrics to monitor during training
    optimizer = optimizers.Adam(lr=config.learn_rate, beta_1=config.beta_1,
        beta_2=config.beta_2, epsilon=config.epsilon, decay=config.decay)
    value_model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    # load saved data and labels
    data_train = np.loadtxt(f"{TRAIN_GAME_FILES_DIR}/data_train.csv", delimiter=',', dtype=np.int32)
    labels_train = np.loadtxt(f"{TRAIN_GAME_FILES_DIR}/labels_train.csv", delimiter=',', dtype=np.float64)
    data_val = np.loadtxt(f"{VAL_GAME_FILES_DIR}/data_val.csv", delimiter=',', dtype=np.int32)
    labels_val = np.loadtxt(f"{VAL_GAME_FILES_DIR}/labels_val.csv", delimiter=',', dtype=np.float64)

    # update model weights using the training data and labels
    value_model.fit(data_train, labels_train, epochs=config.epochs, batch_size=config.batch_size,
                    validation_data=(data_val, labels_val), callbacks=[WandbCallback()])

    # Save entire model (weights, model config, optimizer config) to a HDF5 file
    print("==== Saving model to HDF5 file ====")
    value_model.save(args.save_model)
    value_model.save(os.path.join(wandb.run.dir, args.save_model))



    # Restore the saved model, allowing you to resume training from the state where you left off.
    # value_model = tf.keras.models.load_model('value_model.h5')

    """
    # to scale up, use the dataset API
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.batch(32).repeat()

    value_model.fit(dataset, epochs=10, steps_per_epoch=30)
    """

if __name__ == "__main__": main()
