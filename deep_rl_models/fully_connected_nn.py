from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras import backend

from .model_factory import ModelFactory


class FullyConnectedNNModelFactory(ModelFactory):
    @classmethod
    def get_model(cls, config):
        # model architecture
        config.layer_1_size = 64
        config.layer_2_size = 4

        # optimizer settings
        config.learn_rate = 0.003
        config.rho = 0.9
        config.momentum = 0.0
        config.epsilon = 1e-07

        # training settings
        config.epochs = 100
        config.batch_size = 32

        # Set up model architecture/configuration
        value_model = Sequential()
        value_model.add(Dense(config.layer_1_size, activation='relu', input_dim=16 * 17))
        value_model.add(Dense(config.layer_2_size, activation='relu'))
        value_model.add(Dense(1, activation='relu'))

        return value_model
