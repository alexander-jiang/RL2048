from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras import backend

from .model_factory import ModelFactory


class FullyConnectedNNModelFactory(ModelFactory):
    @classmethod
    def get_model(cls, config):
        # model architecture
        config.layer_1_size = 128
        config.layer_2_size = 32

        # Set up model architecture/configuration
        value_model = Sequential()
        value_model.add(Dense(config.layer_1_size, activation='relu', input_dim=16 * 17))
        value_model.add(Dense(config.layer_2_size, activation='relu'))
        value_model.add(Dense(4, activation='relu'))
        # 4 outputs, one for each possible action (in this order: up, down, left, right)

        return value_model
