import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import numpy as np

K.clear_session()
gpus= tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

np.random.seed(1)

def model_builder():
    # initializer = RandomNormal(mean=0.0, stddev=0.05, seed=1)
    timesteps = 300
    input_dim = 1
    latent_dim = 150

    inputs = keras.Input(shape=(timesteps, input_dim))
    encoded = layers.LSTM(latent_dim)(inputs)

    decoded = layers.RepeatVector(timesteps)(encoded)
    decoded = layers.LSTM(input_dim, return_sequences=True)(decoded)

    sequence_autoencoder = keras.Model(inputs, decoded)
    sequence_autoencoder.compile(Adam())
    encoder = keras.Model(inputs, encoded)
    return sequence_autoencoder
