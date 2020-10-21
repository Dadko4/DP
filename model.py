import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import numpy as np

K.clear_session()
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

np.random.seed(0)


def model_builder(timesteps=300, inner=32, firs_and_last=64, input_dim=1,
                  num_of_inner_conv=3, filter_size=12):
    # initializer = RandomNormal(mean=0.0, stddev=0.05, seed=1)
    input_ = keras.Input(shape=(timesteps, input_dim))
    x = layers.Conv1D(firs_and_last, filter_size, 
                      activation='relu', padding='same')(input_)
    for _ in num_of_inner_conv:
        x = layers.MaxPooling1D(2, padding='same')(x)
        x = layers.Conv1D(inner, filter_size, activation='relu', padding='same')(x)
    encoded = layers.MaxPooling1D(2, padding='same')(x)

    x = layers.Conv1D(inner, filter_size, activation='relu', padding='same')(encoded)
    x = layers.UpSampling1D(2)(x)
    for _ in num_of_inner_conv:
        x = layers.Conv1D(inner, filter_size, activation='relu', padding='same')(x)
        x = layers.UpSampling1D(2)(x)
    decoded = layers.Conv1D(1, filter_size, activation='linear', padding='same')(x)
    autoencoder = keras.Model(input_, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_absolute_error')
    return autoencoder
