import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (Conv1D, AveragePooling1D, ELU, MaxPooling1D,
                                     UpSampling1D, Bidirectional, LSTM,
                                     RepeatVector, TimeDistributed, Dense,
                                     BatchNormalization, Concatenate, Dropout)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
import numpy as np

K.clear_session()
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

np.random.seed(0)


def cnn_builder(timesteps=300, inner=32, first_and_last=64, input_dim=1,
                  num_of_inner_conv=3, filter_size=12, loss='mse'):
    input_ = keras.Input(shape=(timesteps, input_dim))
    x = Conv1D(first_and_last, filter_size, 
                      activation=ELU(alpha=1), padding='same')(input_)
    for _ in range(num_of_inner_conv):
        x = AveragePooling1D(2, padding='same')(x)
        x = Conv1D(inner, filter_size, activation=ELU(alpha=1), padding='same')(x)
    encoded = AveragePooling1D(2, padding='same')(x)

    middle = Conv1D(1, filter_size, activation=ELU(alpha=1), padding='same')(encoded)
    
    x = UpSampling1D(2)(middle)
    for _ in range(num_of_inner_conv - 1):
        x = Conv1D(inner, filter_size, activation=ELU(alpha=1), padding='same')(x)
        x = UpSampling1D(2)(x)
    x = Conv1D(first_and_last, filter_size, activation=ELU(alpha=1), padding='same')(x)
    x = UpSampling1D(2)(x)
    decoded = Conv1D(1, filter_size, activation='linear', padding='same')(x)
    autoencoder = keras.Model(input_, decoded)
    autoencoder.compile(optimizer='adam', loss=loss)
    return autoencoder


def lstm_ae_builder(timesteps, loss='mae', compile=True, lr=0.001, **kwargs):
    opt = Adam(learning_rate=lr, clipnorm=1.0)
    input_ = keras.Input(shape=(timesteps, 1))
    enc = Bidirectional(LSTM(100, return_sequences=True))(input_)
    enc = Bidirectional(LSTM(64, return_sequences=False))(enc)
    rv = RepeatVector(timesteps)(enc)
    dec = Bidirectional(LSTM(64, return_sequences=True))(rv)
    dec = Bidirectional(LSTM(100, return_sequences=True))(dec)
    decoded = TimeDistributed(Dense(1))(dec)
    autoencoder = keras.Model(input_, decoded)
    if compile:
        autoencoder.compile(optimizer=opt, loss='mae')
    return autoencoder


def lstm_builder(timesteps, loss='mae', **kwargs):
    opt = Adam(clipnorm=1.0)
    input_ = keras.Input(shape=(timesteps, 1))
    x = Conv1D(64, 7, padding='same')(input_)
    x = ELU(alpha=1)(x)
    x = AveragePooling1D(2, padding='same')(x)
    
    x = Conv1D(64, 3, padding='same')(x)
    x = ELU(alpha=1)(x)
    x = AveragePooling1D(2, padding='valid')(x)
    
    encoded = LSTM(128)(x)
    
    x = Dense(256)(encoded)
    x = ELU(alpha=1)(x)
    
    decoded_seq = Dense(timesteps)(x)
    autoencoder = keras.Model(input_, decoded_seq)
    autoencoder.compile(optimizer=opt, loss=loss)
    return autoencoder












######### potialto ukazujeme (mame odskusane)

def cnnlstm_builder(timesteps, loss='mse', **kwargs):
    input_ = keras.Input(shape=(timesteps, 1))
    x = Conv1D(64, 7, strides=2, padding='same')(input_)
    x = ELU(alpha=1)(x)  
    x = Conv1D(64, 7, padding='same')(x)
    x = ELU(alpha=1)(x)
    x = AveragePooling1D(2)(x)
    
    x = Conv1D(32, 3, padding='same')(input_)
    x = ELU(alpha=1)(x)  
    x = Conv1D(32, 3, padding='same')(x)
    x = ELU(alpha=1)(x)
    x = AveragePooling1D(2)(x)
    
    encoded = LSTM(128, return_sequence=True)(x)
    
    x = Conv1D(32, 3, padding='same')(encoded)
    x = ELU(alpha=1)(x)  
    x = Conv1D(32, 3, padding='same')(x)
    x = ELU(alpha=1)(x)
    x = UpSampling1D(2)(x)
    
    x = Conv1D(32, 7, padding='same')(input_)
    x = ELU(alpha=1)(x)  
    x = Conv1D(64, 7, padding='same')(x)
    x = ELU(alpha=1)(x)
    x = UpSampling1D(2)(x)
    
    x = Conv1D(1,  padding='same')(x)
    autoencoder = keras.Model(input_, x)
    autoencoder.compile(optimizer='adam', loss=loss)
    return autoencoder


def unet_builder(timesteps, dropout=False, loss='mae', compile=False, **kwargs):
    
    def down_block(filters, x, name="down_block"):
        x = Conv1D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = ELU(alpha=1)(x)
        x = Conv1D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        elu = ELU(alpha=1)(x)
        return elu
    
    def up_block(filters, input1, input2, name="up_block"):
        x = Conv1D(filters, 3, padding='same')(input1)
        x = ELU(alpha=1)(x)
        x = Concatenate(axis=2)([input2, x])
        x = Conv1D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = ELU(alpha=1)(x)
        x = Conv1D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        elu = ELU(alpha=1)(x)
        return elu
    
    input_ = keras.Input(shape=(timesteps, 1))

    first = down_block(16, input_)
    x = MaxPooling1D(4, padding='same')(first)
    second = down_block(32, x)
    x = MaxPooling1D(4, padding='same')(second)
    third = down_block(64, x)
    x = MaxPooling1D(4, padding='same')(third)
    fourth = down_block(128, x)
    x = MaxPooling1D(4, padding='same')(fourth)
    fifth = down_block(256, x)
    
    x = UpSampling1D(4)(fifth)
    x = up_block(128, x, fourth)
    x = UpSampling1D(4)(x)
    x = up_block(64, x, third)
    x = UpSampling1D(4)(x)
    x = up_block(32, x, second)
    x = UpSampling1D(4)(x)
    x = up_block(16, x, first)
    
    out = Conv1D(1, 1, padding='same')(x)

    unet = keras.Model(input_, out)
    if compile:
        unet.compile(optimizer='adam', loss=loss)
    return unet
