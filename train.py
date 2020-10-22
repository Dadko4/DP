from data_loader import DataGenerator
from datetime import datetime
from model import model_builder
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.backend import clear_session
import numpy as np
import warnings
from config import model_config, data_generator_config, load_from_file
warnings.filterwarnings("ignore")
np.random.seed(0)
clear_session()


def generator_wrapper(generator):
    while True:
        X = next(generator)
        yield X, X


now = datetime.now()
print(now)
data_generator = DataGenerator(**data_generator_config)
if not data_generator_config['load2ram'] and load_from_file:
    batch_size = data_generator_config['batch_size']
    sample_len = data_generator_config['sample_len']
    quality_threshold = data_generator_config['quality_threshold']
    normalize = data_generator_config['normalize']
    test = data_generator_config['test']
    data_generator.load_from_file(f"{batch_size}_{sample_len}_{quality_threshold}_{normalize}_{test}.npy")
print(datetime.now() - now)
print(data_generator.data[0].shape)
steps_per_epoch = data_generator.data.shape[0]

gen = generator_wrapper(data_generator)

model = model_builder(**model_config)
print(model.summary())

es = EarlyStopping(monitor='loss', mode='min', patience=5, 
                   restore_best_weights=True)
lr_cb = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=2, verbose=1,
                          min_lr=0.0001)
callbacks = [es, lr_cb]

history = model.fit(gen, steps_per_epoch=steps_per_epoch, epochs=10,
                    callbacks=callbacks, verbose=1)
model.save('CNN_model.h5')
