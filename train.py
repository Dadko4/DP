from data_loader import DataGenerator
from datetime import datetime
from model import model_builder
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        ModelCheckpoint)
from tensorflow.keras.backend import clear_session
import numpy as np
import warnings
from config import (model_config, data_generator_config, load_from_file,
                    n_epochs, model_name, test_seq_path, n_validation_baches)
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
batch_size = data_generator_config['batch_size']
if not data_generator_config['load2ram'] and load_from_file:
    sample_len = data_generator_config['sample_len']
    quality_threshold = data_generator_config['quality_threshold']
    normalize = data_generator_config['normalize']
    test = data_generator_config['test']
    data_generator.load_from_file((f"{batch_size}_{sample_len}_"
                                   f"{quality_threshold}_"
                                   f"{normalize}_{test}.npy"))
print(datetime.now() - now)
print(data_generator.data[0].shape)
steps_per_epoch = data_generator.data.shape[0]

gen = generator_wrapper(data_generator)

data_generator_config['load2ram'] = False
data_generator_config['random_sample'] = True
data_generator_config['step_len'] = data_generator_config['sample_len']
test_data_generator = DataGenerator(**data_generator_config,
                                    seq_path=test_seq_path)

val_X = np.array([x for x in next(test_data_generator)
                  for _ in range(n_validation_baches)])

model = model_builder(**model_config)
print(model.summary())

es = EarlyStopping(monitor='val_loss', mode='min', patience=5,
                   restore_best_weights=False)
lr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1,
                          min_lr=0.0001)
mc = ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5',
                     save_best_only=True)
callbacks = [es, lr_cb, mc]

history = model.fit(gen, steps_per_epoch=steps_per_epoch, epochs=n_epochs,
                    callbacks=callbacks, verbose=1,
                    validation_data=(val_X, val_X),
                    validation_steps=n_validation_baches)
model.save(model_name)
