from data_loader import DataGenerator
from datetime import datetime
from model import (lstm_builder, cnn_builder, unet_builder,
                   lstm_ae_builder)
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        ModelCheckpoint, TensorBoard)
from tensorflow.keras.backend import clear_session
import numpy as np
import warnings
from config import (model_config, data_generator_config, load_from_file,
                    n_epochs, model_name, n_validation_baches,
                    tb_logs_path, model_checkpoint_file)
from notify_callback import NotifyCallback

                   
warnings.filterwarnings("ignore")
np.random.seed(0)
clear_session()


def generator_wrapper(generator):
    while True:
        X = next(generator)
        yield X, X


now = datetime.now()
print("loading generator...", end=" ")
data_generator = DataGenerator(**data_generator_config)
print("done")
batch_size = data_generator_config['batch_size']
sample_len = data_generator_config['sample_len']
print("loading_data...", end=" ")
if load_from_file:
    quality_threshold = data_generator_config['quality_threshold']
    normalize = data_generator_config['normalize']
    test = data_generator_config['test']
    data_generator.load_from_file((f"1000_{sample_len}_"
                                   f"{quality_threshold}_"
                                   f"{normalize}_{test}_7.npy"),
                                  new_batch_size=batch_size)
print(datetime.now() - now)

val_X = data_generator.get_validation_data(n_validation_baches)
val_X = val_X.reshape(n_validation_baches*batch_size, sample_len, 1)

steps_per_epoch = data_generator.data.shape[0]
gen = generator_wrapper(data_generator)

# model = cnn_builder(**model_config)
# model = cnn2_builder(timesteps=model_config['timesteps'], compile=True)
# model = unet_builder(timesteps=model_config['timesteps'], compile=True)
model = lstm_builder(timesteps=model_config['timesteps'])
# model = lstm_ae_builder(timesteps=model_config['timesteps'], lr=0.0001)
# model.load_weights(r"1304/w_last.h5")

print(model.summary())
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
lr_cb = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, 
                          verbose=1, min_lr=0.0001)
mc = ModelCheckpoint(filepath=model_checkpoint_file, save_best_only=True)
# tb = TensorBoard(log_dir=tb_logs_path, histogram_freq=2)
callbacks = [es, lr_cb, mc, NotifyCallback()]

history = model.fit(gen, steps_per_epoch=steps_per_epoch, epochs=n_epochs,
                    callbacks=callbacks, verbose=1,
                    validation_data=(val_X, val_X),
                    validation_steps=n_validation_baches)
model.save(model_name)
