from data_loader import DataGenerator
from datetime import datetime
from model import model_builder
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.backend import clear_session
import numpy as np

np.random.seed(0)
clear_session()


def generator_wrapper(generator):
    while True:
        X = next(generator)
        yield X, X


now = datetime.now()
data_generator = DataGenerator(sample_len=256, batch_size=150, quality_threshold=20,
                               normalize="MEDIAN", random_sample=True, step_len=100)
data_generator.load_from_file('150_256_20_MEDIAN_False.npy')
print(datetime.now() - now)
print(data_generator.data[0].shape)
steps_per_epoch = data_generator.data.shape[0]

gen = generator_wrapper(data_generator)

model = model_builder(256)
print(model.summary())

es = EarlyStopping(monitor='loss', mode='min', patience=5, restore_best_weights=True)
lr_cb = ReduceLROnPlateau(factor=0.2, patience=2, verbose=0, min_lr=0.0001)
callbacks = [es]

model.fit(gen, steps_per_epoch=steps_per_epoch, epochs=10, callbacks=callbacks,
          verbose=1)
model.save('CNN_model.h5')
