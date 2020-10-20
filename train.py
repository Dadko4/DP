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
        X = X.reshape(50, 300, 1)
        yield X, X


now = datetime.now()
data_generator = DataGenerator(sample_len=300, batch_size=50, quality_threshold=20,
                               normalize="MEDIAN", random_sample=True, step_len=100)
data_generator.load_from_file('20_MEDIAN_False.npy')
print(datetime.now() - now)
gen = generator_wrapper(data_generator)

model = model_builder()
print(model.summary())

es = EarlyStopping(monitor='loss', mode='min', patience=5, restore_best_weights=True)
lr_cb = ReduceLROnPlateau(factor=0.2, patience=2, verbose=0, min_lr=0.0001)
callbacks = [es]

model.fit(gen, steps_per_epoch=59761, epochs=2, callbacks=callbacks,
          verbose=1)
