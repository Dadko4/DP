from data_loader import DataGenerator
from datetime import datetime
from model import model_builder


def generator_wrapper(generator):
    while True:
        yield generator.make_batch()


now = datetime.now()
data_generator = DataGenerator(sample_len=300, batch_size=50, quality_threshold=16
                               normalize="MEDIAN", random_sample=True)
gen = generator_wrapper(data_generator)

val_set = []
for _ in range(10):
    


model = model_builder()
print(model.summary())



es = EarlyStopping(monitor='loss', mode='min', patience=5, restore_best_weights=True)
lr_cb = ReduceLROnPlateau(factor=0.2, patience=2, verbose=0, min_lr=0.0001)

model.fit(gen, steps_per_epoch=150, epochs=2, callbacks=[es, lr_cb],verbose=1)

