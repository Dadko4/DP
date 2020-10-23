data_generator_config = {
    "sample_len": 512,
    "batch_size": 600,
    "quality_threshold": 19,
    "normalize": "MEDIAN",
    "random_sample": True,
    "step_len": 50,
    "load2ram": False,
    "test": False
}
load_from_file = True
model_config = {
 "num_of_inner_conv": 3,
 "timesteps": 512,
 "input_dim": 1,
 "first_and_last": 64,
 "inner": 32
}
seq_path = (r'/tf/puc19/nanopore/MAP_Data/08_07_16_R9_pUC_BC/'
            r'MA/downloads/pass/NB07/*.fast5')
n_epochs = 25
model_name = "3_layers_CNN.h5"
