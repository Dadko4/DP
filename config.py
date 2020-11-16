data_generator_config = {
    "sample_len": 512,
    "batch_size": 1000,
    "quality_threshold": 14,
    "normalize": "MEDIAN",
    "random_sample": True,
    "step_len": 256,
    "load2ram": True,
    "test": False
}

corrected_group = "RawGenomeCorrected_bwamem_000"

load_from_file = False
model_config = {
 "num_of_inner_conv": 3,
 "timesteps": 512,
 "input_dim": 1,
 "first_and_last": 64,
 "inner": 32,
 "loss": 'mse'
}
seq_path = (r'/tf/puc19/nanopore/MAP_Data/08_07_16_R9_pUC_BC/'
            r'MA/downloads/pass/NB07/*.fast5')
test_seq_path = (r'/tf/puc19/nanopore/MAP_Data/08_07_16_R9_pUC_BC/'
                 r'MA/downloads/pass/NB08/*.fast5')
n_epochs = 30
model_name = "0311/3_layers_CNN_m2e.h5"
n_validation_baches = 1000
tb_logs_path = r'0311/tb_logs'
model_checkpoint_file = '0311/model.{epoch:02d}-{val_loss:.2f}.h5'
