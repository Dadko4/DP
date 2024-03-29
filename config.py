data_generator_config = {
    "sample_len": 512,
    "batch_size": 1000,
    "quality_threshold": 14,
    "normalize": "MEDIAN",
    "random_sample": True,
    "step_len": 256,
    "load2ram": False,
    "test": False,
    "smooth_w_size": 7
}
load_from_file = True

corrected_group = "RawGenomeCorrected_bwamem_000"

model_config = {
 "num_of_inner_conv": 2,
 "timesteps": 512,
 "input_dim": 1,
 "filter_size": 12,
 "first_and_last": 64,
 "inner": 32,
 "loss": 'mae'
}
seq_path = (r'/tf/puc19/nanopore/MAP_Data/08_07_16_R9_pUC_BC/'
            r'MA/downloads/pass/NB07/*.fast5')
test_seq_path = (r'/tf/puc19/nanopore/MAP_Data/08_07_16_R9_pUC_BC/'
                 r'MA/downloads/pass/NB08/*.fast5')
n_epochs = 100
model_name = "lstm_dense_dense/lstm.h5"
n_validation_baches = 500
tb_logs_path = r'lstm_dense_dense/tb_logs'
model_checkpoint_file = 'lstm_dense_dense/best_model.h5'
