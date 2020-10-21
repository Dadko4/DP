data_generator_config = {
    "sample_len": 256,
    "batch_size": 600,
    "quality_threshold": 20,
    "normalize": "MEDIAN",
    "random_sample": True,
    "step_len": 50,
    "load2ram": False
}
load_from_file = True
model_config = {
 "num_of_inner_conv": 4,
 "timesteps": 256,
 "input_dim": 1,
 "first_and_last": 64,
 "inner": 32
}
seq_path = (r'C:\Users\dadom\Desktop\pripDP\DP\FAST5\nanopore\MAP_Data'
            r'\08_07_16_R9_pUC_BC\MA\downloads\pass\NB07\*.fast5')