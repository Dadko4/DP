import numpy as np
from glob import glob
from fast5_research import Fast5, util
from statsmodels import robust
import random
from tqdm import tqdm
import pickle

random.seed(0)
np.random.seed(0)


class DataGenerator:
    def __init__(self, batch_size=50, normalize=None, quality_threshold=0,
                 sample_len=300, step_len=10, load2ram=False, test=False,
                 random_sample=False):
        seq_path = (r'C:\Users\dadom\Desktop\pripDP\DP\FAST5\nanopore\MAP_Data'
                    r'\08_07_16_R9_pUC_BC\MA\downloads\pass\NB07\*.fast5')
        files_list = glob(seq_path)
        self.files_list = files_list
        if random_sample:
            self.files_list = random.sample(files_list, len(files_list))
        self.files_generator = iter(self.files_list)
        self.epoch = 0
        self.actual_signal_generator = None
        self.batch_size = batch_size
        self.normalize = normalize
        self.quality_threshold = quality_threshold
        self.sample_len = sample_len
        self.step_len = step_len
        self.test = test
        self.load2ram = load2ram
        if load2ram:
            self.data = []
            while True:
                batch = self.make_batch()
                if self.epoch < 1:
                    self.data.append(batch)
                else:
                    self.epoch = 0
                    break
            np.save(f"{self.quality_threshold}_{self.normalize}_{test}", 
                    np.array(self.data))
            self.actual_signal_generator = iter(self.data)

    def __next__(self):
        if self.load2ram:
            try:
                return next(self.actual_signal_generator)
            except StopIteration:
                return self._reset_actual_generator(return_next=True)
        else:
            return self.make_batch()

    def make_batch(self):
        X = []
        for _ in range(self.batch_size):
            try:
                sample = next(self.actual_signal_generator)
            except (StopIteration, TypeError):
                sample = self._next_data(return_next=True)
            if sample is not None:
                X.append(sample.reshape(-1, 1))
        return np.array(X)

    def load_from_file(self, filename):
        with open(filename, 'rb') as f:
            self.data = np.load(f)
        self.load2ram = True
        self.actual_signal_generator = iter(self.data)

    def _reset_files_generator(self, return_next=False):
        self.files_generator = iter(self.files_list)
        self.epoch += 1
        if return_next:
            return next(self.files_generator)

    def _reset_actual_generator(self, return_next=False):
        self.epoch += 1
        self.actual_signal_generator = iter(self.data)
        if return_next:
            return next(self.actual_signal_generator)

    def _windows(self, signal):
        i = 0
        partitioned = []
        while True:
            if i + self.sample_len < signal.shape[0]:
                partitioned.append(signal[i:i + self.sample_len])
                i += self.step_len
            elif self.test and i < signal.shape[0]:
                from_ = signal.shape[0] - 1 - self.sample_len
                to = signal.shape[0] - 1
                partitioned.append(signal[from_:to])
                i += self.step_len
            else:
                return partitioned

    def _next_data(self, return_next=False):
        while True:
            try:
                filename = next(self.files_generator)
            except StopIteration:
                print("all files were iterated")
                filename = self._reset_files_generator(return_next=True)
            with Fast5(filename) as fh:
                signal = fh.get_read(raw=True)
                mean_quality = 1000
                if self.quality_threshold > 0:
                    fastq = fh.get_fastq()
                    quality_str = fastq.decode('UTF-8').split('\n')[3]
                    qualities = [util.qstring_to_phred(char) for char in quality_str]
                    mean_quality = np.mean(qualities)
            if mean_quality < self.quality_threshold:
                continue
            uniq_arr = np.unique(signal)
            if self.normalize == 'MEAN':
                signal = (signal - np.mean(uniq_arr)) / np.float(np.std(uniq_arr))
            elif self.normalize == 'MEDIAN':
                signal = (signal - np.median(uniq_arr)) / np.float(robust.mad(uniq_arr))
            self.actual_signal_generator = iter(self._windows(signal))
            if return_next:
                return next(self.actual_signal_generator)
            break
