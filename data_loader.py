import numpy as np
from glob import glob
from fast5_research import Fast5, util
from statsmodels import robust
import random

random.seed(0)
np.random.seed(0)


class DataGenerator:
    def __init__(self, batch_size=50, normalize=None, quality_threshold=0,
                 sample_len=300, step_len=10, load2ram=False, test=False):
        seq_path = (r'C:\Users\dadom\Desktop\pripDP\DP\FAST5\nanopore\MAP_Data'
                    r'\08_07_16_R9_pUC_BC\MA\downloads\pass\NB08\*.fast5')
        files_list = glob(seq_path)
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
            self.actual_signal_generator = iter(self.data)

    def __next__(self):
        if self.load2ram:
            try:
                return next(self.actual_signal_generator)
            except StopIteration:
                return next(self._reset_actual_generator)
        else:
            return self.make_batch()

    def make_batch(self):
        batch = []
        for _ in range(self.batch_size):
            try:
                sample = next(self.actual_signal_generator)
            except (StopIteration, TypeError):
                sample = self._next_data(return_next=True)
            if sample is not None:
                batch.append((sample, sample))
        return batch

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
        while True:
            while True:
                if i + self.sample_len < signal.shape[0]:
                    yield signal[i:i + self.sample_len]
                    i += self.step_len
                elif self.test:
                    from_ = signal.shape[0] - 1 - self.sample_len
                    to = signal.shape[0] - 1
                    yield signal[from_:to]

    def _next_data(self, return_next=False):
        while True:
            try:
                filename = next(self.files_generator)
            except StopIteration:
                filename = self._reset_files_generator(return_next=True)
            with Fast5(filename) as fh:
                signal = fh.get_read(raw=True)
                mean_quality = np.inf
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
            self.actual_signal_generator = self._windows(signal)
            if return_next:
                return next(self.actual_signal_generator)
