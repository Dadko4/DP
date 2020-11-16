import numpy as np
from glob import glob
from fast5_research import Fast5, util
from statsmodels import robust
import random
import re
from config import seq_path, corrected_group

random.seed(0)
np.random.seed(0)


class DataGenerator:

    def __init__(self, batch_size=50, normalize=None, quality_threshold=0,
                 sample_len=300, step_len=10, load2ram=False, test=False,
                 random_sample=False, seq_path=seq_path,
                 corrected_group=corrected_group, motifs=["CCAGG", "CCTGG", "GATC"]):
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
        self.corrected_group = corrected_group
        self.data = None
        self.motifs = motifs
        if load2ram:
            self.data = []
            while self.epoch == 0:
                batch = self.make_batch()
                if self.epoch < 1:
                    self.data.append(batch)
                else:
                    self.epoch = 0
                    break
            self.data = np.array(self.data)
            np.save(f"{batch_size}_{sample_len}_{quality_threshold}_{normalize}_{test}",
                    self.data)
            self.actual_signal_generator = iter(self.data)

    def __next__(self):
        if self.load2ram:
            try:
                return next(self.actual_signal_generator)
            except StopIteration:
                return self._reset_actual_generator(return_next=True)
        else:
            return self.make_batch()

    def get_validation_data(self, num_of_batches):
        if self.data is None:
            raise ValueError("Cannot return validation data if data isn't loaded")
        idx = np.random.randint(self.data.shape[0], size=num_of_batches)
        validation_data = self.data[idx]
        self.data = np.delete(self.data, idx, axis=0)
        self.actual_signal_generator = iter(self.data)
        return validation_data

    def _is_correct(self, file_):
        subgroups = ['BaseCalled_complement', 'BaseCalled_template']
        is_correct = self.corrected_group in file_['Analyses']
        if not is_correct:
            return False
        corrected_group_keys = file_['Analyses'][self.corrected_group]
        has_subgroups = set(subgroups).issubset(corrected_group_keys)
        return True if is_correct and has_subgroups else False

    def _parse_starts(self, file_):
        """
        Returns starting positions in raw signal corresponding to starts of events
        """
        corr_slot = f"Analyses/{self.corrected_group}/BaseCalled_template"
        corr_data = file_[corr_slot]
        start = corr_data['Events'].attrs['read_start_rel_to_raw']

        corr_slot = f"Analyses/{self.corrected_group}/BaseCalled_complement"
        corr_data = file_[corr_slot]
        r_start = corr_data['Events'].attrs['read_start_rel_to_raw']
        return start, r_start

    def make_batch(self):
        if self.load2ram and self.epoch > 0:
            return None
        X = []
        y = []
        for _ in range(self.batch_size):
            try:
                sample = next(self.actual_signal_generator)
            except (StopIteration, TypeError):
                sample = self._next_data(return_next=True)
            if sample is not None:
                if self.test:
                    X.append(sample[0].reshape(-1, 1))
                    y.append(sample[1])
                else:
                    X.append(sample.reshape(-1, 1))
        if self.test:
            return np.array(X), y
        return np.array(X)

    def load_from_file(self, filename):
        with open(filename, 'rb') as f:
            self.data = np.load(f)
        self.load2ram = True
        self.actual_signal_generator = iter(self.data)

    def _reset_files_generator(self, return_next=False):
        self.files_generator = iter(self.files_list)
        self.epoch += 1
        print("all files were iterated")
        print(self.epoch)
        if return_next:
            return next(self.files_generator)

    def _reset_actual_generator(self, return_next=False):
        self.epoch += 1
        self.actual_signal_generator = iter(self.data)
        if return_next:
            return next(self.actual_signal_generator)

    def _windows(self, signal, events=None):
        if self.test:
            events_list = []
            modif_idx = []
            bases_str = "".join(events['base'].astype(str))
            for mt in self.motifs:
                for m in re.finditer(mt, bases_str):
                    events_list.append([events['start'][m.start() + 1],
                                        events['length'][m.start() + 1]])
            events_list = np.array(events_list)
            events_list = events_list[np.argsort(events_list[:, 0])]
            events_list[:, 0] -= events['start'][0]
            for idx, len_ in events_list:
                modif_idx.extend(list(range(idx, idx+len_)))
            modif_idx = np.array(modif_idx)
        i = 0
        partitioned = []
        modifs = []
        while True:
            if i + self.sample_len < signal.shape[0]:
                if self.test:
                    act = np.arange(i, i + self.sample_len)
                    act_m = [np.arange(act.shape[0])[np.in1d(act, modif_idx)]]
                    modifs.append(act_m)
                partitioned.append(signal[i:i + self.sample_len])
                i += self.step_len
            # elif self.test and i < signal.shape[0]:
            #     from_ = signal.shape[0] - 1 - self.sample_len
            #     to = signal.shape[0] - 1
            #     partitioned.append(signal[from_:to])
            #     i += self.step_len
            else:
                if self.test:
                    return list(zip(partitioned, modifs))
                return partitioned

    def _normalize(self, signal):
        uniq_arr = np.unique(signal)
        if self.normalize == 'MEAN':
            signal = (signal - np.mean(uniq_arr)) / np.float(np.std(uniq_arr))
        elif self.normalize == 'MEDIAN':
            signal = (signal - np.median(uniq_arr)) / np.float(robust.mad(uniq_arr))
        return signal

    def _next_data(self, return_next=False):
        while True:
            try:
                filename = next(self.files_generator)
            except StopIteration:
                filename = self._reset_files_generator(return_next=True)
                if self.load2ram:
                    break
            try:
                fh = Fast5(filename)
            except:
                print(f"unable to open file {filename}")
                continue
            if self._is_correct(fh):
                start, start_r = self._parse_starts(fh)
                t_path = f'Analyses/{self.corrected_group}/BaseCalled_template/Events/'
                templt_stop = fh[t_path]['start'][-1] + fh[t_path]['length'][-1] + start
                c_path = f'Analyses/{self.corrected_group}/BaseCalled_complement/Events/'
                cmplmt_stop = fh[c_path]['start'][-1] + fh[c_path]['length'][-1] + start_r
                tmplt_events = None
                cmplmt_events = None
                if self.test:
                    tmplt_events = np.array(fh[t_path])
                    cmplmt_events = np.array(fh[c_path])
            else:
                continue
            signal = fh.get_read(raw=True)
            mean_quality = 1000
            if self.quality_threshold > 0:
                fastq = fh.get_fastq(analysis="Basecall_2D", section="2D")
                quality_str = fastq.decode('UTF-8').split('\n')[3]
                qualities = [util.qstring_to_phred(char) for char in quality_str]
                mean_quality = np.mean(qualities)
            fh.close()
            if mean_quality < self.quality_threshold:
                continue
            template_signal = self._normalize(signal[start:templt_stop])
            complement_signal = self._normalize(signal[start_r:cmplmt_stop])
            partitioned = self._windows(template_signal, tmplt_events)
            partitioned.extend(self._windows(complement_signal, cmplmt_events))
            self.actual_signal_generator = iter(partitioned)
            if return_next:
                return next(self.actual_signal_generator)
            break
