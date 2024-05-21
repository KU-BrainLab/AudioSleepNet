# -*- coding:utf-8 -*-
import os
import glob
import torch
import numpy as np
from typing import List
from torch.utils.data import Dataset


class AudioDataset(Dataset):
    def __init__(self, paths: List,
                 temporal_context_length=40, temporal_context_window_size=40):
        super().__init__()
        self.total_x, self.total_y = self.get_data(paths=paths,
                                                   temporal_context_length=temporal_context_length,
                                                   temporal_context_window_size=temporal_context_window_size)

    def get_data(self, paths, temporal_context_length, temporal_context_window_size):
        from zipfile import BadZipFile
        total_x, total_y = [], []
        for base_path in paths:
            subject_paths = glob.glob(os.path.join(base_path, '*.npz'))
            subject_paths.sort()
            sample_x, sample_y = [], []
            for path in subject_paths[:50]:
                try:
                    data = np.load(path)
                    x, y = data['x'], data['y']
                    sample_x.append(x)
                    sample_y.append(y)
                except BadZipFile:
                    continue
            sample_x, sample_y = np.stack(sample_x), np.stack(sample_y)
            sample_x = self.many_to_many(sample_x, temporal_context_length, temporal_context_window_size)
            sample_y = self.many_to_many(sample_y, temporal_context_length, temporal_context_window_size)
            # print(sample_x.shape)
            # print(sample_y.shape)
            # exit()
            total_x.append(sample_x)
            total_y.append(sample_y)
        total_x, total_y = np.concatenate(total_x), np.concatenate(total_y)
        return total_x, total_y

    @staticmethod
    def many_to_many(elements, temporal_context_length, window_size):
        size = len(elements)
        total = []
        if size <= temporal_context_length:
            return elements
        for i in range(0, size-temporal_context_length+1, window_size):
            temp = np.array(elements[i:i+temporal_context_length])
            total.append(temp)
        total.append(elements[size-temporal_context_length:size])
        total = np.array(total)
        return total

    def __getitem__(self, idx):
        x, y = self.total_x[idx], self.total_y[idx]
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        return x, y

    def __len__(self):
        return self.total_x.shape[0]
