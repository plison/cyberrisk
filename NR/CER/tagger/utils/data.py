# -*- coding: utf-8 -*-

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler
import time 

def collate_fn(data):
    reprs = (pad_sequence(i, True) for i in zip(*data))
    if torch.cuda.is_available():
        reprs = (i.cuda() for i in reprs)

    return reprs




class TextDataset(Dataset):
    def __init__(self, items):
        super(TextDataset, self).__init__()
        self.items = items

    def __getitem__(self, index):
        return tuple(item[index] for item in self.items)

    def __len__(self):
        return len(self.items[0])

    @property
    def lengths(self):
        return [len(i) for i in self.items[0]]


def batchify(dataset, batch_size, n_buckets=1, shuffle=False):

    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        collate_fn=collate_fn)

    return loader
