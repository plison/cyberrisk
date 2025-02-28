# -*- coding: utf-8 -*-

import torch


class Embedding(object):

    def __init__(self, tokens, vectors, unk=None):
        super(Embedding, self).__init__()

        self.tokens = tokens
        self.vectors = vectors
        self.pretrained = {w: v for w, v in zip(tokens, vectors)}
        self.unk = unk

    def __len__(self):
        return len(self.tokens)

    def __contains__(self, token):
        return token in self.pretrained

    def __getitem__(self, token):
        return torch.tensor(self.pretrained[token])

    @property
    def dim(self):
        return len(self.vectors[0])

    @classmethod
    def load(cls, fname, unk=None, dim=100):
        with open(fname, 'r') as f:
            lines = [line for line in f]
        if len(lines[0]) < dim:
            splits_fecked = [line.split() for line in lines[1:]]
        else:
            splits_fecked = [line.split() for line in lines]
        splits = []
        for line in splits_fecked:
            if len(line) > 101:
                length = len(line)
                word_end = length-100
                word = " ".join(line[:word_end])
                numbers = line[word_end:]
                splits.append([word]+numbers)
            else:
                splits.append(line)
        tokens, vectors = zip(*[(s[0], list(map(float, s[1:])))
        for s in splits])
        embedding = cls(tokens, vectors, unk=unk)
            
        return embedding
