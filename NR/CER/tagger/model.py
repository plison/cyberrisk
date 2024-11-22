# -*- coding: utf-8 -*-

from tagger.metric import Metric
from tagger.utils import CEOR
import numpy as np
import torch, copy
import torch.nn as nn
from torch import log, exp
import torch.nn.functional as F
from scipy.stats import entropy as calc_entropy
from math import log, sqrt
from sklearn.metrics import f1_score

class Model(object):

    def __init__(self, vocab, tagger):
        super(Model, self).__init__()

        self.vocab = vocab
        self.tagger = tagger
        label_tags = copy.deepcopy(self.vocab.tags)
        label_tags.remove("O")
        
        label_tags.remove(self.vocab.UNK)
        label_tags.remove(self.vocab.PAD)
        self.eval_labels = self.vocab.tag2id(label_tags)
        weights = np.ones(len(self.vocab.tag_dict))
        self.outside_class_id = self.vocab.tag_dict["O"]
        weights[self.outside_class_id] = 0.9
        class_weights = torch.FloatTensor(weights)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        #self.criterion = nn.CrossEntropyLoss()

    def calc_label_accuracy(self, g_enc_all, p_enc_all):
        correct = 0.
        tokens = 0.
        gold, pred = [], []
        not_13 = 0
        
        for g_enc, p_enc in zip(g_enc_all, p_enc_all):
            for g, p in zip(g_enc, p_enc):
                #if g.item() == self.outside_class_id:
                #    continue
                if g.item() == p.item():
                    correct += 1.
                tokens += 1
                gold.append(g.item())
                pred.append(p.item())
                if p.item() != 12:
                    not_13 += 1
        #print("not 13", not_13)
        #f1 = correct / tokens
        
        f1 =  f1_score(gold, pred, average='weighted', labels=self.eval_labels) 
        return f1*100
    
    def train(self, loader):
        self.tagger.train()
        
        for words, chars, tags in loader:
            self.optimizer.zero_grad()

            mask = words.ne(self.vocab.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            s_tag = self.tagger(words, chars)
            s_tag = s_tag[mask]
            gold_tags = tags[mask]

            loss = self.get_loss(s_tag, gold_tags)
            loss.backward()
            nn.utils.clip_grad_norm_(self.tagger.parameters(), 5.0)
            self.optimizer.step()
            self.scheduler.step()

    def get_loss(self, s_tag, gold_tags):
        loss = self.criterion(s_tag, gold_tags)
        return loss

    
    @torch.no_grad()
    def evaluate(self, loader, punct=True):
        self.tagger.eval()

        loss, acc = 0, 0.
        all_pred = []
        all_gold = [] 
        for words, chars, tags in loader:
            mask = words.ne(self.vocab.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            s_tag = self.tagger(words, chars)
            s_tag = s_tag[mask]
            pred_tag = self.decode(s_tag)
            gold_tags= tags[mask]
            all_gold.append(gold_tags)
            all_pred.append(pred_tag)
            loss += self.get_loss(s_tag,gold_tags)
        acc = self.calc_label_accuracy(all_gold, all_pred)
        loss /= len(loader)
        return loss, acc

    @torch.no_grad()
    def predict(self, loader):
        self.tagger.eval()

        all_tags, all_rels = [], []
        for words, chars in loader:
            mask = words.ne(self.vocab.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            lens = mask.sum(dim=1).tolist()
            s_tag = self.tagger(words, chars)
            s_tag = s_tag[mask]
            pred_tag = self.decode(s_tag)
            all_tags.extend(torch.split(pred_tag, lens))
        all_tags = [self.vocab.id2tag(seq) for seq in all_tags]
        return all_tags

    def decode(self, s_tag):
        pred_tag =  s_tag.argmax(dim=-1)
        return pred_tag
