# -*- coding: utf-8 -*-

from src.modules import (CHAR_LSTM, MLP, Biaffine, BiLSTM,
                            IndependentDropout, SharedDropout)

import torch
import torch.nn as nn
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)


class Tagger(nn.Module):

    def __init__(self, config, embeddings):
        super(Tagger, self).__init__()
        
        self.config = config
        
        # the embedding layer
        #print(embeddings[20])
        self.embed = nn.Embedding.from_pretrained(embeddings)
   
        #exit()
        #self.embed = nn.Embedding(num_embeddings=config.n_words,
        #                          embedding_dim=config.n_embed)
        #self.feat_embed = nn.Embedding(num_embeddings=config.n_feats,
        #                               embedding_dim=config.n_feat_embed)

        # the char-lstm layer
        self.char_lstm = CHAR_LSTM(n_chars=config.n_chars,
                                   n_embed=config.n_char_embed,
                                   n_out=config.n_char_out)

        
        self.embed_dropout = IndependentDropout(p=config.embed_dropout)

        # the word-lstm layer
        
        input_size = config.n_embed+config.n_char_out
        self.lstm = BiLSTM(input_size=input_size,
                           hidden_size=config.n_lstm_hidden,
                           num_layers=config.n_lstm_layers,
                           dropout=config.lstm_dropout)
        self.lstm_dropout = SharedDropout(p=config.lstm_dropout)

        # the MLP layers
        self.mlp_tag = MLP(n_in=config.n_lstm_hidden*2,
                           n_hidden=config.n_tags,
                           dropout=config.mlp_dropout)
        
        self.pad_index = config.pad_index
        self.unk_index = config.unk_index

        self.reset_parameters()


    
    def reset_parameters(self):
        pass #nn.init.zeros_(self.embed.weight)

    def forward(self, words, chars):
        # get the mask and lengths of given batch
        mask = words.ne(self.pad_index)
        lens = mask.sum(dim=1)
        # set the indices larger than num_embeddings to unk_index
        ext_mask = words.ge(self.embed.num_embeddings)
        ext_words = words.masked_fill(ext_mask, self.unk_index)
        # get outputs from embedding layers
        embed = self.embed(words) #+ self.embed(ext_words)
        #input = torch.LongTensor([20], device="cuda")
        char_embed = self.char_lstm(chars[mask])
        char_embed = pad_sequence(torch.split(char_embed, lens.tolist()), True)
        # concatenate the word and char representations
        embed, char_embed = self.embed_dropout(embed, char_embed)
        x = torch.cat((embed, char_embed), dim=-1)
            
        sorted_lens, indices = torch.sort(lens, descending=True)
        inverse_indices = indices.argsort()

        x = pack_padded_sequence(x[indices], sorted_lens.cpu(), True)
        x = self.lstm(x)
        x, _ = pad_packed_sequence(x, True)
        x = self.lstm_dropout(x)[inverse_indices]

        # apply MLPs to the BiLSTM output states
        s_tag = self.mlp_tag(x)
        
        return s_tag

    @classmethod
    def load(cls, fname):
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        state = torch.load(fname, map_location=device)
        parser = cls(state['config'], state['embeddings'])
        parser.load_state_dict(state['state_dict'])
        parser.to(device)
      
        return parser

    def save(self, fname):
        state = {
            'config': self.config,
            'embeddings': self.embed.weight,
            'state_dict': self.state_dict()
        }
        torch.save(state, fname)
