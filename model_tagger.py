import dynet as dn
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
import random

# http://dynet.readthedocs.io/en/latest/python_saving_tutorial.html

class TaggerModel(object):
    def __init__(self, model, embedding_size, hidden_dim, out_size, vocab_size, char_size, type):
        pc =  model.add_subcollection()
        self.lstm_f_1 = dn.LSTMBuilder(1, embedding_size, hidden_dim, pc)
        self.lstm_f_2 = dn.LSTMBuilder(1, 2*hidden_dim, hidden_dim, pc)
        self.lstm_b_1 = dn.LSTMBuilder(1, embedding_size, hidden_dim, pc)
        self.lstm_b_2 = dn.LSTMBuilder(1, 2*hidden_dim, hidden_dim, pc)
        self.output_w = model.add_parameters((out_size, 2*hidden_dim))
        if type == "a":
            self.tagger = WordEmbedding(pc, embedding_size, vocab_size)
        elif type == "b":
            self.tagger = CharEmbedding(pc, embedding_size, char_size)
        elif type == "c":
            self.tagger = WordEmbedding(pc, embedding_size, vocab_size)
        elif type == "d":
            self.tagger = WordCharEmbedding(pc, embedding_size, vocab_size, char_size, hidden_dim)
        self.trainer = dn.AdamTrainer(pc, 0.005)
        self.pc = pc
        self.spec = (embedding_size, hidden_dim, out_size, vocab_size, char_size, type)

    def __call__(self, X):

        embedded = [ self.tagger(word) for word in X ]
        state_back_1 = self.lstm_b_1.initial_state()
        state_forw_1 = self.lstm_f_1.initial_state()
        fw_exps = state_forw_1.transduce(embedded)
        bw_exps = state_back_1.transduce(reversed(embedded))
        # bw_exps.reverse()
        b_1 = [dn.concatenate([f,b]) for f,b in zip(fw_exps, bw_exps)]
        state_back_2 = self.lstm_b_2.initial_state()
        state_forw_2 = self.lstm_f_2.initial_state()
        out_f = state_forw_2.transduce(b_1)
        out_b = state_back_2.transduce(reversed(b_1))
        # out_b.reverse()
        size_vector = len(embedded)
        w = dn.parameter(self.output_w)
        b_2 = [ dn.concatenate([out_f[i], out_b[i]]) for i in range(size_vector) ]
        probs = [ w*b_2_i for b_2_i in b_2 ]
        return probs

    def param_collection(self): return self.pc

    @staticmethod
    def from_spec(spec, model):
        embedding_size, hidden_dim, out_size, vocab_size, char_size, type = spec
        return TaggerModel(model, embedding_size, hidden_dim, out_size, vocab_size, char_size, type)

class WordEmbedding(object):
    def __init__(self, model, embedding_size, vocab_size):
        pc =  model.add_subcollection()
        self.input_lookup = pc.add_lookup_parameters((vocab_size, embedding_size))
        self.pc = pc
        self.spec = (embedding_size, vocab_size)

    def __call__(self, input_exp):
        embedded = dn.lookup_batch(self.input_lookup, [input_exp])
        return embedded

    def param_collection(self): return self.pc

    @staticmethod
    def from_spec(spec, model):
        embedding_size, vocab_size = spec
        return WordEmbedding(model, embedding_size, vocab_size)

class CharEmbedding(object):
    def __init__(self, model, embedding_size, char_size):
        pc =  model.add_subcollection()
        self.e_c = pc.add_lookup_parameters((char_size, embedding_size))
        self.lstm_c = dn.LSTMBuilder(1, embedding_size, embedding_size, pc)
        self.pc = pc
        self.spec = (embedding_size, char_size)

    def __call__(self, X):
        state_char = self.lstm_c.initial_state()
        embedded = state_char.transduce([dn.lookup_batch(self.e_c, [c]) for c in X])
        # take the last output of the lstm
        return embedded[-1]

    def param_collection(self): return self.pc

    @staticmethod
    def from_spec(spec, model):
        embedding_size, char_size = spec
        return CharEmbedding(model, embedding_size, char_size)

class WordCharEmbedding(object):
    def __init__(self, model, embedding_size, vocab_size, char_size, hidden_dim):
        pc =  model.add_subcollection()
        self.spec = (embedding_size, vocab_size, char_size, hidden_dim)
        self.word_embedding = WordEmbedding(model, embedding_size, vocab_size)
        self.char_embedding = CharEmbedding(model, embedding_size, char_size)
        self.pc = pc
        # self.input_lookup = self.pc.add_parameters((embedding_size, embedding_size * 2))
        self.output_w = self.pc.add_parameters((embedding_size, 2*embedding_size))

    def __call__(self, input_exp):
        word_repr = input_exp[0]
        char_repr = input_exp[1]
        embedded_char = self.char_embedding(char_repr)
        embedded_word = self.word_embedding(word_repr)
        out = dn.concatenate([embedded_word, embedded_char])
        w = dn.parameter(self.output_w)
        embedded = w*out
        return embedded

    def param_collection(self): return self.pc

    @staticmethod
    def from_spec(spec, model):
        embedding_size, vocab_size, hidden_dim = spec
        return WordCharEmbedding(model, embedding_size, vocab_size, char_size, hidden_dim)
