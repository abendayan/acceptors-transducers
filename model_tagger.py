import dynet as dn
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
import random

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
        vector = [dn.lookup_batch(self.e_c, [c]) for c in X]
        embedded = state_char.transduce(vector)
        # take the last output of the lstm
        return embedded[-1]

    def param_collection(self): return self.pc

    @staticmethod
    def from_spec(spec, model):
        embedding_size, char_size = spec
        return CharEmbedding(model, embedding_size, char_size)

class PreTrained(object):
    def __init__(self, model, embedding_size, vocab_size, word_vector_file):
        pc =  model.add_subcollection()
        self.input_lookup = pc.add_lookup_parameters((vocab_size, embedding_size))
        vecs = np.loadtxt(word_vector_file)
        self.input_lookup.init_from_array(vecs)
        self.pc = pc
        self.spec = (embedding_size, vocab_size, word_vector_file)

    def __call__(self, input_exp):
        embedded = dn.lookup_batch(self.input_lookup, [input_exp])
        return embedded

    def param_collection(self): return self.pc

    @staticmethod
    def from_spec(spec, model):
        embedding_size, vocab_size, word_vector_file = spec
        return PreTrained(model, embedding_size, vocab_size, word_vector_file)

class WordCharEmbedding(object):
    def __init__(self, model, embedding_size, vocab_size, char_size, hidden_dim):
        pc =  model.add_subcollection()
        self.spec = (embedding_size, vocab_size, char_size, hidden_dim)
        self.word_embedding = WordEmbedding(model, embedding_size, vocab_size)
        self.char_embedding = CharEmbedding(model, embedding_size, char_size)

    def __call__(self, input_exp):
        # TODO finish this
        embedded_char = self.char_embedding(input_exp)
        embedded_word = self.word_embedding(input_exp)
        embedded = [dn.concatenate([w,c]) for w,c in zip(embedded_word, embedded_char)]
        return embedded

    def param_collection(self): return self.pc

    @staticmethod
    def from_spec(spec, model):
        embedding_size, vocab_size, hidden_dim = spec
        return WordCharEmbedding(model, embedding_size, vocab_size, char_size, hidden_dim)
