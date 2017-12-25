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
        embedded = [dn.lookup_batch(self.input_lookup, chars) for chars in input_exp]
        return embedded

    def param_collection(self): return self.pc

    @staticmethod
    def from_spec(spec, model):
        embedding_size, vocab_size = spec
        return WordEmbedding(model, embedding_size, vocab_size)


class CharEmbedding(object):
    def __init__(self, model, embedding_size, char_size, hidden_dim):
        pc =  model.add_subcollection()
        self.input_lookup = pc.add_lookup_parameters((char_size, embedding_size))
        self.lstm_c = dn.LSTMBuilder(1, embedding_size, hidden_dim, pc)
        self.pc = pc
        self.spec = (embedding_size, char_size, hidden_dim)

    def __call__(self, X):
        input_chars = [ [dn.lookup(self.input_lookup, char) for char in chars] for chars in X ]
        embedded = None
        state_char = self.lstm_c.initial_state()
        embedded = []
        for input_char in input_chars:
            embedded.append(input_char[len(input_char) - 1])
        return embedded

    def param_collection(self): return self.pc

    @staticmethod
    def from_spec(spec, model):
        embedding_size, char_size, hidden_dim = spec
        return CharEmbedding(model, embedding_size, char_size, hidden_dim)

class PreTrained(object):
    def __init__(self, model, embedding_size, vocab_size, word_vector_file):
        pc =  model.add_subcollection()
        self.input_lookup = pc.add_lookup_parameters((vocab_size, embedding_size))
        vecs = np.loadtxt(word_vector_file)
        self.input_lookup.init_from_array(vecs)
        self.pc = pc
        self.spec = (embedding_size, vocab_size, word_vector_file)

    def __call__(self, input_exp):
        embedded = [dn.lookup(self.input_lookup, chars) for chars in input_exp]
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
        self.char_embedding = CharEmbedding(model, embedding_size, char_size, hidden_dim)

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
