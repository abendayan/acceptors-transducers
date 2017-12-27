import dynet as dn
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
import random
import model_tagger as mt
import pickle

EMBEDDINGS_SIZE = 50
LSTM_NUM_OF_LAYERS = 2
INPUT_DIM = 50
HIDDEN_DIM = 50
EPOCHS = 5
UNK = "UUUNKKK"
start_time = time.time()

def passed_time(previous_time):
    return round(time.time() - previous_time, 3)

def read_file(input_file):
    words = open(input_file, "r").read().split("\n")
    sentences = []
    sentence = []
    for word in words:
        if word == "":
            sentences.append(sentence)
            sentence = []
        else:
            sentence.append(word)
    return sentences

class PredictBiLSTM:
    def __init__(self, type, model_file, input_file):
        self.type = type
        self.input_file = input_file
        self.sequences = read_file(input_file)
        model = dn.Model()
        self.model = dn.load("model_type"+self.type, model)[0]
        self.vocab, self.tags = pickle.load(open(model_file+".vocab", "rb"))
        self.tags_to_ix = { id:tag for tag, id in self.tags.iteritems() }
        self.define_data()

    def word_or_unk(self, word):
        if word not in self.vocab:
            return UNK
        return word

    def define_data(self):
        self.x = [self.prepare_x(sentence) for sentence in self.sequences]

    def prepare_x(self, sequence):
        if self.type == "a" or self.type == "c":
            x = [ self.vocab[self.word_or_unk(word)] for word in sequence ]
        elif self.type == "b":
            # when d, needs to have the words and the char
            x = []
            for word in sequence:
                word = self.word_or_unk(word)
                if word != UNK:
                    x.append([self.chars[char] for char in word ])
                else:
                    x.append([self.chars[word]])
        elif self.type == "d":
            x = []
            for word in sequence:
                word = self.word_or_unk(word)
                if word != UNK:
                    char = [self.chars[char] for char in word ]
                else:
                    char = [self.chars[word]]
                x.append((self.vocab[word], char))
        return x

    def get_probs(self, X):
        dn.renew_cg(True, True)
        return self.model(X)

    def learn(self):
        pred_file = ""
        for j, X in enumerate(self.x):
            start_time = time.time()
            probs = self.get_probs(X)
            for i, prob in enumerate(probs):
                softmax = dn.softmax(prob).npvalue()
                pred = np.argmax(softmax)
                pred_file += self.sequences[j][i] + " " + self.tags_to_ix[pred]
        out_file = open("test4."+self.input_file.split("/")[0], "w")
        out_file.write(pred_file)
        out_file.close()
            # print sentence

if __name__ == '__main__':
    type_word = sys.argv[1]
    model_file = sys.argv[2]
    input_file = sys.argv[3]
    predict = PredictBiLSTM(type_word, model_file, input_file)
    predict.learn()
