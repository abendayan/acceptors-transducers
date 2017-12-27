import dynet as dn
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
import random
import model_tagger as mt

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
        self.sentences = read_file(input_file)
        model = dn.Model()
        self.model = dy.load("model_type"+self.type, model)

    def learn(self):

if __name__ == '__main__':
    type_word = sys.argv[1]
    model_file = sys.argv[2]
    input_file = sys.argv[3]
