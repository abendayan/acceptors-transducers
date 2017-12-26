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
START = "<s>"
END = "</s>"
start_time = time.time()

def passed_time(previous_time):
    return round(time.time() - previous_time, 3)

class PredictBiLSTM:
    def __init__(self, type, model_file, folder_name):
        self.type = type
        self.folder_name = folder_name


if __name__ == '__main__':
    type_word = sys.argv[1]
    model_file = sys.argv[2]
    folder_name = sys.argv[3]
