import dynet as dn
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
import random

# https://github.com/neulab/dynet-benchmark/blob/master/dynet-py/bilstm-tagger.py
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

def parse_file(name_file):
    words = open(name_file, "r").read().split("\n")
    sequences = []
    sequence = [(START, START), (START, START)]
    vocab = { UNK: 0, START: 1, END: 2 }
    tags = { START: 0, END: 1 }
    chars = { UNK: 0, START: 1, END: 2 }
    for word_tag in words:
        if word_tag == "":
            sequence.append((END, END))
            sequence.append((END, END))
            sequences.append(sequence)
            sequence = [(START, START), (START, START)]
        else:
            word, tag = word_tag.split(" ")
            if tag not in tags:
                tags[tag] = len(tags)
            if word not in vocab:
                vocab[word] = len(vocab)
            for char in word:
                if char not in chars:
                    chars[char] = len(chars)
            sequence.append((word, tag))
    return sequences, vocab, tags, chars

class TaggerBiLSTM:
    def __init__(self, name_file, dev_file, type="a"):
        self.sequences, self.vocab, self.tags, self.chars = parse_file(name_file)
        if self.type == "c":
            self.vocab = open("vocab.txt", r).read().split("\n")
        print "number of sentences " + str(len(self.sequences))
        self.sequences_dev = parse_file(dev_file)[0]
        print "defined all of the data in " + str(passed_time(start_time))
        self.vocab_size = len(self.vocab)
        self.type = type
        # TODO all of the stuff here are for a, needs to adapt for the other options
        self.model = dn.Model()
        self.input_lookup = self.model.add_lookup_parameters((self.vocab_size, EMBEDDINGS_SIZE))
        if self.type == "b":
            self.lstm_c = dn.LSTMBuilder(1, EMBEDDINGS_SIZE, HIDDEN_DIM, self.model)
        self.lstm_f_1 = dn.LSTMBuilder(1, INPUT_DIM, HIDDEN_DIM, self.model)
        self.lstm_f_2 = dn.LSTMBuilder(1, 100, HIDDEN_DIM, self.model)
        self.lstm_b_1 = dn.LSTMBuilder(1, INPUT_DIM, HIDDEN_DIM, self.model)
        self.lstm_b_2 = dn.LSTMBuilder(1, 100, HIDDEN_DIM, self.model)
        self.output_w = self.model.add_parameters((self.vocab_size, LSTM_NUM_OF_LAYERS*EMBEDDINGS_SIZE))
        self.output_b = self.model.add_parameters((self.vocab_size))
        self.trainer = dn.AdamTrainer(self.model)
        random.shuffle(self.sequences)

    def word_or_unk(self, word):
        if word not in self.vocab:
            return UNK
        return word

    def prepare_sequence(self, sequence):
        if self.type == "a":
            x = [ self.vocab[self.word_or_unk(word)] for (word, tag) in sequence ]
        elif self.type == "b":
            x = []
            for (word, _) in sequence:
                word = self.word_or_unk(word)
                if word not in [UNK, START, END]:
                    x.append([self.chars[char] for char in word ])
                else:
                    x.append([self.chars[word]])
        y = [ self.tags[tag] for (word, tag) in sequence ]
        return x, y

    def validate(self):
        good = 0.0
        bad = 0.0
        for i, sequence in enumerate(self.sequences_dev):
            X, Y = self.prepare_sequence(sequence)
            b_1 = self.first_layer(X)
            b_2 = self.second_layer(b_1)
            for j in range(len(X)):
                probs = self.get_probs(b_2, j).npvalue()
                pred = np.argmax(probs)
                label = Y[j]
                if pred == label:
                    good += 1
                else:
                    bad += 1
        return good / (good + bad)

    def first_layer(self, X):
        dn.renew_cg()
        if self.type == "a":
            embedded = [dn.lookup(self.input_lookup, chars) for chars in X]
        elif self.type == "b":
            input_chars = [ [dn.lookup(self.input_lookup, char) for char in chars] for chars in X ]
            embedded = None
            state_char = self.lstm_c.initial_state()
            embedded = []
            for input_char in input_chars:
                embedded.append(input_char[len(input_char) - 1])
                # temp_embedd = state_char.transduce(input_char)
                # char_embedd = temp_embedd[0]
                #
                # for i in range(1, len(temp_embedd)):
                #     char_embedd = dn.concatenate([char_embedd, temp_embedd[i]])
                # embedded.append(char_embedd)

        state_back_1 = self.lstm_b_1.initial_state()
        state_forw_1 = self.lstm_f_1.initial_state()
        fw_exps = state_forw_1.transduce(embedded)
        bw_exps = state_back_1.transduce(reversed(embedded))
        b_1 = [dn.concatenate([f,b]) for f,b in zip(fw_exps, reversed(bw_exps))]
        return b_1

    def second_layer(self, b_1):
        state_back_2 = self.lstm_b_2.initial_state()
        state_forw_2 = self.lstm_f_2.initial_state()
        fw_exps = state_forw_2.transduce(b_1)
        bw_exps = state_back_2.transduce(reversed(b_1))
        b_2 = [dn.concatenate([f,b]) for f,b in zip(fw_exps, reversed(bw_exps))]
        return b_2

    def get_probs(self, b_2, i):
        # print output_vec
        w = dn.parameter(self.output_w)
        b = dn.parameter(self.output_b)

        linear = w*b_2[i] + b
        # will go through softmax in the loss function
        return linear

    def train(self):
        start_time = time.time()
        sum_of_losses = 0.0
        checked = 0
        total_checked = 0
        total_loss = 0.0
        for i, sequence in enumerate(self.sequences):
            X, Y = self.prepare_sequence(sequence)
            b_1 = self.first_layer(X)
            b_2 = self.second_layer(b_1)
            for j in range(1, len(X)-1):
                checked += 1
                probs = self.get_probs(b_2, j)
                loss = dn.pickneglogsoftmax(probs, Y[j])
                loss_value = loss.value()
                sum_of_losses += loss_value
                loss.backward()
                self.trainer.update()
            print "loss: " + str(sum_of_losses / checked) + " for sequence number " + str(i)
            total_checked += checked
            checked = 0
            total_loss += sum_of_losses
            sum_of_losses = 0.0
            if (i+1)%500 == 0:
                print "evaluate 500 sequence in " + str(passed_time(start_time))
                start_time = time.time()
                accuracy_dev = self.validate()
                print "accuracy on dev: " + str(accuracy_dev)
        return total_loss / total_checked

    def learn(self):
        for i in range(EPOCHS):
            loss = self.train()
            print "epoch number " + str(i) + " loss: " + str(loss)

if __name__ == '__main__':
    type_word = sys.argv[1]
    folder_name = sys.argv[2]
    model_file = sys.argv[3]
    tagger_train = TaggerBiLSTM(folder_name + "/train", folder_name + "/dev", type_word)
    tagger_train.learn()
