import dynet as dn
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
import random
import model_tagger as mt

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
        self.type = type
        self.sequences, self.vocab, self.tags, self.chars = parse_file(name_file)
        if self.type == "c":
            vocab_txt = np.array(open("vocab.txt", 'r').read().split('\n'))
            for vocab in vocab_txt:
                if vocab not in self.vocab:
                    self.vocab[vocab] = len(self.vocab)
            # vecs = np.loadtxt("wordVectors.txt")
        print "number of sentences " + str(len(self.sequences))
        self.sequences_dev = parse_file(dev_file)[0]
        print "defined all of the data in " + str(passed_time(start_time))
        self.vocab_size = len(self.vocab)
        self.model = dn.Model()
        # self.input_lookup = self.model.add_lookup_parameters((self.vocab_size, EMBEDDINGS_SIZE))
        # if self.type == "c":
        #     self.input_lookup.init_from_array(vecs)
        # if self.type == "b":
        #     self.lstm_c = dn.LSTMBuilder(1, EMBEDDINGS_SIZE, HIDDEN_DIM, self.model)
        self.lstm_f_1 = dn.LSTMBuilder(1, INPUT_DIM, HIDDEN_DIM, self.model)
        self.lstm_f_2 = dn.LSTMBuilder(1, 100, HIDDEN_DIM, self.model)
        self.lstm_b_1 = dn.LSTMBuilder(1, INPUT_DIM, HIDDEN_DIM, self.model)
        self.lstm_b_2 = dn.LSTMBuilder(1, 100, HIDDEN_DIM, self.model)
        self.output_w = self.model.add_parameters((self.vocab_size, LSTM_NUM_OF_LAYERS*EMBEDDINGS_SIZE))
        self.output_b = self.model.add_parameters((self.vocab_size))

        if self.type == "a":
            self.tagger = mt.WordEmbedding(self.model, EMBEDDINGS_SIZE, self.vocab_size)
        elif self.type == "b":
            self.tagger = mt.CharEmbedding(self.model, EMBEDDINGS_SIZE, len(self.chars), HIDDEN_DIM)
        elif self.type == "c":
            self.tagger = mt.PreTrained(self.model, EMBEDDINGS_SIZE, self.vocab_size, "wordVectors.txt")
        elif self.type == "d":
            self.tagger = mt.WordCharEmbedding(self.model, EMBEDDINGS_SIZE, self.vocab_size, len(self.chars), HIDDEN_DIM)
        self.trainer = dn.AdamTrainer(self.model)
        self.define_data()
        self.define_dev_data()

    def word_or_unk(self, word):
        if word not in self.vocab:
            return UNK
        return word

    def define_data(self):
        self.x = [self.prepare_x(sentence) for sentence in self.sequences]
        self.y = [self.prepare_y(sentence) for sentence in self.sequences]
        print len(self.x)

    def define_dev_data(self):
        self.x_dev = [self.prepare_x(sentence) for sentence in self.sequences_dev]
        self.y_dev = [self.prepare_y(sentence) for sentence in self.sequences_dev]
        print len(self.x_dev)


    def prepare_x(self, sequence):
        if self.type == "a" or self.type == "c":
            # print sequence
            x = [ self.vocab[self.word_or_unk(word)] for (word, tag) in sequence ]
        elif self.type == "b" or self.type == "d":
            # when d, needs to have the words and the char
            x = []
            for (word, _) in sequence:
                word = self.word_or_unk(word)
                if word not in [UNK, START, END]:
                    x.append([self.chars[char] for char in word ])
                else:
                    x.append([self.chars[word]])
        return x

    def prepare_y(self, sequence):
        return [ self.tags[tag] for (word, tag) in sequence ]

    def prepare_sequence(self, sequence):
        x = self.prepare_x(sequence)
        y = self.prepare_y(sequence)
        return x, y

    def validate(self):
        good = 0.0
        bad = 0.0
        for X, Y in zip(self.x_dev, self.y_dev):
            # b_1 = self.first_layer(X)
            # b_2 = self.second_layer(b_1)
            probs = self.get_probs(X)
            for i in range(len(probs)):
                pred = np.argmax(probs[:, i])
                # probs = self.get_probs(b_2, j).npvalue()
                # pred = np.argmax(probs)
                label = Y[i]
                if pred == label:
                    good += 1
                else:
                    bad += 1
        return good / (good + bad)

    def first_layer(self, X):
        dn.renew_cg()
        embedded = self.tagger(X)
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
        bw_exps.reverse()
        # b_2 = [dn.concatenate([f,b]) for f,b in zip(fw_exps, reversed(bw_exps))]
        return fw_exps, bw_exps

    def get_probs(self, X):
        b_1 = self.first_layer(X)
        out_f, out_b = self.second_layer(b_1)
        size_vector = len(b_1)
        probs = []
        w = dn.parameter(self.output_w)
        b = dn.parameter(self.output_b)
        for i in range(size_vector):
            bi_output = dn.concatenate([out_f[i], out_b[i]])
            probs.append(dn.softmax(w*bi_output+b))
        # print output_vec

        # print b_2[-1]
        # linear = [(w*b_2[i] + b) for i in range(len(b_2))]
        # will go through softmax in the loss function
        # return self.tagger(b_2, i)
        return probs

    def train(self):
        start_time = time.time()
        sum_of_losses = 0.0
        checked = 0
        total_checked = 0
        total_loss = 0.0
        # i = 0?
        for  i, (X, Y) in enumerate(zip(self.x, self.y)):
            # # X, Y = self.prepare_sequence(sequence)
            # b_1 = self.first_layer(X)
            # b_2 = self.second_layer(b_1)
            probs = self.get_probs(X)
            # print Y
            losses = []
            for j, prob in enumerate(probs):
                # print prob
                # print Y[j]
                losses.append(-dn.log(dn.pick(prob, Y[j])))
            loss = dn.esum(losses)
            loss_value = loss.value()
            sum_of_losses += loss_value
            loss.backward()
            self.trainer.update()
            checked += len(X)
            # for j in range(1, len(X)-1):
            #     checked += 1
            #     probs = self.get_probs(b_2, j)
            #     loss = dn.pickneglogsoftmax(probs, Y[j])
            #     loss_value = loss.value()
            #     sum_of_losses += loss_value
            #     loss.backward()
            #     self.trainer.update()
            # print "loss: " + str(sum_of_losses / checked) + " for sequence number " + str(i)
            # total_checked += checked
            # checked = 0
            # total_loss += sum_of_losses
            # sum_of_losses = 0.0
            # i += 1
            if (i+1)%(500) == 0:
                print "evaluate 500 sequence in " + str(passed_time(start_time))
                start_time = time.time()
                accuracy_dev = self.validate()
                print "accuracy on dev: " + str(accuracy_dev)
        return sum_of_losses / checked

    def learn(self):
        for i in range(EPOCHS):
            loss = self.train()
            print "epoch number " + str(i) + " loss: " + str(loss)
        dn.save("model_type"+self.type,[self.tagger])

if __name__ == '__main__':
    type_word = sys.argv[1]
    folder_name = sys.argv[2]
    model_file = sys.argv[3]
    tagger_train = TaggerBiLSTM(folder_name + "/train", folder_name + "/dev", type_word)
    tagger_train.learn()
