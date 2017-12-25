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
        self.tags_to_ix = { id:tag for tag, id in self.tags.iteritems() }
        print self.tags_to_ix
        if self.type == "c":
            vocab_txt = np.array(open("vocab.txt", 'r').read().split('\n'))
            for vocab in vocab_txt:
                if vocab not in self.vocab:
                    self.vocab[vocab] = len(self.vocab)
        print "number of sentences " + str(len(self.sequences))
        self.sequences_dev = parse_file(dev_file)[0]
        print "defined all of the data in " + str(passed_time(start_time))
        self.vocab_size = len(self.vocab)
        self.model = dn.Model()
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
            probs = self.get_probs(X)
            probs_values = [prob.npvalue() for prob in probs]
            for i, prob in enumerate(probs_values):
                print self.tags_to_ix[Y[i]]
                tag = self.tags_to_ix[np.argmax(prob)]
                if tag != 'O': # for NER
                    label = self.tags_to_ix[Y[i]]
                    if pred == label:
                        good += 1
                    else:
                        bad += 1
        return good / (good + bad)

    def get_probs(self, X):
        dn.renew_cg()
        embedded = self.tagger([X])
        state_back_1 = self.lstm_b_1.initial_state()
        state_forw_1 = self.lstm_f_1.initial_state()
        fw_exps = state_forw_1.transduce(embedded)
        bw_exps = state_back_1.transduce(reversed(embedded))
        bw_exps.reverse()
        b_1 = [dn.concatenate([f,b]) for f,b in zip(fw_exps, bw_exps)]
        state_back_2 = self.lstm_b_2.initial_state()
        state_forw_2 = self.lstm_f_2.initial_state()
        out_f = state_forw_2.transduce(b_1)
        out_b = state_back_2.transduce(reversed(b_1))
        out_b.reverse()
        size_vector = len(b_1)
        probs = []
        w = dn.parameter(self.output_w)
        b = dn.parameter(self.output_b)
        for i in range(size_vector):
            bi_output = dn.concatenate([out_f[i], out_b[i]])
            probs.append(dn.softmax(w*bi_output+b))
        return probs

    def train(self):
        start_time = time.time()
        sum_of_losses = 0.0
        checked = 0
        total_checked = 0
        total_loss = 0.0
        # i = 0?
        for  i, (X, Y) in enumerate(zip(self.x, self.y)):
            probs = self.get_probs(X)
            losses = []
            for j, prob in enumerate(probs):
                losses.append(dn.sum_batches(-dn.log(dn.pick_batch(prob, Y))))
            # loss = -dn.log(dn.pick_batch(probs, Y))
            # loss = dn.sum_batches(losses)
            loss = dn.esum(losses)
            loss_value = loss.scalar_value()
            # print loss_value
            sum_of_losses += loss_value
            loss.forward()
            loss.backward()
            self.trainer.update()
            checked += len(X)
            # print "sequence " + str(i) + " with loss " + str(loss_value/len(X))
            if (i+1)%(500) == 0:
                print "evaluate 500 sequence in " + str(passed_time(start_time))
                start_time = time.time()
                accuracy_dev = self.validate()
                print "accuracy on dev: " + str(accuracy_dev)
        return sum_of_losses

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
