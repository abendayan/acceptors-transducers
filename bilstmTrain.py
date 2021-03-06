import dynet as dn
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
import random
import model_tagger as mt
import pickle

LSTM_NUM_OF_LAYERS = 2
INPUT_DIM = 50
HIDDEN_DIM = 30
EPOCHS = 5
UNK = "UUUNKKK"
EMBEDDINGS_SIZE = 50
start_time = time.time()

def passed_time(previous_time):
    return round(time.time() - previous_time, 3)

def csv_writer(data, path):
    """
    Write data to a CSV file path
    """
    with open(path, "wb") as csv_file:
        writer = csv.writer(csv_file, delimiter=';')
        for line in data:
            writer.writerow(line)

def parse_file(name_file):
    words = open(name_file, "r").read().split("\n")
    sequences = []
    sequence = []
    vocab = { UNK: 0 }
    tags = { }
    chars = { UNK: 0 }
    for word_tag in words:
        if word_tag == "":
            sequences.append(sequence)
            sequence = []
        else:
            word, tag = word_tag.split()
            if tag not in tags:
                tags[tag] = len(tags)
            if word not in vocab:
                if len(vocab) > 0.9*len(words):
                    word = UNK
                else:
                    vocab[word] = len(vocab)
            for char in word:
                if char not in chars:
                    chars[char] = len(chars)
            sequence.append((word, tag))
    return sequences, vocab, tags, chars

class TaggerBiLSTM:
    def __init__(self, name_file, type, model_name, dev_file):
        self.type = type
        self.sequences, self.vocab, self.tags, self.chars = parse_file(name_file)
        self.tags_to_ix = { id:tag for tag, id in self.tags.iteritems() }
        self.out_size = len(self.tags)
        print len(self.vocab)
        if self.type == "c":
            suffixes = {}
            preffix = {}
            i = len(self.vocab)
            suffixes = [ "@" + word[:3] for word in self.vocab ]
            preffix = [ word[-3:] + "@" for word in self.vocab ]
            for word in suffixes:
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)
            for word in preffix:
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)
        print "number of sentences " + str(len(self.sequences))
        pickle.dump([self.vocab, self.tags, self.chars], open(model_name+self.type+".vocab", "wb"))
        self.sequences_dev = parse_file(dev_file)[0]
        print len(self.sequences_dev)
        print "defined all of the data in " + str(passed_time(start_time))
        self.vocab_size = len(self.vocab)
        model = dn.Model()
        self.model = mt.TaggerModel(model, EMBEDDINGS_SIZE, HIDDEN_DIM, self.out_size, self.vocab_size, len(self.chars), self.type)
        self.define_data()
        self.define_dev_data()

    def word_or_unk(self, word):
        if word not in self.vocab:
            if self.type == "c":
                if "@" + word[:3] in self.vocab:
                    return "@" + word[:3]
                if word[-3:] + "@" in self.vocab:
                    return word[-3:] + "@"
            return UNK
        return word

    def char_or_unk(self, char):
        if char not in self.chars:
            return UNK
        return char

    def define_data(self):
        self.x = [self.prepare_x(sentence) for sentence in self.sequences]
        self.y = [self.prepare_y(sentence) for sentence in self.sequences]

    def define_dev_data(self):
        self.x_dev = [self.prepare_x(sentence) for sentence in self.sequences_dev]
        self.y_dev = [self.prepare_y(sentence) for sentence in self.sequences_dev]

    def prepare_x(self, sequence):
        if self.type == "a" or self.type == "c":
            x = [ self.vocab[self.word_or_unk(word)] for (word, tag) in sequence ]
        elif self.type == "b":
            x = [ [self.chars[self.char_or_unk(char)] for char in word] for (word, _) in sequence ]
        elif self.type == "d":
            x = []
            for (word, _) in sequence:
                x.append((self.vocab[self.word_or_unk(word)], [self.chars[self.char_or_unk(char)] for char in word]))
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
            for prob, y in zip(probs, Y):
                softmax = dn.softmax(prob).npvalue()
                pred = np.argmax(softmax)
                label = y
                if self.tags_to_ix[pred] == 'O' and self.tags_to_ix[label] == 'O':
                    continue
                if pred == label:
                    good += 1
                else:
                    bad += 1
        return good / (good + bad)

    def get_probs(self, X):
        dn.renew_cg(True, True)
        return self.model(X)

    def train(self):
        to_write = ""
        for epoch in range(EPOCHS):
            start_epoch = time.time()
            print "===================== EPOCH " + str(epoch+1) + " ====================="
            start_time = time.time()
            sum_of_losses = 0.0
            total_losses = 0.0
            checked = 0.0
            total_checked = 0.0
            accuracy_all = []
            j = 0
            to_print_loss = []
            to_print_time = []
            to_print_time_valid = []
            print "========================================================="
            print "N*batch|| Loss      || Time train || Time dev || accuracy"
            for  i, (X, Y) in enumerate(zip(self.x, self.y)):
                probs = self.get_probs(X)
                losses = [ dn.pickneglogsoftmax(prob, y) for prob, y in zip(probs, Y)]
                if losses == []:
                    continue
                loss = dn.esum(losses)
                loss_value = loss.value()
                sum_of_losses += loss_value
                total_losses += loss_value
                loss.backward()
                checked += len(Y)
                total_checked += len(Y)
                self.model.trainer.update()
                if (i+1)%(500) == 0:
                    to_print_loss.append(sum_of_losses/checked)
                    to_print_time.append(passed_time(start_time))
                    checked = 0.0
                    sum_of_losses = 0.0
                    start_time = time.time()
                    accuracy_dev = self.validate()
                    accuracy_all.append(accuracy_dev)
                    to_print_time_valid.append(passed_time(start_time))
                    start_time = time.time()
                    to_write += str(j)+";"+str(to_print_loss[j])+";"+str(to_print_time[j])+";"+str(to_print_time_valid[j])+";"+str(accuracy_all[j])+"\n"
                    print str(j) + "\t|| " + str(to_print_loss[j]) + " || " + str(to_print_time[j]) + " || " + str(to_print_time_valid[j]) + " || " + str(accuracy_all[j])
                    j += 1
            print "epoch loss: " + str(total_losses/total_checked) + " last accuracy " + str(accuracy_all[len(accuracy_all)-1])
            print "epoch number " + str(epoch+1) + " done in " + str(passed_time(start_epoch))
            start_epoch = time.time()
        dn.save("model_type"+self.type,[self.model])
        write_file = open("output"+self.type+".txt", "w")
        write_file.write(to_write)
        write_file.close()

if __name__ == '__main__':
    type_word = sys.argv[1]
    folder_name = sys.argv[2]
    model_file = sys.argv[3]
    dev_file = sys.argv[4]
    tagger_train = TaggerBiLSTM(folder_name, type_word, model_file, dev_file)
    tagger_train.train()
