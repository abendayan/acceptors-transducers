import dynet as dn
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
import random
import model_tagger as mt

LSTM_NUM_OF_LAYERS = 2
INPUT_DIM = 50
HIDDEN_DIM = 30
EPOCHS = 5
UNK = "UUUNKKK"
START = "<s>"
END = "</s>"
EMBEDDINGS_SIZE = 50
start_time = time.time()

def passed_time(previous_time):
    return round(time.time() - previous_time, 3)

def parse_file(name_file):
    words = open(name_file, "r").read().split("\n")
    sequences = []
    sequence = []
    # sequence = [(START, START)]
    vocab = { UNK: 0 }
    tags = { }
    chars = { UNK: 0 }
    # chars = { UNK: 0, START: 1, END: 2 }
    for word_tag in words:
        if word_tag == "":
            # sequence.append((END, END))
            sequences.append(sequence)
            sequence = []
            # sequence = [(START, START)]
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
    tags[END] = len(tags)
    return sequences, vocab, tags, chars

class TaggerBiLSTM:
    def __init__(self, name_file, type="a"):
        # dev_file = folder + "/dev"
        self.type = type
        self.sequences, self.vocab, self.tags, self.chars = parse_file(name_file)
        self.tags_to_ix = { id:tag for tag, id in self.tags.iteritems() }
        self.out_size = len(self.tags)
        if self.type == "c":
            vocab_txt = np.array(open("vocab.txt", 'r').read().split('\n'))
            for vocab in vocab_txt:
                if vocab not in self.vocab:
                    self.vocab[vocab] = len(self.vocab)
        print "number of sentences " + str(len(self.sequences))
        # sequence_dev = parse_file(dev_file)[0]
        # random.shuffle(sequence_dev)
        dev_size = len(self.sequences)/100
        print dev_size
        self.sequences_dev = self.sequences[:dev_size]
        print "defined all of the data in " + str(passed_time(start_time))
        self.vocab_size = len(self.vocab)
        model = dn.Model()
        self.model = mt.TaggerModel(model, EMBEDDINGS_SIZE, HIDDEN_DIM, self.out_size, self.vocab_size, len(self.chars), self.type)
        # self.lstm_f_1 = dn.LSTMBuilder(1, EMBEDDINGS_SIZE, HIDDEN_DIM, self.model)
        # self.lstm_f_2 = dn.LSTMBuilder(1, 2*HIDDEN_DIM, HIDDEN_DIM, self.model)
        # self.lstm_b_1 = dn.LSTMBuilder(1, EMBEDDINGS_SIZE, HIDDEN_DIM, self.model)
        # self.lstm_b_2 = dn.LSTMBuilder(1, 2*HIDDEN_DIM, HIDDEN_DIM, self.model)
        # self.output_w = self.model.add_parameters((self.out_size, 2*HIDDEN_DIM))
        # # self.output_b = self.model.add_parameters((self.out_size))
        #
        # if self.type == "a":
        #     self.tagger = mt.WordEmbedding(self.model, EMBEDDINGS_SIZE, self.vocab_size)
        # elif self.type == "b":
        #     self.tagger = mt.CharEmbedding(self.model, EMBEDDINGS_SIZE, len(self.chars))
        # elif self.type == "c":
        #     self.tagger = mt.PreTrained(self.model, EMBEDDINGS_SIZE, self.vocab_size, "wordVectors.txt")
        # elif self.type == "d":
        #     self.tagger = mt.WordCharEmbedding(self.model, EMBEDDINGS_SIZE, self.vocab_size, len(self.chars), HIDDEN_DIM)
        # self.trainer = dn.AdamTrainer(self.model)
        self.define_data()
        self.define_dev_data()

    def word_or_unk(self, word):
        if word not in self.vocab:
            return UNK
        return word

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
            # when d, needs to have the words and the char
            x = []
            for (word, _) in sequence:
                word = self.word_or_unk(word)
                if word not in [UNK, START, END]:
                    x.append([self.chars[char] for char in word ])
                else:
                    x.append([self.chars[word]])
        elif self.type == "d":
            x = []
            for (word, _) in sequence:
                word = self.word_or_unk(word)
                if word not in [UNK, START, END]:
                    char = [self.chars[char] for char in word ]
                else:
                    char = [self.chars[word]]
                x.append((self.vocab[word], char))
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
                if pred == label:
                    good += 1
                else:
                    bad += 1
        return good / (good + bad)

    def get_probs(self, X):
        dn.renew_cg(True, True)
        return self.model(X)
        # embedded = [ self.tagger(word) for word in X ]
        # # print embedded
        # state_back_1 = self.lstm_b_1.initial_state()
        # state_forw_1 = self.lstm_f_1.initial_state()
        # fw_exps = state_forw_1.transduce(embedded)
        # bw_exps = state_back_1.transduce(reversed(embedded))
        # bw_exps.reverse()
        # b_1 = [dn.concatenate([f,b]) for f,b in zip(fw_exps, bw_exps)]
        # state_back_2 = self.lstm_b_2.initial_state()
        # state_forw_2 = self.lstm_f_2.initial_state()
        # out_f = state_forw_2.transduce(b_1)
        # out_b = state_back_2.transduce(reversed(b_1))
        # out_b.reverse()
        # size_vector = len(embedded)
        # w = dn.parameter(self.output_w)
        # # b = dn.parameter(self.output_b)
        # b_2 = [ dn.concatenate([out_f[i], out_b[i]]) for i in range(size_vector) ]
        # probs = [ w*b_2_i for b_2_i in b_2 ]
        # probs = [ w*b_2_i+b for b_2_i in b_2 ]
        # return probs

    def train(self):
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
                # loss.forward()
                checked += len(Y)
                total_checked += len(Y)
                self.model.trainer.update()
                if (i+1)%(500) == 0:
                    to_print_loss.append(sum_of_losses/checked)
                    to_print_time.append(passed_time(start_time))
                    # print "batch of sequences number " + str(j) + " with loss " + str(sum_of_losses/checked)
                    checked = 0.0
                    sum_of_losses = 0.0
                    # print "evaluate 500 sequence in " + str(passed_time(start_time))
                    start_time = time.time()
                    accuracy_dev = self.validate()
                    accuracy_all.append(accuracy_dev)
                    to_print_time_valid.append(passed_time(start_time))
                    # print "accuracy on dev: " + str(accuracy_dev) + " in time " + str(passed_time(start_time))
                    start_time = time.time()
                    print str(j) + "\t|| " + str(to_print_loss[j]) + " || " + str(to_print_time[j]) + " || " + str(to_print_time_valid[j]) + " || " + str(accuracy_all[j])
                    j += 1
            print "epoch loss: " + str(total_losses/total_checked) + " last accuracy " + str(accuracy_all[len(accuracy_all)-1])
            print "epoch number " + str(epoch+1) + " done in " + str(passed_time(start_epoch))
            start_epoch = time.time()
        dn.save("model_type"+self.type,[self.model])
        return accuracy_all

if __name__ == '__main__':
    type_word = sys.argv[1]
    folder_name = sys.argv[2]
    model_file = sys.argv[3]
    tagger_train = TaggerBiLSTM(folder_name, type_word)
    accuracy_type = tagger_train.train()
    accuracy_all = [accuracy_type, [0], [0], [0]]
