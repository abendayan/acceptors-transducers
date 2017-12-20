import dynet as dn
import numpy as np
import time
import sys
import matplotlib.pyplot as plt

EMBEDDINGS_SIZE = 10
LSTM_NUM_OF_LAYERS = 1
STATE_SIZE = 10
NUM_OF_CLASSES = 2
EPOCHS = 200
BATCH_SIZE = 50
HIDDEN_LAYER = 1
NUM_OUT = 100
VOCAB_SIZE = 13
start_time = time.time()

model = dn.Model()
input_lookup = model.add_lookup_parameters((VOCAB_SIZE, EMBEDDINGS_SIZE))
lstm = dn.LSTMBuilder(LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, model)
output_w_1 = model.add_parameters((VOCAB_SIZE, STATE_SIZE))
output_b_1 = model.add_parameters((VOCAB_SIZE))

output_w_2 = model.add_parameters((NUM_OUT, VOCAB_SIZE))
output_b_2 = model.add_parameters((NUM_OUT))
trainer = dn.AdamTrainer(model)

def passed_time(previous_time):
    return round(time.time() - previous_time, 3)

def read_to_data(name_file):
    words_tags = []
    max_length = 0
    words = open(name_file, "r").read().split('\n')
    for word_tag in words:
        if word_tag != "":
            word, tag = word_tag.split('/')
            words_tags.append((word, tag))
            if len(word) > max_length:
                max_length = len(word)
    return words_tags

def prepare_sequence(seq, to_ix):
    return [to_ix[str(w)] for w in seq]

def save_iter_in_graph(data, folder):
    graphes = {
        "accuracy_train": 0,
        "loss_train": 1,
        "accuracy_test": 2,
    }

    for graph, i in graphes.items():
        plt.figure(i)
        plt.plot(range(len(data)), [a[i] for a in data])
        plt.xlabel('Epochs')
        plt.ylabel(graph)
        plt.savefig(folder + graph + '.png')

class Tagger:
    def __init__(self, file_read, word_to_ix):
        print "start creating " + file_read + " in: "  + str(passed_time(start_time))
        self.training_data = read_to_data(file_read)
        self.word_to_ix = {}
        self.tag_to_ix = { "0":0, "1":1 }
        if word_to_ix is None:
            self.define_words()
        else:
            self.word_to_ix = word_to_ix
        self.define_data()
        self.batch_x, self.batch_y = self.to_batch(self.x, self.y, BATCH_SIZE)
        self.batched_X_padded = list(map(self.pad_batch, self.batch_x))
        print "define data for " + file_read + " in: "  + str(passed_time(start_time))

    def define_data(self):
        self.x = [prepare_sequence(sentence, self.word_to_ix) for sentence, tag in self.training_data]
        self.y = [self.tag_to_ix[str(tag)] for sentence, tag in self.training_data]

    def to_batch(self, X, Y, batch_size):
        #sort dataset by length
        data = list(zip(*sorted(zip(X,Y), key=lambda x: len(x[0]))))
        batched_X = []
        batched_Y = []
        for i in range(int(np.ceil(len(X)/batch_size))):
            batched_X.append(data[0][i*batch_size:(i+1)*batch_size])
            batched_Y.append(data[1][i*batch_size:(i+1)*batch_size])
        return batched_X, batched_Y

    def define_words(self):
        for sent, tags in self.training_data:
            for word in sent:
                if word not in self.word_to_ix:
                    self.word_to_ix[word] = len(self.word_to_ix)

    def print_dataset(self, X, Y):
        for seq, label in zip(X, Y):
            print('label:', label, 'seq:', seq)

    def train(self, trainX, trainY):
        start_time = time.time()
        sum_of_losses = 0.0
        for X, Y in zip(trainX, trainY):
            probs = self.get_probs(X)
            # y_expr = dn.vecInput(len(Y))
            # y_expr.set(Y)
            soft_max = dn.pickneglogsoftmax_batch(probs, Y)
            loss = dn.sum_batches(soft_max)
            loss_value = loss.value()
            sum_of_losses += loss_value
            loss.backward()
            trainer.update()
        return sum_of_losses / len(trainX)

    def validate(self, testX, testY, tagger):
        acc = []
        good = 0.0
        bad = 0.0
        for X, Y in zip(testX, testY):
            probs = tagger.get_probs(X).npvalue()
            for i in range(len(probs[0])):
                pred = np.argmax(probs[:, i])

                label = Y[i]
                if pred == label:

                    good += 1
                    acc.append(1)
                else:
                    bad += 1
                    acc.append(0)
        return np.mean(acc)

    def pad_batch(self, batch):
        max_len = len(batch[-1])
        padded_batch = []
        for x in batch:
            x = [VOCAB_SIZE-1]*(max_len-len(x)) + x
            padded_batch.append(x)
        return padded_batch

    def get_probs(self, batch):
        dn.renew_cg()

        # The I iteration embed all the i-th items in all batches
        embedded = [dn.lookup_batch(input_lookup, chars) for chars in zip(*batch)]
        state = lstm.initial_state()

        # embedded = embedded.expr()
        output_vec = state.transduce(embedded)[-1]

        w_1 = dn.parameter(output_w_1)
        w_2 = dn.parameter(output_w_2)
        b_1 = dn.parameter(output_b_1)
        b_2 = dn.parameter(output_b_2)
        linear_1 = w_1*output_vec + b_1
        non_linear = dn.tanh(linear_1)
        linear_2 = w_2*non_linear + b_2
        # will go through softmax in the loss function
        return linear_2

    def learn(self, train, tagger):
        # self.validate(self.batched_X_padded, self.batch_y)
        if train:
            loss = self.train(self.batched_X_padded, self.batch_y)
        acc = self.validate(self.batched_X_padded, self.batch_y, tagger)
        if train:
            return loss, acc
        return acc

if __name__ == '__main__':
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    folder = ""
    if len(sys.argv) > 3:
        folder = sys.argv[3] + "/"
    train_tagger = Tagger(train_file, None)
    test_tagger = Tagger(test_file, train_tagger.word_to_ix)

    print "Read all of the data "  + str(passed_time(start_time))
    data_graph = []
    start_test = time.time()
    for epoch in range(EPOCHS):
        print "Start epoch "  + str(epoch)
        start_time = time.time()
        loss_train, acc_train = train_tagger.learn(True, train_tagger)
        print "Finish train "  + str(passed_time(start_time)) + " loss: " + str(loss_train) + " accuracy: " + str(acc_train)
        start_time = time.time()
        acc_test = test_tagger.learn(False, train_tagger)
        print "Finish test "  + str(passed_time(start_time)) + " accuracy: " + str(acc_test)
        data_graph.append([acc_train, loss_train, acc_test])
    save_iter_in_graph(data_graph, folder)
    print "Finish all of the training and testing " + str(passed_time(start_test))
