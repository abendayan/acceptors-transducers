import sys
import random

def add_int_sequence(size_seq):
    string = ""
    for j in range(1, size_seq + 1):
        string += str(random.randint(1, 9))
    return string

def add_letter_sequence(size_seq, letter):
    string = ""
    for j in range(1, size_seq + 1):
        string += letter
    return string

test = False
if sys.argv[1] == "n":
    to_add = ["c", "b"]
elif sys.argv[1] == "p":
    to_add = ["b", "c"]
elif sys.argv[1] == "t":
    test = True
    to_add = ["b", "c"]

number_sentences = int(sys.argv[2])

max_length_seq = 100

for i in range(number_sentences):
    sequence = ""
    rand1 = random.randint(0, 20)
    rand2 = rand1
    while True:
        sequence += str(random.randint(1, 9))
        rand1 = random.randint(0, 20)
        if rand1 == rand2:
            break

    rand1 = random.randint(0, 20)
    rand2 = rand1
    while True:
        sequence += "a"
        rand1 = random.randint(0, 20)
        if rand1 == rand2:
            break

    rand1 = random.randint(0, 20)
    rand2 = rand1
    while True:
        sequence += str(random.randint(1, 9))
        rand1 = random.randint(0, 20)
        if rand1 == rand2:
            break

    index = 0
    if test:
        index = random.randint(0, 1)

    rand1 = random.randint(0, 20)
    rand2 = rand1
    while True:
        sequence += to_add[index]
        rand1 = random.randint(0, 20)
        if rand1 == rand2:
            break

    rand1 = random.randint(0, 20)
    rand2 = rand1
    while True:
        sequence += str(random.randint(1, 9))
        rand1 = random.randint(0, 20)
        if rand1 == rand2:
            break
    index += 1
    index = index % 2
    rand1 = random.randint(0, 20)
    rand2 = rand1
    while True:
        sequence += to_add[index]
        rand1 = random.randint(0, 20)
        if rand1 == rand2:
            break

    rand1 = random.randint(0, 20)
    rand2 = rand1
    while True:
        sequence += str(random.randint(1, 9))
        rand1 = random.randint(0, 20)
        if rand1 == rand2:
            break

    rand1 = random.randint(0, 20)
    rand2 = rand1
    while True:
        sequence += "a"
        rand1 = random.randint(0, 20)
        if rand1 == rand2:
            break

    rand1 = random.randint(0, 20)
    rand2 = rand1
    while True:
        sequence += str(random.randint(1, 9))
        rand1 = random.randint(0, 20)
        if rand1 == rand2:
            break
    if test:
        tag = "/" + str(index)
        sequence += tag

    print sequence
