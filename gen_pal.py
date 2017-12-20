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
if sys.argv[1] == "t":
    test = True

number_sentences = int(sys.argv[2])

max_length_seq = 100

map_int_letter = {
    0: '1',
    1: '2',
    2: '3',
    3: '4',
    4: '5',
    5: '6',
    6: '7',
    7: '8',
    8: '9',
    9: 'a',
    10: 'b',
    11: 'c',
    12: 'd'
}

for i in range(number_sentences):
    sequence = ""
    rand1 = 1
    while rand1 != 0:
        sequence += str(map_int_letter[random.randint(0, 12)])
        rand1 = random.randint(0, 20)
        rand1 = random.randint(0, 20)

    index = random.randint(0, 1)
    if index == 1:
        sequence += sequence[::-1]
    else:
        rand1 = 1
        while rand1 != 0:
            sequence += str(map_int_letter[random.randint(0, 12)])
            rand1 = random.randint(0, 20)
            rand1 = random.randint(0, 20)
    if test:
        tag = "/" + str(index)
        sequence += tag

    print sequence
