'''Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM, Dropout
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import re

#path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
path = './mindre1m.txt'
text = open(path).read().lower()
print('corpus length:', len(text))

#chars = sorted(list(set(text)))
regexp = re.compile("[a-zåäö]+")
text_sliced = regexp.findall(text.lower())
words = sorted(list(set(text_sliced)))

print('total words:', len(words))
word_indices = dict((c, i) for i, c in enumerate(words))
indices_word = dict((i, c) for i, c in enumerate(words))

# cut the text in semi-redundant sequences of maxlen characters
bigrams = []
next_words = []
for i in range(0, len(text_sliced) - 2):
    bigrams.append(text_sliced[i: i + 1])
    next_words.append(text_sliced[i + 2])
# print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(bigrams), 2, len(words)), dtype=np.bool)
y = np.zeros((len(bigrams), len(words)), dtype=np.bool)
for i, bigram in enumerate(bigrams):
    for t, word in enumerate(bigram):
        X[i, t, word_indices[word]] = 1
    y[i, word_indices[next_words[i]]] = 1

# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, input_shape=(2, len(words))))
model.add(Dense(len(words)))
model.add(Activation('softmax'))
optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# train the model, output generated text after each iteration
tbCallBack = keras.callbacks.TensorBoard(log_dir='./graph', histogram_freq=0, write_graph=True, write_images=False)

for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('Iteration', iteration)

    model.fit(X, y,
              batch_size=128,
              epochs=1,
              callbacks=[tbCallBack])

    start_index = random.randint(0, len(text_sliced) - 2)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = []
        sentence = text_sliced[start_index: start_index + 2]
        print("start_sentence", sentence)
        generated.extend(sentence)
        print('------- Generating with seed: "' + " ".join(sentence) + '"')
        print()

        for i in range(40):
            x = np.zeros((1, 2, len(words)))
            for t, word in enumerate(sentence):
                x[0, t, word_indices[word]] = 1. # vectorization

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_word = indices_word[next_index]

            generated.append(next_word)
            sentence = sentence[1:]
            sentence.append(next_word)

        # print(generated)
        print(" ".join(generated))
        print()
