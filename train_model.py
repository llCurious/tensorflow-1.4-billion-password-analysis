# -*- coding: utf-8 -*-
import argparse
import multiprocessing
from collections import Counter

import numpy as np
import os
from keras import layers
from keras.layers import Dropout
from keras.models import Sequential
from keras_transformer import get_model
from keras_transformer import decode
from keras.models import load_model

from data_gen import get_chars_and_ctable, colors, get_words_and_wtable
from train_constants import *
import itertools

def add_chars_for_transformer():
    chars, c_table = get_chars_and_ctable()
    # print('chars: [ ', chars, ' ]')
    chars = chars + start_token + end_token
    print('chars: [ ', chars, ' ]')
    print(c_table.indices_char)
    c_table.add_token(start_token)
    c_table.add_token(end_token)
    c_table.add_token(pad_char)
    return chars, c_table


def add_words_for_transformer():
    words, w_table = get_words_and_wtable()
    print('words: [ ', words, ' ]')
    words.append(start_token)
    words.append(end_token)
    words.append(pad_char)
    print('words: [ ', words, ' ]')
    w_table.add_token(start_token)
    w_table.add_token(end_token)
    w_table.add_token(pad_char)
    return words, w_table


def get_arguments(parser):
    args = None
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        exit(1)
    return args


def get_script_arguments():
    parser = argparse.ArgumentParser(description='Training a password model.')
    # Something like: /home/premy/BreachCompilationAnalysis/edit-distances/1.csv
    # Result of run_data_processing.py.
    # parser.add_argument('--training_filename', required=True, type=str)
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    args = get_arguments(parser)
    print(args)
    return args


def gen_large_chunk_single_thread(inputs_, targets_, chunk_size, iteration):
    random_indices = np.random.choice(a=range(len(inputs_)), size=chunk_size, replace=True)
    sub_inputs = inputs_[random_indices]
    sub_targets = targets_[random_indices]

    # sub_inputs = inputs_[chunk_size*(iteration-1):chunk_size*iteration]
    # sub_targets = targets_[chunk_size*(iteration-1):chunk_size*iteration]
    print(sub_inputs[:5])
    print(sub_targets[:5])
    x = np.zeros((chunk_size, ENCODING_MAX_PASSWORD_LENGTH))
    y = np.zeros((chunk_size, ENCODING_MAX_PASSWORD_LENGTH))
    output = np.zeros((chunk_size, ENCODING_MAX_PASSWORD_LENGTH, 1))

    for i_, element in enumerate(sub_inputs):
        # todo: add segment and word embeddings
        x[i_] = c_table.encode(element, ENCODING_MAX_PASSWORD_LENGTH)
    for i_, element in enumerate(sub_targets):
        y[i_] = c_table.encode(element, ENCODING_MAX_PASSWORD_LENGTH)
        # print(sub_targets[i_], " : ", y[i_])
        res = [[x] for x in c_table.encode(element, ENCODING_MAX_PASSWORD_LENGTH)[1:]]
        res.append([c_table.encode_char(pad_char)])
        # print(res)
        output[i_] = res

    split_at = len(x) - len(x) // 10
    (x_train, x_val) = x[:split_at], x[split_at:]
    (y_train, y_val) = y[:split_at], y[split_at:]
    (output_y_train, output_y_val) = output[:split_at], output[split_at:]

    print(np.array(x_train).shape)
    return x_train, y_train, x_val, y_val, output_y_train, output_y_val
    # return [np.array(list(x)) for x in x_train], [np.array(list(x)) for x in y_train], [np.array(list(x)) for x in x_val], [np.array(list(x)) for x in y_val]


def predict_top_most_likely_passwords_monte_carlo(model_, rowx_, n_, mc_samples=10000):
    samples = predict_top_most_likely_passwords(model_, rowx_, mc_samples)
    return dict(Counter(samples).most_common(n_)).keys()


def predict_top_most_likely_passwords(model_, rowx_, n_):
    p_ = model_.predict(rowx_, batch_size=32, verbose=0)[0]
    most_likely_passwords = []
    for ii in range(n_):
        # of course should take the edit distance constraint.
        pa = np.array([np.random.choice(a=range(ENCODING_MAX_SIZE_VOCAB + 2), size=1, p=p_[jj, :])
                       for jj in range(ENCODING_MAX_PASSWORD_LENGTH)]).flatten()
        most_likely_passwords.append(c_table.decode(pa, calc_argmax=False))
    return most_likely_passwords
    # Could sample 1000 and take the most_common()


def predict_top_most_likely_passwords_transformer_monte_carlo(model_, rowx_, rowy, n_):
    p_ = model_.predict([rowx_, rowy], batch_size=32, verbose=0)[0]
    most_likely_passwords = []
    for ii in range(n_):
        # of course should take the edit distance constraint.
        # print('a: ', len(range(ENCODING_MAX_SIZE_VOCAB + 4)), ', p: ', len(p_[1, :]))
        pa = np.array([np.random.choice(a=range(ENCODING_MAX_SIZE_VOCAB + 5), size=1, p=p_[jj, :]) for jj in range(ENCODING_MAX_PASSWORD_LENGTH)]).flatten()
        most_likely_passwords.append(c_table.decode(pa, calc_argmax=False))
    return dict(Counter(most_likely_passwords).most_common(n_)).keys()


def gen_large_chunk_multi_thread(inputs_, targets_, chunk_size):
    ''' This function is actually slower than gen_large_chunk_single_thread()'''

    def parallel_function(f, sequence, num_threads=None):
        from multiprocessing.pool import ThreadPool
        pool = ThreadPool(processes=num_threads)
        result = pool.map(f, sequence)
        cleaned = np.array([x for x in result if x is not None])
        pool.close()
        pool.join()
        return cleaned

    random_indices = np.random.choice(a=range(len(inputs_)), size=chunk_size, replace=True)
    sub_inputs = inputs_[random_indices]
    sub_targets = targets_[random_indices]

    def encode(elt):
        return c_table.encode(elt, ENCODING_MAX_PASSWORD_LENGTH)

    num_threads = multiprocessing.cpu_count() // 2
    x = parallel_function(encode, sub_inputs, num_threads=num_threads)
    y = parallel_function(encode, sub_targets, num_threads=num_threads)

    split_at = len(x) - len(x) // 10
    (x_train, x_val) = x[:split_at], x[split_at:]
    (y_train, y_val) = y[:split_at], y[split_at:]

    return x_train, y_train, x_val, y_val


# init
INPUT_MAX_LEN = ENCODING_MAX_PASSWORD_LENGTH
OUTPUT_MAX_LEN = ENCODING_MAX_PASSWORD_LENGTH

# load vocabulary
try:
    chars, c_table = add_chars_for_transformer()
    # add for transformer
    # chars, c_table = add_words_for_transformer()
    print('char indices: [', c_table.indices_char, ']')
except FileNotFoundError:
    print('Run first run_encoding.py to generate the required files.')
    exit(1)

# load data
if not os.path.exists('/tmp/x_y.npz'):
    raise Exception('Please run the vectorization script before.')

print('Loading data from prefetch...')
data = np.load('/tmp/x_y.npz')
inputs = data['inputs']
targets = data['targets']
print(inputs[0])
print('Data:')
print(inputs.shape)
print(targets.shape)

ARGS = get_script_arguments()

# Try replacing GRU.
RNN = layers.LSTM
HIDDEN_SIZE = ARGS.hidden_size
BATCH_SIZE = ARGS.batch_size

print('Build model...')


def model_1():
    num_layers = 1
    m = Sequential()
    m.add(RNN(HIDDEN_SIZE, input_shape=(INPUT_MAX_LEN, len(chars))))
    m.add(layers.RepeatVector(OUTPUT_MAX_LEN))
    for _ in range(num_layers):
        m.add(RNN(HIDDEN_SIZE, return_sequences=True))
    m.add(layers.TimeDistributed(layers.Dense(len(chars))))
    m.add(layers.Activation('softmax'))
    return m


def model_2():
    # too big in Memory!
    m = Sequential()
    from keras.layers.core import Flatten, Dense, Reshape
    from keras.layers.wrappers import TimeDistributed
    m.add(Flatten(input_shape=(INPUT_MAX_LEN, len(chars))))
    m.add(Dense(OUTPUT_MAX_LEN * len(chars)))
    m.add(Reshape((OUTPUT_MAX_LEN, len(chars))))
    m.add(TimeDistributed(Dense(len(chars), activation='softmax')))
    return m


def model_3():
    m = Sequential()
    from keras.layers.core import Dense, Reshape
    from keras.layers.wrappers import TimeDistributed
    m.add(RNN(HIDDEN_SIZE, input_shape=(INPUT_MAX_LEN, len(chars))))
    m.add(Dense(OUTPUT_MAX_LEN * len(chars), activation='relu'))
    m.add(Dropout(0.5))
    m.add(Dense(OUTPUT_MAX_LEN * len(chars), activation='relu'))
    m.add(Dropout(0.5))
    m.add(Reshape((OUTPUT_MAX_LEN, len(chars))))
    m.add(TimeDistributed(Dense(len(chars), activation='softmax')))
    return m


def model_transformer():
    # chars = chars +  '<START>'
    m = get_model(
        token_num=len(c_table.char_indices),
        embed_dim=EMBEDDING_DIM,  # word/character embedding dim
        encoder_num=3,
        decoder_num=2,
        head_num=2,
        hidden_dim=120,
        attention_activation='relu',
        feed_forward_activation='relu',
        dropout_rate=0.05,
        embed_weights=np.random.random((len(c_table.char_indices), EMBEDDING_DIM)),
    )
    return m

PASSWORD_END = '\n'


def read_train_data():
    train_data = []
    PATH = "data/train.txt"
    with open(PATH, 'r') as f:
        for line in f:
            train_data.append(line.strip('\n'))
    return train_data


def all_prefixes(pwd):
    # "password"
    # ["p","pa","pas"...]
    # [" ","p","pa","pas"... "passwor"]+["password"]
    return [pwd[:i] for i in range(len(pwd))] + [pwd]


def all_suffixes(pwd):
    # ["p","a","s","s"..."\n"]
    return [pwd[i] for i in range(len(pwd))] + [PASSWORD_END]


def generate_source_target(pwds):
    # pwd_freqs = dict(pwds)
    pwd_list = list(map(lambda x: x[0], pwds))
    return (list(itertools.chain.from_iterable(map(all_prefixes, pwds))),
            list(itertools.chain.from_iterable(map(all_suffixes, pwds))))
# model = model_3()
# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
# for transformer
model = model_transformer()
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
# add for prefix training like lstm
# todo: delete, just for testing, the input and target 注释掉
train_data_total = read_train_data()
print('Total pwds: ', len(train_data_total), train_data_total[0])
# inputs, targets = generate_source_target(train_data_total)


# directly use the trained model to guess passwords
train = True
if not train:
    model = load_model('my_model.h5')
    x_train, y_train, x_val, y_val, output_y_train, output_y_val = gen_large_chunk_single_thread(inputs, targets, chunk_size=BATCH_SIZE, iteration=iteration)

    decoded = decode(
        model,
        tokens=[x.tolist() for x in x_val],
        start_token=c_table.encode_char(start_token),
        end_token=c_table.encode_char(end_token),
        pad_token=c_table.encode_char(pad_char),
        top_k=1,
        max_len=ENCODING_MAX_PASSWORD_LENGTH,
    )

    for i in range(20):
        print('-' * 50)
        password = c_table.decode(decoded[-i], calc_argmax=False)
        # print('former: ', x_val[i])
        print('former-decode: ', c_table.decode(x_val[-i], False))
        print('target: ', c_table.decode(y_val[-i], False))
        print('decoded: ', decoded[-i])
        print('guess: ', password)

        rowx, rowy = x_val[-i], y_val[-i]
        top_passwords = predict_top_most_likely_passwords_transformer_monte_carlo(model, rowx, rowy, 5)
        print('top passwords for: ', c_table.decode(rowx), ' [[[ ', top_passwords)

# Train the model each generation and show predictions against the validation data set.
iteration_total = 50
for iteration in range(1, iteration_total):
    x_train, y_train, x_val, y_val, output_y_train, output_y_val = gen_large_chunk_single_thread(inputs, targets, chunk_size=BATCH_SIZE, iteration=iteration)
    print()
    print('-' * 50)
    print('Iteration', iteration)

    print(list(x_train[iteration]))
    print(list(y_train[iteration]))
    print(output_y_train[iteration])
    # TODO: we need to update the loss to take into account that x!=y.
    # TODO: We could actually if it's an ADD, DEL or MOD.
    # TODO: Big improvement. We always have hello => hello1 right but never hello => 1hello
    # It's mainly because we pad after and never before. So the model has to shift all the characters.
    # And the risk for doing so is really since its a character based cross entropy loss.
    # Even though accuracy is very high it does not really prove things since Identity would have a high
    # Accuracy too.
    # One way to do that is to predict the ADD/DEL/MOD op along with the character of interest and the index
    # The index can just be a softmax over the indices of the password array, augmented (with a convention)

    # Select 10 samples from the validation set at random so we can visualize
    # errors.

    # model.fit(x_train, y_train,
    #           batch_size=BATCH_SIZE,
    #           epochs=5,
    #           validation_data=(x_val, y_val))

    # print(x_train[0], " : ", y_train[0], " : ", output_y_train[0])
    # transformer decode
    print('*' * 50)
    print(x_val.shape)
    # print(np.array([x.tolist() for x in x_val]))

    model.fit(
        x=[x_train, y_train],
        y=output_y_train,
        batch_size=BATCH_SIZE,
        epochs=5,
        validation_data=([x_val, y_val], output_y_val)
    )

    decoded = decode(
        model,
        tokens=[x.tolist() for x in x_val],
        start_token=c_table.encode_char(start_token),
        end_token=c_table.encode_char(end_token),
        pad_token=c_table.encode_char(pad_char),
        top_k=1,
        max_len=ENCODING_MAX_PASSWORD_LENGTH,
        max_repeat=30,  # 需要设置的较大，目前的batch_size 是25，初步认为得比这个大
    )
    #
    print([x.tolist() for x in x_val])
    print(decoded)
    #

    for i in range(20):
        print('-' * 50)
        password = c_table.decode(decoded[-i], calc_argmax=False)
        print('former-decode: ', c_table.decode(x_val[-i], False))
        print('target: ', c_table.decode(y_val[-i], False))
        print('decoded: ', decoded[-i])
        print('guess: ', password)

        rowx, rowy = x_val[np.array([-i])], y_val[np.array([-i])]
        print(rowx, rowy)
        top_passwords = predict_top_most_likely_passwords_transformer_monte_carlo(model, rowx, rowy, 5)
        print('top passwords for: ', c_table.decode(x_val[-i], False), ' [[[ ', top_passwords)

    for i in range(0):
        ind = np.random.randint(0, len(x_val))
        rowx, rowy = x_val[np.array([ind])], y_val[np.array([ind])]  # replace by x_val, y_val
        preds = model.predict_classes(rowx, verbose=0)
        q = c_table.decode(rowx[0])
        correct = c_table.decode(rowy[0])
        guess = c_table.decode(preds[0], calc_argmax=False)
        top_passwords = predict_top_most_likely_passwords_monte_carlo(model, rowx, 100)
        # p = model.predict(rowx, batch_size=32, verbose=0)[0]
        # p.shape (12, 82)
        # [np.random.choice(a=range(82), size=1, p=p[i, :]) for i in range(12)]
        # s = [np.random.choice(a=range(82), size=1, p=p[i, :])[0] for i in range(12)]
        # c_table.decode(s, calc_argmax=False)
        # Could sample 1000 and take the most_common()
        print('correct    :', correct)
        print('former :', q)
        print('guess  :', guess, end=' ')

        # if correct == guess:
        if correct.strip() in [vv.strip() for vv in top_passwords]:
            print(colors.ok + '☑' + colors.close)
        else:
            print(colors.fail + '☒' + colors.close)
        print('top    :', ', '.join(top_passwords))
        print('---')

model.save('my_model.h5')
