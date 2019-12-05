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

from data_gen import get_chars_and_ctable, colors
from train_constants import ENCODING_MAX_PASSWORD_LENGTH, ENCODING_MAX_SIZE_VOCAB

INPUT_MAX_LEN = ENCODING_MAX_PASSWORD_LENGTH
OUTPUT_MAX_LEN = ENCODING_MAX_PASSWORD_LENGTH

try:
    chars, c_table = get_chars_and_ctable()
except FileNotFoundError:
    print('Run first run_encoding.py to generate the required files.')
    exit(1)


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


def gen_large_chunk_single_thread(inputs_, targets_, chunk_size):
    random_indices = np.random.choice(a=range(len(inputs_)), size=chunk_size, replace=True)
    sub_inputs = inputs_[random_indices]
    sub_targets = targets_[random_indices]

    x = np.zeros((chunk_size, ENCODING_MAX_PASSWORD_LENGTH))
    y = np.zeros((chunk_size, ENCODING_MAX_PASSWORD_LENGTH))
    output = np.zeros((chunk_size, ENCODING_MAX_PASSWORD_LENGTH, 1))


    for i_, element in enumerate(sub_inputs):
        # print('asd ', element)
        # todo: add segment and word embeddings
        x[i_] = c_table.encode(element, ENCODING_MAX_PASSWORD_LENGTH)
    for i_, element in enumerate(sub_targets):
        y[i_] = c_table.encode(element, ENCODING_MAX_PASSWORD_LENGTH)
        res = [[x] for x in c_table.encode(element, ENCODING_MAX_PASSWORD_LENGTH)[1:]]
        res.append([0])
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
        token_num=len(chars) + 2,
        embed_dim=len(chars) + 2,
        encoder_num=3,
        decoder_num=2,
        head_num=2,
        hidden_dim=120,
        attention_activation='relu',
        feed_forward_activation='relu',
        dropout_rate=0.05,
        embed_weights=np.random.random((84, 84)),
    )
    return m

model = model_transformer()
chars = chars + '<START>' + '<END>'
print('chars: [ ', chars, ' ]')
# add for transformer
c_table.add_token('<START>')
c_table.add_token('<END>')
# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train the model each generation and show predictions against the validation data set.
iteration_total = 500
for iteration in range(1, iteration_total):
    x_train, y_train, x_val, y_val, output_y_train, output_y_val = gen_large_chunk_single_thread(inputs, targets, chunk_size=BATCH_SIZE)
    print()
    print('-' * 50)
    print('Iteration', iteration)
    # TODO: we need to update the loss to take into account that x!=y.
    # TODO: We could actually if it's an ADD, DEL or MOD.
    # TODO: Big improvement. We always have hello => hello1 right but never hello => 1hello
    # It's mainly because we pad after and never before. So the model has to shift all the characters.
    # And the risk for doing so is really since its a character based cross entropy loss.
    # Even though accuracy is very high it does not really prove things since Identity would have a high
    # Accuracy too.
    # One way to do that is to predict the ADD/DEL/MOD op along with the character of interest and the index
    # The index can just be a softmax over the indices of the password array, augmented (with a convention)

    print(x_train[0], " : ", y_train[0], " : ", output_y_train[0])
    #
    model.fit(
        x=[x_train, y_train],
        y=output_y_train,
        batch_size=BATCH_SIZE,
        epochs=1,
        validation_data=([x_val, y_val], output_y_val)
    )
    # Select 10 samples from the validation set at random so we can visualize
    # errors.
    print('*' * 50)
    print(x_val.shape)
    decoded = decode(
        model,
        tokens=[x.tolist() for x in x_val],
        start_token=c_table.encode_char('<START>'),
        end_token=c_table.encode_char('<END>'),
        pad_token=0,
        max_len=14,
    )
    print(decoded)

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
        print('new    :', correct)
        print('former :', q)
        print('guess  :', guess, end=' ')

        # if correct == guess:
        if correct.strip() in [vv.strip() for vv in top_passwords]:
            print(colors.ok + '☑' + colors.close)
        else:
            print(colors.fail + '☒' + colors.close)
        print('top    :', ', '.join(top_passwords))
        print('---')
