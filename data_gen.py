import pickle
from collections import Counter

import numpy as np
from tqdm import tqdm

from train_constants import ENCODING_MAX_PASSWORD_LENGTH, ENCODING_MAX_SIZE_VOCAB


def get_indices_token():
    return pickle.load(open('/tmp/indices_token.pkl', 'rb'))


def get_token_indices():
    return pickle.load(open('/tmp/token_indices.pkl', 'rb'))


def get_vocab_size():
    return len(get_token_indices())


def discard_password(password):
    return len(password) > ENCODING_MAX_PASSWORD_LENGTH or ' ' in password


class CharacterTable(object):
    """Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    """

    def __init__(self, chars):
        """Initialize character table.
        # Arguments
            chars: Characters that can appear in the input.
        """
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def add_token(self, char):
        self.char_indices[char] = len(self.char_indices)

    def encode_char(self, char):
        return self.char_indices[char]

    def encode(self, C, num_rows):
        """One hot encode given string C.
        # Arguments
            num_rows: Number of rows in the returned one hot encoding. This is
                used to keep the # of rows for each data the same.
        """
        # x = np.zeros((num_rows, len(self.chars)))
        # add for transformer
        x = np.zeros(num_rows)
        x[0] = self.encode_char('<START>')
        x[-1] = self.encode_char('<END>')
        for i in range(1, num_rows-1):
            try:
                c = C[i]
                if c not in self.char_indices:
                    # x[i, self.char_indices['？']] = 1
                    x[i] = self.char_indices['？']
                else:
                    # x[i, self.char_indices[c]] = 1
                    x[i] = self.char_indices[c]
            except IndexError:
                # x[i, self.char_indices[' ']] = 1
                x[i] = self.char_indices[' ']
        return x.tolist()

    def decode(self, x, calc_argmax=True):
        if calc_argmax:
            x = x.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in x)


class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'


def get_chars_and_ctable():
    chars = ''.join(list(get_token_indices().values()))
    ctable = CharacterTable(chars)

    return chars, ctable


def build_vocabulary(training_filename):
    vocabulary = {}
    print('Reading file {}.'.format(training_filename))
    with open(training_filename, 'rb') as r:
        for l in tqdm(r.readlines(), desc='Build Vocabulary'):
            line_id, x, y = l.decode('utf8').strip().split(' ||| ')
            if discard_password(y) or discard_password(x):
                continue
            for element in list(y + x):
                if element not in vocabulary:
                    vocabulary[element] = 0
                vocabulary[element] += 1
    vocabulary_sorted_list = sorted(dict(Counter(vocabulary).most_common(ENCODING_MAX_SIZE_VOCAB)).keys())
    oov_char = '？'
    pad_char = ' '
    print('Out of vocabulary (OOV) char is {}'.format(oov_char))
    print('Pad char is "{}"'.format(pad_char))
    vocabulary_sorted_list.append(oov_char)  # out of vocabulary.
    vocabulary_sorted_list.append(pad_char)  # pad char.
    print('Vocabulary = ' + ' '.join(vocabulary_sorted_list))
    token_indices = dict((c, i) for (c, i) in enumerate(vocabulary_sorted_list))
    indices_token = dict((i, c) for (c, i) in enumerate(vocabulary_sorted_list))
    assert len(token_indices) == len(indices_token)

    with open('/tmp/token_indices.pkl', 'wb') as w:
        pickle.dump(obj=token_indices, file=w)

    with open('/tmp/indices_token.pkl', 'wb') as w:
        pickle.dump(obj=indices_token, file=w)

    print('Done... File is /tmp/token_indices.pkl')
    print('Done... File is /tmp/indices_token.pkl')


def stream_from_file(training_filename):
    with open(training_filename, 'rb') as r:
        for l in r.readlines():
            _, x, y = l.decode('utf8').strip().split(' ||| ')
            if discard_password(y) or discard_password(x):
                continue
            yield x.strip(), y.strip()


class LazyDataLoader:
    def __init__(self, training_filename):
        self.training_filename = training_filename
        self.stream = stream_from_file(self.training_filename)

    def next(self):
        try:
            return next(self.stream)
        except:
            self.stream = stream_from_file(self.training_filename)
            return self.next()

    def statistics(self):
        max_len_value_x = 0
        max_len_value_y = 0
        num_lines = 0
        self.stream = stream_from_file(self.training_filename)
        for x, y in self.stream:
            max_len_value_x = max(max_len_value_x, len(x))
            max_len_value_y = max(max_len_value_y, len(y))
            num_lines += 1

        print('max_len_value_x =', max_len_value_x)
        print('max_len_value_y =', max_len_value_y)
        print('num_lines =', num_lines)
        return max_len_value_x, max_len_value_y, num_lines


if __name__ == '__main__':
    # how to use it.
    ldl = LazyDataLoader('/Users/HaoqiWu/BreachCompilationAnalysis/edit-distances/1.csv')
    print(ldl.statistics())
    while True:
        print(ldl.next())
