#
# train_constants.py: Contains the constants necessary for the encoding and the training phases.
#

# Maximum password length. Passwords greater than this length will be discarded during the encoding phase.
ENCODING_MAX_PASSWORD_LENGTH = 14

# Maximum number of characters for encoding. By default, we use the 80 most frequent characters and
# we bin the other ones in a OOV (out of vocabulary) group.
ENCODING_MAX_SIZE_VOCAB = 80

EMBEDDING_DIM = 100
oov_char = '？'  # 中文？作为vocabulary之外的字符
pad_char = '<PAD>'
start_token = '<START>'
end_token = '<END>'

sep = ' ||| '

INDICES_TOKEN_CHAR_PATH = '/tmp/indices_token.pkl'
TOKEN_INDICES_CHAR_PATH = '/tmp/token_indices.pkl'

INDICES_TOKEN_WORD_PATH = '/tmp/indices_token_word.pkl'
TOKEN_INDICES_WORD_PATH = '/tmp/token_indices_word.pkl'
