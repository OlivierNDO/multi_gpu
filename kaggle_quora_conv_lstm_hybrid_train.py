# Import Packages
##############################################################################
import numpy as np, pandas as pd, tensorflow as tf, matplotlib.pyplot as plt, seaborn as sns
import gc, pyodbc, random, tqdm, re, string, itertools, time
from string import punctuation
from sqlalchemy import create_engine, MetaData, Table, select
from os.path import getsize, basename
from __future__ import division
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
from collections import Counter
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.utils import multi_gpu_model
from keras import backend as K
from keras.models import load_model

script_start = time.time()

# User Input
##############################################################################
glove_txt_file = 'D:/glove/glove.840B.300d.txt'
train_file = 'D:/quora/data/train.csv'
test_file = 'D:/quora/data/test.csv'

# Define Functions
##############################################################################
def seconds_to_time(sec):
    if (sec // 3600) == 0:
        HH = '00'
    elif (sec // 3600) < 10:
        HH = '0' + str(int(sec // 3600))
    else:
        HH = str(int(sec // 3600))
    min_raw = (np.float64(sec) - (np.float64(sec // 3600) * 3600)) // 60
    if min_raw < 10:
        MM = '0' + str(int(min_raw))
    else:
        MM = str(int(min_raw))
    sec_raw = (sec - (np.float64(sec // 60) * 60))
    if sec_raw < 10:
        SS = '0' + str(int(sec_raw))
    else:
        SS = str(int(sec_raw))
    return HH + ':' + MM + ':' + SS + ' (hh:mm:ss)'

def sec_to_time_elapsed(end_tm, start_tm, return_time = False):
    sec_elapsed = (np.float64(end_tm) - np.float64(start_tm))
    if return_time:
        return seconds_to_time(sec_elapsed)
    else:
        print('Execution Time: ' + seconds_to_time(sec_elapsed))

def load_glove(glove_file_path, progress_print = 5000, encoding_type = 'utf8'):
    """load glove (Stanford NLP) file and return dictionary"""
    num_lines = sum(1 for line in open(glove_txt_file, encoding = encoding_type))
    embed_dict = dict()
    line_errors = []
    f = open(glove_file_path, encoding = encoding_type)
    for i, l in enumerate(f):
        l_split = l.split()
        try:
            embed_dict[l_split[0]] = np.asarray(l_split[1:], dtype = 'float32')
        except:
            line_errors.append(1)
        if ((i / progress_print) > 0) and (float(i / progress_print) == float(i // progress_print)):
            print(str(int(i / 1000)) + ' K of ' + str(int(num_lines / 1000)) + ' K lines completed')
        else:
            pass
    f.close()
    print('failed lines in file: ' + str(int(np.sum(line_errors))))
    return embed_dict
    
def clean_tokenize(some_string):
    """split on punct / whitespace, make lower case, join back together"""
    pattern = re.compile(r'(\s+|[{}])'.format(re.escape(punctuation)))
    clean_lower = ' '.join([part.lower() for part in pattern.split(some_string) if part.strip()])
    return clean_lower

def glove_tokenize_proc(csv_file_path, csv_txt_col, glove_file_path, vocab_size, maxlen, y_col = None):
    """ - read csv file with text in 'csv_txt_col' column"""
    """ - process into 300-dimension embeddings based on Stanford glove embeddings"""
    start_tm = time.time()
    glove_dict = load_glove(glove_file_path)
    df = pd.read_csv(csv_file_path)
    df[csv_txt_col] = df[csv_txt_col].apply(clean_tokenize)
    tokenizer = Tokenizer(num_words = vocab_size)
    tokenizer.fit_on_texts(df[csv_txt_col])
    sequences = tokenizer.texts_to_sequences(df[csv_txt_col])
    x_data = pad_sequences(sequences, maxlen = maxlen)
    embed_wt_matrix = np.zeros((vocab_size, 300))
    for x, i in tokenizer.word_index.items():
        if i > (vocab_size - 1):
            break
        else:
            embed_vec = glove_dict.get(x)
            if embed_vec is not None:
                embed_wt_matrix[i] = embed_vec
    end_tm = time.time()
    sec_to_time_elapsed(end_tm, start_tm)
    if y_col:
        y_data = [y for y in df[y_col]]
        return y_data, x_data, embed_wt_matrix
    else:
        return x_data, embed_wt_matrix

# Execute Functions
##############################################################################
train_y, train_x, embed_wts = glove_tokenize_proc(csv_file_path = train_file,
                                                  csv_txt_col = 'question_text',
                                                  glove_file_path = glove_txt_file,
                                                  vocab_size = 50000,
                                                  maxlen = 50,
                                                  y_col = 'target')

proc_end = time.time()


# Training on Two GPUs
K.clear_session()
file_path = 'D:/quora/model_dir/keras_model.hdf5'
input_len = 50
embed_dim = 300
convd1_size = 12
conv_pool_size = 4
lstm_size = 100
dropout_rate_1 = 0.5
validation_perc = 0.2
num_epochs = 100
batch_size = 3500
trn_pos_perc = sum(train_y) / len(train_y)
class_weight = {0: 1., 1: 1/trn_pos_perc}
vocab_size = 50000

check_point = ModelCheckpoint(file_path,
                              monitor = 'val_loss',
                              verbose = 1,
                              save_best_only = True,
                              mode = 'min')

early_stop = EarlyStopping(monitor = 'val_loss',
                           mode = 'min',
                           patience = 3)

model_glove = Sequential()
model_glove.add(Embedding(vocab_size, embed_dim, input_length = input_len,
                          weights = [embed_wts],
                          trainable = False))
model_glove.add(Dropout(dropout_rate_1))
model_glove.add(Conv1D(convd1_size, 5, activation = 'relu'))
model_glove.add(MaxPooling1D(pool_size = conv_pool_size))
model_glove.add(LSTM(lstm_size))
model_glove.add(Dense(1, activation = 'sigmoid'))
parallel_model = multi_gpu_model(model_glove, gpus = 2)
parallel_model.compile(loss='binary_crossentropy',
                    optimizer = 'adam',
                    metrics = ['accuracy'])
parallel_model.fit(train_x,
                   np.array(train_y),
                   validation_split = validation_perc,
                   epochs = num_epochs,
                   batch_size = batch_size,
                   class_weight = class_weight,
                   callbacks = [check_point, early_stop])

train_end = time.time()


sec_to_time_elapsed(train_end, proc_end)


"""
loaded_model = load_model(file_path)
test_prediction = loaded_model.predict(test_data)
test_prediction_binary = [int(np.round(i,0)) for i in test_prediction]
test_prediction_df = pd.DataFrame({'qid': [i for i in test['qid']],
                                   'prediction': test_prediction_binary})
"""


# Time Tracking
# Data Proc Time:
sec_to_time_elapsed(proc_end, script_start)



