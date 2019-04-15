# To Activate Tensorboard
###############################################################################
#<open Anaconda prompt>
#<activate octenv>
#tensorboard --logdir=C:\Users\user\logs\keras_model_20190414_1832 --host localhost --port 8088
#<in browser> : http://localhost:8088/


# Import Packages
###############################################################################
import numpy as np, pandas as pd, tensorflow as tf, os
import os, re, itertools, time, datetime, gc, random
from os.path import getsize, basename
from tqdm import tqdm
from string import punctuation
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, LSTM, Dropout, Activation, Bidirectional
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard
from keras.utils import multi_gpu_model
from keras import backend as K, optimizers
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tensorflow.python.client import device_lib

# Data Processing Configuration
##############################################################################
# Folders with Source Data
liberal_folders = ['reddit_antira',
                   'reddit_democrats',
                   'reddit_demsocialist',
                   'reddit_esist',
                   'reddit_impeach_trump',
                   'reddit_liberal',
                   'reddit_progressive',
                   'reddit_radicalfeminism',
                   'reddit_feminism',
                   'reddit_socialdemocracy',
                   'reddit_voteblue',
                   'reddit_the_mueller',
                   'reddit_enoughtrumpspam',
                   'reddit_twoxchromosomes',
                   'reddit_latestagecapitalism',
                   'reddit_fuckthealtright',
                   'reddit_neoliberal']

conserv_folders = ['reddit_conservative',
                   'reddit_conservativelounge',
                   'reddit_louderwithcrowder',
                   'reddit_republican',
                   'reddit_tea_party',
                   'reddit_the_donald',
                   'reddit_jordanpeterson',
                   'reddit_mensrights']

subfold_name = '/page_links/'
master_folder = 'D:/poli_text/'
vocab_size = 100000
max_length = 75
embed_dim = 150

# Model Configuration
##############################################################################
model_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
model_save_name = "D:/poli_text/model_dir/keras_model_{dt_tm}.hdf5".format(dt_tm = model_timestamp)
result_folder = 'D:/poli_text/pyscripts/raw_embed_train_scripts/param_results/'
log_folder = 'D:/poli_text/pyscripts/log_files/'
num_epochs = 20
max_worse_epochs = 1
h_layer_sizes = [150, 100, 50]
dropout_rates = [0.5, 0.5, 0.0]
learning_rate = 0.00025
batch_size = 2500
validation_percent = 0.15

### Misc. Manipulation Functions
##############################################################################
def unnest_list_of_lists(LOL):
        return list(itertools.chain.from_iterable(LOL))
    
class poli_file_paths:
    def __init__(self, master_dir, con_folders, lib_folders, subfolder_name):
        self.master_dir = master_dir
        self.con_folders = con_folders
        self.lib_folders = lib_folders
        self.subfolder_name = subfolder_name
    """return input files by class given directory, folder and subfolder names"""
    
    # Return Full Paths of Liberal Text Files    
    def lib_file_paths(self):
        output = []
        for folder in self.lib_folders:
            path_str = "{a}{b}{c}".format(a = self.master_dir,
                                          b = folder,
                                          c = self.subfolder_name) 
            output.append([path_str + p for p in os.listdir(path_str)])
        return unnest_list_of_lists(output)
    
    # Return Full Paths of Conservative Text Files 
    def con_file_paths(self):
        output = []
        for folder in self.con_folders:
            path_str = "{a}{b}{c}".format(a = self.master_dir,
                                          b = folder,
                                          c = self.subfolder_name) 
            output.append([path_str + p for p in os.listdir(path_str)])
        return unnest_list_of_lists(output)
    
# Text Reading and Cleaning Functions
##############################################################################
def read_line_delim_txt(files, encoding = 'utf-8'):
    """read and process text files in list of folder paths"""
    output = []
    for fpath in files:
        output.append(open(fpath, encoding = encoding).read().splitlines())
    return unnest_list_of_lists(output)

def filter_clean_txt(txt_dat_lst,
                     lemmatizer = WordNetLemmatizer(),
                     stop_list = set(stopwords.words('english')) ):
    """filter and clean list of text objects"""
    ### Define Inner Functions ###
    def remove_http_https(some_string):
        # remove 'http' and 'https' + following characters from <some_string> #
        return re.sub(r'http\S+', '', str(some_string), flags = re.MULTILINE)
    
    def remove_nonalpha_lower(some_string):
        # remove non-alphabetic characters from <some_string> #
        return re.sub(r"([^a-zA-Z]|_)+", " ", some_string).lower()
    
    def remove_stopwords(some_string, stop_list = stop_list):
        # remove stopwords from <some_string> #
        return ' '.join([w for w  in some_string.split() if w not in stop_list])
    
    def apply_wordnet_lemm(some_string):
        # apply wordnet lemmatizer on space-separated words from <some_string>
        return ' '.join([lemmatizer.lemmatize(s) for s in some_string.split()])
    
    ### Apply Inner Functions ###
    output = []
    for txt in tqdm(txt_dat_lst):
        proc_txt = remove_http_https(txt)
        proc_txt = remove_nonalpha_lower(proc_txt)
        proc_txt = remove_stopwords(proc_txt)
        output.append(apply_wordnet_lemm(proc_txt))
    return output
    
def std_tokenize_proc(txt_doc, vocab_size, maxlen):
    """tokenize text data for learned embedding training"""
    tokenizer = Tokenizer(num_words = vocab_size, filters = '')
    tokenizer.fit_on_texts(txt_doc)
    sequences = tokenizer.texts_to_sequences(txt_doc)
    x_data = pad_sequences(sequences, maxlen = maxlen)
    return x_data

### Model Functions
##############################################################################
    
def curr_time_str(time_format = '%Y-%m-%d %H:%M:%S'):
    """return current time as string"""
    return datetime.datetime.utcfromtimestamp(time.time()).strftime(time_format)

def curr_time_str_custom_msg(custom_msg, time_format = '%Y-%m-%d %H:%M:%S'):
    """print current time as string & whatever additional text user wants"""
    return curr_time_str(time_format = time_format) + ' ' + custom_msg

def get_number_gpu():
    """number of available GPUs"""
    n_gpu = len([x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU'])
    return n_gpu

def bidir_lstm_parallel_gpu_train(save_model_as,
                                  max_worse_epochs,
                                  embed_dim,
                                  input_len,
                                  h_layer_sizes,
                                  learning_rate,
                                  dropout_rates,
                                  validation_percent,
                                  num_epochs,
                                  batch_size,
                                  vocab_size,
                                  train_y,
                                  train_x,
                                  log_folder,
                                  start_timestamp):
    """train binary bidirectional LSTM model on all available GPUs"""
    print(curr_time_str_custom_msg(custom_msg = 'begin training'))
    K.clear_session()
    
    ### Checkpoint and Logging ###
    check_point = ModelCheckpoint(save_model_as, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'min')
    early_stop = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = max_worse_epochs)
    csv_logger = CSVLogger("{fold}log_{ts}.csv".format(fold = log_folder,
                                                       ts = start_timestamp))
    tensorboard = TensorBoard(log_dir = 'logs/{}'.format(save_model_as.split('/')[-1].split('.')[0]),
                              histogram_freq = 0,
                              write_graph = True,
                              write_images = False)
    ### Define Model Arch. ###
    model = Sequential()
    model.add(Embedding(input_dim = vocab_size,
                        output_dim = embed_dim,
                        input_length = input_len,
                        trainable = True))
    if len(h_layer_sizes) == 1:
        model.add(Dropout(dropout_rates[0]))
        model.add(Bidirectional(LSTM(h_layer_sizes[0])))
    elif len(h_layer_sizes) == 2:
        model.add(Dropout(dropout_rates[0]))
        model.add(Bidirectional(LSTM(h_layer_sizes[0], return_sequences = True)))
        model.add(Dropout(dropout_rates[1]))
        model.add(Bidirectional(LSTM(h_layer_sizes[1])))
    elif len(h_layer_sizes) == 3:
        model.add(Dropout(dropout_rates[0]))
        model.add(Bidirectional(LSTM(h_layer_sizes[0], return_sequences = True)))
        model.add(Dropout(dropout_rates[1]))
        model.add(Bidirectional(LSTM(h_layer_sizes[1], return_sequences = True)))
        model.add(Dropout(dropout_rates[2]))
        model.add(Bidirectional(LSTM(h_layer_sizes[2])))
    elif len(h_layer_sizes) == 4:
        model.add(Dropout(dropout_rates[0]))
        model.add(Bidirectional(LSTM(h_layer_sizes[0], return_sequences = True)))
        model.add(Dropout(dropout_rates[1]))
        model.add(Bidirectional(LSTM(h_layer_sizes[1], return_sequences = True)))
        model.add(Dropout(dropout_rates[2]))
        model.add(Bidirectional(LSTM(h_layer_sizes[2], return_sequences = True)))
        model.add(Dropout(dropout_rates[3]))
        model.add(Bidirectional(LSTM(h_layer_sizes[3])))
    elif len(h_layer_sizes) == 5:
        model.add(Dropout(dropout_rates[0]))
        model.add(Bidirectional(LSTM(h_layer_sizes[0], return_sequences = True)))
        model.add(Dropout(dropout_rates[1]))
        model.add(Bidirectional(LSTM(h_layer_sizes[1], return_sequences = True)))
        model.add(Dropout(dropout_rates[2]))
        model.add(Bidirectional(LSTM(h_layer_sizes[2], return_sequences = True)))
        model.add(Dropout(dropout_rates[3]))
        model.add(Bidirectional(LSTM(h_layer_sizes[3], return_sequences = True)))
        model.add(Dropout(dropout_rates[4]))
        model.add(Bidirectional(LSTM(h_layer_sizes[4])))
    model.add(Dense(1, activation = 'sigmoid'))
    parallel_model = multi_gpu_model(model, gpus = get_number_gpu())
    parallel_model.compile(loss='binary_crossentropy',
                           optimizer = optimizers.Adam(lr = learning_rate), 
                           metrics = ['accuracy'],
                           weighted_metrics = ['accuracy'])
    
    ### Fit Model ###
    parallel_model.fit(train_x,
                       np.array(train_y),
                       validation_split = validation_percent,
                       epochs = num_epochs,
                       batch_size = batch_size,
                       class_weight = {0: 1., 1: 1/(sum(train_y) / len(train_y))},
                       callbacks = [check_point, early_stop, csv_logger, tensorboard])
    
    print(curr_time_str_custom_msg(custom_msg = 'end training'))

# Read & Process Text
##############################################################################
# All File Names
temp_input_files = poli_file_paths(master_dir = master_folder,
                              con_folders = conserv_folders,
                              lib_folders = liberal_folders,
                              subfolder_name = subfold_name)

# Conservative Text
temp_con_paths = temp_input_files.con_file_paths()
temp_con_txt = read_line_delim_txt(files = temp_con_paths)
temp_con_proc_txt = filter_clean_txt(txt_dat_lst = temp_con_txt)

# Liberal Text
temp_lib_paths = temp_input_files.lib_file_paths()
temp_lib_txt = read_line_delim_txt(files = temp_lib_paths)
temp_lib_proc_txt = filter_clean_txt(txt_dat_lst = temp_lib_txt)

# Create X and Y Data
y_dat = [1 for i in range(0, len(temp_lib_proc_txt))] + [0 for i in range(0, len(temp_con_proc_txt))]
x_dat = std_tokenize_proc(txt_doc = temp_lib_proc_txt + temp_con_proc_txt,
                          vocab_size = vocab_size,
                          maxlen = max_length)

# Delete Excess Objects
del temp_con_paths; del temp_con_txt; del temp_con_proc_txt;
del temp_lib_paths; del temp_lib_txt; del temp_lib_proc_txt;
gc.collect()

# Model Fitting
##############################################################################
bidir_lstm_parallel_gpu_train(save_model_as = model_save_name,
                              max_worse_epochs = max_worse_epochs,
                              embed_dim = embed_dim,
                              input_len = max_length,
                              h_layer_sizes = h_layer_sizes,
                              learning_rate = learning_rate,
                              dropout_rates = dropout_rates,
                              validation_percent = validation_percent,
                              num_epochs = num_epochs,
                              batch_size = batch_size,
                              vocab_size = vocab_size,
                              train_y = y_dat,
                              train_x = x_dat,
                              log_folder = log_folder,
                              start_timestamp = model_timestamp)
