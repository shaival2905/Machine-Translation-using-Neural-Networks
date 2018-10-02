# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 09:21:18 2018

@author: shaival
"""

import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import GRU, Input, Dense, TimeDistributed, Activation, RepeatVector, Bidirectional
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy

import os

def load_data(path):
    """
    Load dataset
    """
    input_file = os.path.join(path)
    with open(input_file, "r", encoding='utf-8') as f:
        data = f.read()
        
    return data.split('\n')
english_data = load_data('D:/Nirma/minor project/parallel/IITB.en-hi.en')#("D:/Nirma/minor project/dev_test_tokenized/dev.en")
hindi_data = load_data('D:/Nirma/minor project/parallel/IITB.en-hi.hi')
#hindi_data = load_data("D:/Nirma/minor project/dev_test_tokenized/dev.hi")

tokenizer = Tokenizer(char_level = False)
tokenizer.fit_on_texts(english_data)
token_eng = tokenizer.texts_to_sequences(english_data)
max_length_seq = max([len(txt) for txt in token_eng])




