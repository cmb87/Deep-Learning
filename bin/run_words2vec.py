import tensorflow as tf
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

import collections
import math
import os
import errno
import random
import zipfile

from six.moves import urllib
from six.moves import xrange

from collections import Counter


data_dir = '../data/04-Recurrent-Neural-Networks/word2vec_data/words'
data_url = "http://mattmahoney.net/dc/text8.zip"

def fetch_words_data(url=data_url, words_data=data_dir):
    # Make dirs
    os.makedirs(words_data, exist_ok=True)

    zip_path = os.path.join(words_data, "words.zip")

    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path) as f:
        data = f.read(f.namelist()[0])

    return data.decode("ascii").split()


def create_counts(vocab_size=50000):
    words = fetch_words_data()
    vocab = [] + Counter(words).most_common(vocab_size)

    vocab = np.array([word for word, _ in vocab])
    dictionary = {word: code for code, word in enumerate(vocab)}

    data = np.array([dictionary.get(word,0) for word in words])

    return data, vocab




data, vocab = create_counts()

print(vocab[data[0:100]])