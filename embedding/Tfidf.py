import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer


class Tfidf:
    def __init__(self, corpus):
        self.tokenizer = Tokenizer(oov_token='<OOV>')
        self.tokenizer.fit_on_texts(corpus)
        self.vocab = self.tokenizer.word_index
        self.vocab_size = len(self.vocab) + 1

    def tokenize(self):
        return self.tokenizer.texts_to_matrix(self.corpus, mode='tfidf')