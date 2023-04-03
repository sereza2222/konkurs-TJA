import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize


class TextNormalizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        X_copy = X.copy()
        for i in range(len(X)):
            X_copy[i] = ' '.join([token.lower()
                                  for token in word_tokenize(X_copy[i])])
        return X_copy

class WordExtractor(BaseEstimator, TransformerMixin):
    def init(self, stop_words):
        self.stop_words = stop_words

    def fit(self, X, y=None, **fit_params):
        self.general_freq = FreqDist()
        for document in X:
            tokens = word_tokenize(document)
            freq = FreqDist(tokens)
            self.general_freq.update(freq)
        self.hapaxes = self.general_freq.hapaxes()
        return self

    def transform(self, X, y=None, **fit_params):
        X_copy = X.copy()
        for i in range(len(X)):
            X_copy[i] = ' '.join([token for token in word_tokenize(X[i])
                                  if token not in self.hapaxes and
                                  token not in self.stop_words])
        return X_copy

class ApplyStemmer(BaseEstimator, TransformerMixin):
    def init(self, stemmer):
        self.stemmer = stemmer

    def fit(self, X, y=None, **fit_tranform):
        return self

    def transform(self, X, y=None, **fit_tranform):
        X_copy = X.copy()
        for i in range(len(X)):
            X_copy[i] = ' '.join([self.stemmer.stem(token)
                                  for token in word_tokenize(X_copy[i])])
        return X_copy