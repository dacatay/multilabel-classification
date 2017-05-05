import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import time
import os
import json
import string

from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag

from sklearn.base import BaseEstimator, TransformerMixin


import settings


class NLTKPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, stopwords=None, punct=None, lower=True, strip=True):
        self.lower = lower
        self.strip = strip
        self.stopwords = stopwords or set(sw.words('english'))
        self.punct = punct or set(string.punctuation)
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        return self

    def inverse_transform(self, X):
        return [" ".join(doc) for doc in X]

    def transform(self, X):
        """
        :param X: list of original documents
        :return: list of tokenized documents
        """
        return [list(self.tokenize(doc)) for doc in X]

    def tokenize(self, document):
        # break raw document strings into sentences
        for sentence in sent_tokenize(document):
            # divide sentence into parts of speech tagged tokens
            for token, tag in pos_tag(wordpunct_tokenize(sentence)):
                # apply preprocessing to token
                token = token.lower() if self.lower else token
                token = token.strip() if self.strip else token
                token = token.strip('_') if self.strip else token
                token = token.strip('*') if self.strip else token
                # if stopword is encountered, ignore token and continue
                if token in self.stopwords:
                    continue
                # if punctuation is encountered, ignore token and continue
                if all(char in self.punct for char in token):
                    continue
                # lemmatize token and yield
                lemma = self.lemmatize(token, tag)
                yield lemma

    def lemmatize(self, token, tag):
        """
        :param token: not ignored token
        :param tag: 
        :return: lematized token
        """
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(tag[0], wn.NOUN)

        return self.lemmatizer.lemmatize(token, tag)


def main():
    # Create tag list and submission header from topicDictionary.txt
    submission_header = ['id']
    topics = []

    with open(r'data/topicDictionary.txt', 'r') as topics_file:
        for line in topics_file:
            line = line.strip('\n')
            submission_header.append(line)
            topics.append(line)
        topics_file.close()

    # List directory for training data files
    dirs = os.listdir(settings.PATH_TRAINING_DATA)

    # Import json file
    with open(settings.PATH_TRIMMED_TRAINING_DATA + dirs[0],  'r') as json_file:
        data = json.load(json_file)
        json_file.close()
    print(data)


if __name__ == '__main__':
    main()




