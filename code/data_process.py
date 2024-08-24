import numpy as np
import random as rd


def split_train_test(data, test_rate=0.3, maxitem=1000):
    train, test = [], []
    for i, item in enumerate(data[:maxitem]):
        if rd.random() > test_rate:
            train.append(item)
        else:
            test.append(item)
    return train, test


class BaseTextProcessor:
    def __init__(self, data, maxitem=1000):
        self.data = data[:maxitem]
        self.maxitem = maxitem
        self.words = {}
        self.len = 0
        self.train, self.test = split_train_test(data, test_rate=0.3, maxitem=maxitem)
        self.train_sentiment = [int(term[3]) for term in self.train]
        self.test_sentiment = [int(term[3]) for term in self.test]
        self.train_matrix = None
        self.test_matrix = None

    def initialize_matrices(self):
        self.train_matrix = np.zeros((len(self.train), self.len))
        self.test_matrix = np.zeros((len(self.test), self.len))

    def fill_matrix(self, data, matrix):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_matrix(self):
        self.fill_matrix(self.train, self.train_matrix)
        self.fill_matrix(self.test, self.test_matrix)


class BagOfWords(BaseTextProcessor):
    def get_words(self):
        all_data = self.train + self.test
        for term in all_data:
            words = term[2].upper().split()
            for word in words:
                if word not in self.words:
                    self.words[word] = len(self.words)
        self.len = len(self.words)
        self.initialize_matrices()

    def fill_matrix(self, data, matrix):
        for i, sample in enumerate(data):
            words = sample[2].upper().split()
            for word in words:
                if word in self.words:
                    matrix[i][self.words[word]] = 1


class Ngram(BaseTextProcessor):
    def __init__(self, data, dimension=3, maxitem=1000):
        self.dimension = dimension
        super().__init__(data, maxitem)

    def get_words(self):
        for d in range(1, self.dimension + 1):
            for term in self.data:
                words = term[2].upper().split()
                for i in range(len(words) - d + 1):
                    word = '_'.join(words[i:i + d])
                    if word not in self.words:
                        self.words[word] = len(self.words)
        self.len = len(self.words)
        self.initialize_matrices()

    def fill_matrix(self, data, matrix):
        for idx, sample in enumerate(data):
            words = sample[2].upper().split()
            for d in range(1, self.dimension + 1):
                for j in range(len(words) - d + 1):
                    word = '_'.join(words[j:j + d])
                    if word in self.words:
                        matrix[idx][self.words[word]] = 1
