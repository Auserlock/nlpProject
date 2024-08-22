import pandas as pd
import numpy as np
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

dir_data = '../data/train.tsv'
dir_test = '../data/test.tsv'


class DataProcess(object):
    def __init__(self, data_path=None, model_path=None):
        self.data_path = data_path
        self.model_path = model_path

    def read_data(self):
        data = pd.read_csv(self.data_path, sep='\t')
        return data

    # 数据预处理
    def process_data(self, data, test_size=0.2):
        x = data['Phrase']
        y = data['Sentiment']
        return train_test_split(x, y, test_size=test_size, random_state=42)

    # 提取词袋模型特征
    def get_bow(self, x_train, x_val, min_df=1):
        vectorizer = CountVectorizer(min_df=min_df)
        x_train_bow = vectorizer.fit_transform(x_train)
        x_val_bow = vectorizer.transform(x_val)
        return x_train_bow, x_val_bow, vectorizer

    # 提取tf-idf特征(word)
    def get_tfidf(self, x_train, x_val, min_df=1):
        vectorizer = TfidfVectorizer(min_df=min_df)
        x_train_tfidf = vectorizer.fit_transform(x_train)
        x_val_tfidf = vectorizer.transform(x_val)
        return x_train_tfidf, x_val_tfidf, vectorizer

    # 提取tf-idf特征(ngram)
    def get_tfidf_ngram(self, x_train, x_val, min_df=1, ngram_range=(2, 3), max_features=None):
        vectorizer = TfidfVectorizer(min_df=min_df, ngram_range=ngram_range, max_features=max_features)
        x_train_tfidf_ngram = vectorizer.fit_transform(x_train)
        x_val_tfidf_ngram = vectorizer.transform(x_val)
        return x_train_tfidf_ngram, x_val_tfidf_ngram, vectorizer

    # 合并特征
    def combine_features(self, x_train, x_val, min_df=1, ngram_range=(2, 3), max_features=None):
        x_train_bow, x_val_bow, bow_vectorizer = self.get_bow(x_train, x_val, min_df=min_df)

        x_train_tfidf, x_val_tfidf, tfidf_vectorizer = self.get_tfidf(x_train, x_val, min_df=min_df)

        x_train_tfidf_ngram, x_val_tfidf_ngram, tfidf_ngram_vectorizer = self.get_tfidf_ngram(
            x_train, x_val, min_df=min_df, ngram_range=ngram_range, max_features=max_features)

        x_train_combined = hstack([x_train_bow, x_train_tfidf, x_train_tfidf_ngram])
        x_val_combined = hstack([x_val_bow, x_val_tfidf, x_val_tfidf_ngram])

        return x_train_combined, x_val_combined

    # 降低维度
    def apply_svd(self, x_train, x_val, n_components=1000):
        svd = TruncatedSVD(n_components=n_components)
        x_train_svd = svd.fit_transform(x_train)
        x_val_svd = svd.transform(x_val)
        return x_train_svd, x_val_svd

    # word2vec 特征提取
    def word2vec(self, x_train, x_val):
        pass

    # 提供数据
    def provide_data(self, min_df=1, n_components=1000):
        data = self.read_data()

        #  1、特征提取与组合
        x_train, x_val, y_train, y_val = self.process_data(data)
        x_train_combined, x_val_combined = self.combine_features(x_train, x_val, min_df=min_df)

        #  2、降低维度
        x_train_svd, x_val_svd = self.apply_svd(x_train_combined, x_val_combined, n_components=n_components)
        return x_train_svd, x_val_svd, y_train, y_val

    # 迭代器，将数据分批传给模型
    def batch_iter(self, x, y, batch_size=64):
        data_len = len(x)
        num_batch = int((data_len - 1) / batch_size) + 1
        indices = np.random.permutation(np.arange(data_len))
        x_shuffle = x[indices]
        y_shuffle = y[indices]
        for i in range(num_batch):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, data_len)
            yield x_shuffle[start_id: end_id], y_shuffle[start_id: end_id]
