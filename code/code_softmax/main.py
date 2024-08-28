# Description: This file is the main file for the project. It will be used to run the project.
import numpy as np
import csv
import random as rd
from data_process import BagOfWords, Ngram
from process_compare import alpha_gradient

with open('../data/train.tsv') as f:
    tsv_data = csv.reader(f, delimiter='\t')
    temp = list(tsv_data)

data = temp[1:]  #取出数据(去掉第一行)
max_item = 1000  #最大数据量
rd.seed(2024)  #设置随机种子,保证每次结果一样,便于比较
np.random.seed(2024)

# 词袋模型和N-gram模型提取特征
# 分别进行两种数据量的处理，同时比较词袋模型和N-gram模型(bigram, trigram)的特征提取效果
bag = BagOfWords(data, max_item)
bag.get_words()
bag.get_matrix()

bigram = Ngram(data, 2, max_item)
bigram.get_words()
bigram.get_matrix()

trigram = Ngram(data, 3, max_item)
trigram.get_words()
trigram.get_matrix()

alpha_gradient(bag, bigram, trigram, 10000, 10)
alpha_gradient(bag, bigram, trigram, 50000, 10)