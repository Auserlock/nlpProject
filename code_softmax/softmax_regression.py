import numpy as np
import random as rd


class Softmax:
    def __init__(self, sample_num, features, categories=5):
        self.sample_num = sample_num  #集合中有多少条数据
        self.features = features  #特征的维度
        self.categories = categories  #类别的数量
        self.W = np.random.randn(features, categories)  #权重矩阵W初始化为特征数*类别数的随机矩阵

    def softmax_calculation(self, vector):
        """计算softmax函数值，x是一个向量(softmax函数只依赖于输入向量的相对大小，所以减去最大值并将其映射到0附近不会改变结果)"""
        exp = np.exp(vector - np.max(vector))
        return exp / exp.sum()

    def softmax_all(self, matrix):
        """对矩阵进行softmax计算"""
        matrix -= np.max(matrix, axis=1, keepdims=True)  #逐行减去最大值
        matrix = np.exp(matrix)  #对矩阵中的每个元素进行指数运算
        matrix /= np.sum(matrix, axis=1, keepdims=True)  #逐行求和并求比例
        return matrix

    def oneHotVector_transition(self, y):
        """转换为one-hot向量"""
        ans = np.array([0] * self.categories)
        ans[y] = 1
        return ans.reshape(-1, 1)

    def prediction(self, matrix):
        """给定矩阵matrix(matrix是数据条目数*特征数的矩阵),预测每一条的类别,返回类别的索引"""
        prob = self.softmax_all(matrix.dot(self.W))
        return prob.argmax(axis=1)

    def correct_rate(self, train_matrix, train_sentiment, test_matrix, test_y):
        """计算分类的准确率"""
        # train set
        n_train = len(train_matrix)
        pred_train = self.prediction(train_matrix)
        train_correct = sum([train_sentiment[i] == pred_train[i] for i in range(n_train)]) / n_train
        # test set
        n_test = len(test_matrix)
        pred_test = self.prediction(test_matrix)
        test_correct = sum([test_y[i] == pred_test[i] for i in range(n_test)]) / n_test
        print(train_correct, test_correct)
        return train_correct, test_correct

    def regression(self, data_matrix, sentiment, alpha, times, strategy="mini", mini_size=100):
        """Softmax regression, data_matrix是数据条目数*特征数的矩阵，sentiment是情感标签，alpha是学习率，times是迭代次数，strategy是优化策略，mini_size
        是mini-batch的大小"""
        #   这里选择minibatch的规模为100，每次迭代选择100个样本进行梯度下降(1%)
        if self.sample_num != len(data_matrix) or self.sample_num != len(sentiment):
            raise Exception("Sample size does not match!")
        if strategy == "mini":
            for i in range(times):
                increment = np.zeros((self.features, self.categories))  # The gradient
                for j in range(mini_size):  # Choose a mini-batch of samples
                    k = rd.randint(0, self.sample_num - 1)
                    yhat = self.softmax_calculation(self.W.T.dot(data_matrix[k].reshape(-1, 1)))
                    increment += data_matrix[k].reshape(-1, 1).dot(
                        (self.oneHotVector_transition(sentiment[k]) - yhat).T)
                self.W += alpha / mini_size * increment
        elif strategy == "shuffle":
            for i in range(times):
                k = rd.randint(0, self.sample_num - 1)  # Choose a sample
                yhat = self.softmax_calculation(self.W.T.dot(data_matrix[k].reshape(-1, 1)))
                increment = data_matrix[k].reshape(-1, 1).dot(
                    (self.oneHotVector_transition(sentiment[k]) - yhat).T)  # The gradient
                self.W += alpha * increment
        elif strategy == "batch":
            for i in range(times):
                increment = np.zeros((self.features, self.categories))  # The gradient
                for j in range(self.sample_num):  # Calculate all samples
                    yhat = self.softmax_calculation(self.W.T.dot(data_matrix[j].reshape(-1, 1)))
                    increment += data_matrix[j].reshape(-1, 1).dot(
                        (self.oneHotVector_transition(sentiment[j]) - yhat).T)
                self.W += alpha / self.sample_num * increment
        else:
            raise Exception("Unknown strategy")
