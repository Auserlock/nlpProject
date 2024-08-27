import pandas as pd
from sklearn.metrics import accuracy_score


class AccuracyEvaluator:
    def __init__(self, predictions_path,sampleSubmission_path):

        self.predictions_path = predictions_path
        self.sampleSubmission_path = sampleSubmission_path
    def load_data(self):
        """
        加载预测结果和真实标签数据，并根据PhraseId进行合并。
        """
        # 加载模型预测的结果
        predicted_df = pd.read_csv(self.predictions_path)

        # 加载测试集的真实标签
        true_labels_df = pd.read_csv(self.sampleSubmission_path)

        # 根据 PhraseId 合并两个数据集，以确保对齐
        merged_df = pd.merge(predicted_df, true_labels_df, on='PhraseId', suffixes=('_pred', '_true'))

        return merged_df

    def calculate_accuracy(self):
        """
        计算并返回测试集的准确率。
        """
        # 加载并合并数据
        merged_df = self.load_data()

        # 提取预测的标签和真实的标签
        y_pred = merged_df['Sentiment_pred']
        y_true = merged_df['Sentiment_true']

        # 计算准确率
        accuracy = accuracy_score(y_true, y_pred)
        return accuracy
