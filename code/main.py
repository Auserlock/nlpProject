import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

# 读取数据与数据预处理
dir_data='./data/train.tsv'
dir_test='./data/test.tsv'

data=pd.read_csv(dir_data,sep='\t')
test=pd.read_csv(dir_test,sep='\t',keep_default_na=False)

x_data=data['Phrase']
y_data=data['Sentiment']

x_test=test['Phrase']
y_test=test['PhraseId']

# 划分训练集和验证集
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=42, shuffle=True)

# 提取特征(TF-IDF)
tfidf_vec=TfidfVectorizer()
tfidf_train_matrix=tfidf_vec.fit_transform(x_train)
tfidf_val_matrix=tfidf_vec.transform(x_val)
tfidf_test_matrix=tfidf_vec.transform(x_test)

tfidf_ngram_vec=TfidfVectorizer(ngram_range=(2, 3), max_features=50000)
tfidf_ngram_train_matrix=tfidf_ngram_vec.fit_transform(x_train)
tfidf_ngram_val_matrix=tfidf_ngram_vec.transform(x_val)
tfidf_ngram_test_matrix=tfidf_ngram_vec.transform(x_test)

# 合并特征
train_features = hstack([tfidf_train_matrix, tfidf_ngram_train_matrix])
val_features = hstack([tfidf_val_matrix, tfidf_ngram_val_matrix])
test_features = hstack([tfidf_test_matrix, tfidf_ngram_test_matrix])

# 降低维度
# svd = TruncatedSVD(n_components=100)
# train_features_svd = svd.fit_transform(train_features)
# val_features_svd = svd.transform(val_features)
# test_features_svd = svd.transform(test_features)

# 模型构建
multi_class = 'multinomial'
clf = LogisticRegression(random_state=0, multi_class=multi_class, max_iter=1000)
clf.fit(train_features, y_train)

# 测试
predict=clf.predict(val_features)

acc=accuracy_score(y_val,predict)
precision=precision_score(y_val,predict,average='macro')
recall=recall_score(y_val,predict,average='macro')
f1=f1_score(y_val,predict,average='macro')

print('acc:{0},precision:{1},recall:{2},f1:{3}.'.format(acc,precision,recall,f1))

# 测试集
test['Sentiment']=clf.predict(test_features)
test.to_csv('../data/submission.csv',index=False,columns=['PhraseId','Sentiment'])
