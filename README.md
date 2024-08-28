# nlpProject
nlp-beginner task1
# 项目介绍
该项目旨在使用两种不同的方法进行文本分类：Softmax 回归模型和神经网络模型。目标是将文本数据分类到预定义的类别中，并实现较高的准确率。项目包括特征提取技术，如词袋模型（Bag of Words）和 N-gram 模型，然后进行模型训练和评估。
# 目录结构
## 神经网络模型
### data_process.py
#### 数据读取
通过 read_data() 方法读取训练和测试数据，数据文件为 .tsv 格式，使用 pandas 库进行读取。
#### 数据预处理
通过 process_data() 方法，将数据集分割为训练集和验证集，并提取文本和标签。
#### 特征提取
词袋模型（Bag of Words）：通过 get_bow() 方法提取词频特征。
TF-IDF 特征：通过 get_tfidf() 方法提取基于词的 TF-IDF 特征；通过 get_tfidf_ngram() 方法提取基于 n-gram 的 TF-IDF 特征。
特征组合：通过 combine_features() 方法将词袋模型特征、TF-IDF 特征和 TF-IDF n-gram 特征组合在一起，形成最终的特征矩阵。
#### 降维处理
通过 apply_svd() 方法使用 TruncatedSVD 降低特征矩阵的维度，主要用于稀疏矩阵的降维。
#### 数据提供
provide_data() 方法综合使用上述方法，处理数据并返回最终用于训练和验证的数据集。
#### 转换为 TensorFlow 数据集
通过 get_dataset() 方法将 numpy 数组转换为 TensorFlow 数据集，便于后续神经网络模型的训练。
### lr_config.py
存储训练模型时所需的各种路径和参数
### lr_model.py
用于创建、训练、保存和预测文本分类任务中的神经网络模型
#### 类初始化
__init__: 构造函数，初始化模型参数。
input_shape: 模型输入数据的形状，默认是 2000 维的向量。
num_classes: 分类任务中的类别数，默认为 5（对应 5 类分类）。
config: 传入的配置对象（如前面提到的 LrConfig），用于获取训练相关参数，如训练轮数、保存路径等。
EarlyStopping: 早停回调函数，用于在验证集损失不再下降时提前停止训练，防止过拟合。
ReduceLROnPlateau: 学习率调度器，当验证集损失不再改善时，自动降低学习率。
model: 调用 build_model() 方法创建的神经网络模型。
#### 构建模型
build_model(): 构建神经网络模型。
Input Layer: 输入层，形状为 input_shape。
Dense Layers: 全连接层，使用 ReLU 激活函数，有 512、256 和 128 个神经元。
BatchNormalization: 批归一化层，用于加速训练并减少过拟合。
Dropout Layers: Dropout 层，随机丢弃一定比例的神经元（70%），防止过拟合。
Output Layer: 输出层，使用 softmax 激活函数，进行多类别分类。
#### 模型编译
compile(): 编译模型。
optimizer: 使用 Adam 优化器，学习率为 0.01。
loss: 使用 SparseCategoricalCrossentropy 损失函数，适合多分类问题。
metrics: 评价指标为准确率（accuracy）。
#### 模型训练
fit(): 训练模型。
dataset: 训练数据集。
dataset_val: 验证数据集。
epochs: 训练轮数，根据配置类中的 num_epochs 进行设置。
callbacks: 包含早停和学习率调度回调函数。
#### 模型预测
predict(): 对输入数据进行预测，返回每个输入样本的预测类别（通过 argmax 获取最大概率的类别索引）
#### 模型保存
save(): 将训练好的模型保存到指定路径。
os.makedirs: 创建保存路径的目录，如果目录不存在，则创建它。
### main.py
使用了前面定义的 DataProcess 类来处理数据，用 Lr_Model 类构建并训练一个神经网络模型。它包括从数据预处理、模型训练、结果可视化到模型保存和预测输出的整个流程。
#### 配置与初始化
创建了 config 对象，用于获取数据路径、模型保存路径和其他配置信息。
初始化 DataProcess 对象，传入数据路径，准备数据预处理。
初始化 Lr_Model 对象，传入配置，准备构建和训练神经网络模型。
#### 数据预处理
调用 provide_data() 方法，读取数据并进行预处理。
x_train_svd、x_val_svd 和 x_test_svd 分别是训练集、验证集和测试集的特征矩阵。
y_train 和 y_val 是训练集和验证集的标签。
test 是测试集的数据。
#### 模型构建与编译
build_model(): 构建神经网络模型，添加输入层、隐藏层、输出层等。
summary(): 打印模型结构，展示各层的参数信息。
compile(): 编译模型，指定优化器（Adam）、损失函数（SparseCategoricalCrossentropy）和评价指标（准确率）
#### 模型训练
使用 fit() 方法训练模型。
get_dataset() 将数据转换为 TensorFlow 数据集，指定批量大小为 128。
history 保存了训练过程中的损失值和准确率等信息。
#### 结果可视化
使用 matplotlib 绘制训练和验证集的损失和准确率曲线，展示模型在不同训练轮次下的表现。
history_dict 中存储了每个 epoch 的损失值和准确率，用于绘制曲线。
####  预测与输出
对测试集进行预测，得到每个样本的类别标签。
将预测结果添加到测试集的 Sentiment 列中，并保存为 CSV 文件，用于提交结果。
## Softmax回归模型
### data_process.py
这段代码实现了一个文本分类的基础框架，包括文本数据的处理、词袋模型 (Bag of Words) 和 N-gram 模型的构建。代码分为两部分：文本处理的基类 BaseTextProcessor 及其两个子类 BagOfWords 和 Ngram，用于不同的文本特征提取方法。
#### 函数 split_train_test
功能：该函数将数据集按一定比例（test_rate）随机划分为训练集和测试集。
参数：
data: 输入的完整数据集。
test_rate: 测试集的比例（默认为 30%）。
maxitem: 限制处理的数据量（默认为 1000 条）。
输出：返回训练集和测试集。
#### 基类 BaseTextProcessor
功能：BaseTextProcessor 是一个处理文本数据的基类，定义了数据的基本操作和结构。
属性：
self.data: 原始数据，限制为 maxitem 条。
self.words: 存储词汇表的字典。
self.len: 词汇表的长度。
self.train 和 self.test: 训练集和测试集。
self.train_sentiment 和 self.test_sentiment: 训练集和测试集的标签（情感类别）。
self.train_matrix 和 self.test_matrix: 用于存储训练集和测试集的特征矩阵。
方法：
initialize_matrices: 初始化训练和测试矩阵，大小根据训练集和测试集的样本数量以及词汇表的长度来确定。
fill_matrix: 填充矩阵的方法，需要在子类中实现。
get_matrix: 获取训练集和测试集的特征矩阵，通过调用 fill_matrix 方法填充矩阵。
#### 子类 BagOfWords
功能：BagOfWords 类实现了词袋模型的特征提取。
get_words: 生成词汇表，遍历所有数据集（训练集和测试集），将文本中的单词（大写）加入词汇表，并记录其索引。
fill_matrix: 填充特征矩阵，如果某个单词在词汇表中，则在对应位置将值设为 1，实现词袋模型的表示方式。
#### 子类 Ngram
功能：Ngram 类实现了 N-gram 模型的特征提取。
__init__: 初始化 N-gram 模型，dimension 表示 N-gram 的维度（默认为 3）。
get_words: 生成 N-gram 词汇表，遍历数据中的所有 N-gram 组合，将其加入词汇表。
fill_matrix: 填充特征矩阵，如果某个 N-gram 组合在词汇表中，则在对应位置将值设为 1。
### main.py
这段代码是用于运行一个文本特征提取和比较的主程序。代码通过读取数据、提取特征并进行比较，展示了如何使用词袋模型（Bag of Words）和 N-gram 模型进行特征提取
#### 导入模块
numpy 和 random 用于数据处理和随机数生成。
csv 用于读取 TSV 格式的数据文件。
从 data_process 模块中导入 BagOfWords 和 Ngram 类，用于特征提取。
从 process_compare 模块中导入 alpha_gradient 函数，用于模型比较或实验。
#### 读取数据
打开并读取 train.tsv 文件。
使用制表符（\t）作为分隔符将数据读取到 tsv_data 中。
将数据转换为列表，并去掉第一行（通常是表头），保存到 data 变量中。
#### 设置参数
max_item 设置最大处理数据量为 1000。
使用 rd.seed(2024) 和 np.random.seed(2024) 设置随机种子，以确保结果可重复。
#### 特征提取
实例化 BagOfWords 类，设置最大数据量为 1000。
调用 get_words() 和 get_matrix() 方法来生成词汇表和特征矩阵。
实例化 Ngram 类，分别为二元（bigram）和三元（trigram）模型。
调用 get_words() 和 get_matrix() 方法来生成 N-gram 词汇表和特征矩阵。
### process_compare.py
这段代码用于训练并评估不同特征提取方法（词袋模型、二元和三元 N-gram）在不同学习率下的性能，最后通过绘图展示结果。
#### 导入模块
matplotlib.pyplot 用于绘图。
numpy 用于数值计算。
Softmax 从 softmax_regression 模块中导入，用于训练模型。
#### 训练模型
train_alphas 函数用于训练模型并计算训练集和测试集的正确率。
Softmax 模型使用不同的学习率 (alpha)、训练轮数 (total_times)、训练数据 (sample.train_matrix 和 sample.train_sentiment)、策略 (strategy) 和批次大小 (mini_size) 进行训练。
返回训练集和测试集的正确率 (r_train 和 r_test)。
#### 绘制学习率与准确率的关系
alpha_gradient 函数用于绘制不同特征提取方法（词袋模型、二元和三元 N-gram）在不同学习率下的训练集和测试集准确率。
定义了学习率列表 alphas。
#### 词袋模型特征提取效果
对词袋模型在不同学习率下进行训练和测试。
分别使用 shuffle、batch 和 mini-batch 三种策略。
记录每种策略下的训练集和测试集准确率。
#### 二元 N-gram 特征提取效果
#### 三元 N-gram 特征提取效果
#### 绘制图表
### softmax_regression.py
这段代码定义了一个用于多类分类的 Softmax 回归模型
#### 类初始化
sample_num: 样本数量。
features: 特征的维度。
categories: 类别的数量，默认为 5。
self.W: 权重矩阵，维度为 (features, categories)，用随机值初始化。
#### Softmax 计算
softmax_calculation: 计算一个向量的 softmax 值。为了提高数值稳定性，先减去向量中的最大值。
softmax_all: 对矩阵的每一行应用 softmax 函数。先减去每行的最大值，然后计算指数和归一化
#### One-hot 向量转换
#### 预测
#### 计算准确率
#### 回归训练
regression 方法实现了 Softmax 回归的训练，支持 mini-batch、shuffle 和 batch 三种策略。
Mini-batch: 从数据集中随机选择 mini-batch 样本进行训练。
Shuffle: 从数据集中随机选择一个样本进行训练。
Batch: 使用整个数据集进行训练。
训练过程通过梯度更新权重矩阵 self.W。
## 数据集
包括训练集和测试集
## 实验图片
## 报告

# 安装
运行项目需要 Python 3.x 以及以下库：
pip install numpy matplotlib scikit-learn tensorflow

# 数据准备
数据集应为 .tsv 格式，包含以下列：
ID
文本
标签
确保数据格式正确并将其放置在 data/ 目录中。

# 结果
结果以准确率的形式呈现，并通过图表可视化，比较不同模型和参数设置的性能。

主要观察结果：
Softmax 回归：在简单的文本分类任务中表现良好，尤其是在超参数优化之后。
神经网络：在更复杂的数据集上表现优于 Softmax 回归，因为它能够捕捉非线性关系。
3-gram 特征 通常比 2-gram 或词袋模型提供更高的准确率，因为它捕捉了更好的上下文信息。
参数调优 对两种模型都至关重要。合适的学习率和批次大小能显著提高性能。
