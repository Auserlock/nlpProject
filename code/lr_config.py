# 模型配置
import os

pwd_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

class LrConfig(object):
    #  训练模型用到的路径
    data_path = os.path.join("../data/train.tsv")

    lr_save_dir = os.path.join(pwd_path + '\modl' + "\checkpoints")
    lr_save_path = os.path.join(lr_save_dir, 'best_validation')
    #  变量
    num_epochs = 100  # 总迭代轮次
    num_classes = 10  # 类别数
    print_per_batch = 10  # 每多少轮输出一次结果
