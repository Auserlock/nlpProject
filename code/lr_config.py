import os

pwd_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


class LrConfig(object):
    #  训练模型用到的路径
    dataset_path = os.path.join(pwd_path + r'/data' + r"/train.tsv")  # linux 文件路径, windows可能需要更改
    testdata_path = os.path.join(pwd_path + r'/data' + r"/test.tsv")
    model_save_path = os.path.join(pwd_path + r'/model' + r"/classification_model.keras")
    predict_save_path = os.path.join(pwd_path + r'/data' + r"/submission.csv")

    #  变量
    num_epochs = 20
