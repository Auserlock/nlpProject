from data_process import DataProcess
from lr_model import Lr_Model
from lr_config import LrConfig
from getaccuracy import AccuracyEvaluator
import matplotlib.pyplot as plt

if __name__ == '__main__':
    config = LrConfig()
    DataProcess = DataProcess(config.dataset_path, config.testdata_path)
    Lr_Model = Lr_Model(config)
    x_train_svd, x_val_svd, y_train, y_val, x_test_svd, test = DataProcess.provide_data()
    Lr_Model.build_model()
    Lr_Model.summary()
    Lr_Model.compile()
    history = Lr_Model.fit(DataProcess.get_dataset(x_train_svd, y_train, 128, 100000),
                           DataProcess.get_dataset(x_val_svd, y_val, 128, 10000))

    history_dict = history.history
    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.plot(epochs, loss, 'bo', label='Training loss')  # 'bo' 表示蓝色的点
    plt.plot(epochs, val_loss, 'b-', label='Validation loss')  # 'b-' 表示蓝色的线
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.plot(epochs, acc, 'bo', label='Training accuracy')  # 'bo' 表示蓝色的点
    plt.plot(epochs, val_acc, 'b-', label='Validation accuracy')  # 'b-' 表示蓝色的线
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()

    Lr_Model.save()
    predict = Lr_Model.predict(x_test_svd)
    test['Sentiment'] = predict
    test.to_csv(config.predict_save_path, index=False, columns=['PhraseId', 'Sentiment'])

    # 实例化 AccuracyEvaluator 类
    evaluator = AccuracyEvaluator(config.predict_save_path, config.sampleSubmission_path)

    # 计算准确率
    accuracy = evaluator.calculate_accuracy()
    print(f"测试集的准确率为: {accuracy:.4f}")