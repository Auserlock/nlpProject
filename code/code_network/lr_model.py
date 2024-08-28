from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import layers, models, losses, metrics
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
import numpy as np
import os


class Lr_Model:
    def __init__(self, config, input_shape=(2000,), num_classes=5):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.config = config
        # 定义早停和学习率调度器
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        self.lr_reduction = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
        self.model = self.build_model()

    # 构建模型
    def build_model(self):
        model = models.Sequential([
            layers.Input(shape=self.input_shape),  # 输入形状,与降维后维度有关
            layers.Dense(512, activation='relu'),
            BatchNormalization(),
            layers.Dropout(0.7),
            layers.Dense(256, activation='relu'),
            BatchNormalization(),
            layers.Dropout(0.7),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.num_classes, activation='softmax')  # 使用softmax进行多类别分类
        ])
        return model

    def summary(self):
        print(self.model.summary())

    def compile(self, learning_rate=0.01):  # 学习率
        self.model.compile(optimizer=Adam(learning_rate=learning_rate),
                           loss=losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    def fit(self, dataset, dataset_val):
        history = self.model.fit(dataset,
                                 validation_data=dataset_val,
                                 epochs=self.config.num_epochs,  # 训练轮数
                                 callbacks=[self.early_stopping, self.lr_reduction])
        return history

    def predict(self, data):
        return np.argmax(self.model.predict(data), axis=1)

    def save(self):
        os.makedirs(os.path.dirname(self.config.model_save_path), exist_ok=True)
        self.model.save(self.config.model_save_path)
