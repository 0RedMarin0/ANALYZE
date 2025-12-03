import os

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
from keras.src.layers import Dropout
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler  # ← Изменил на StandardScaler
from tensorflow.python.keras.legacy_tf_layers.core import dropout

matplotlib.use('TkAgg')

VERSION = "3.0.2"  # ← Обновил версию
MIN = 10
TIMESTEP = 30
BATCH_SIZE = 16
PREDICTION = -5
VOLUME_DATA = 50000
FILE_NAME = 'BD/SBER_10.csv'

EPOCH = 50  # ← Увеличил эпохи

MODEL_NAME = f"MOEX_model_{MIN}min_step_{TIMESTEP}_pred_{abs(PREDICTION)}_{VOLUME_DATA}_e{EPOCH}"


class RiskAwareLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.3, name='risk_aware_loss'):
        super().__init__(name=name)
        self.alpha = alpha  # коэффициент "рискованности"
        self.mse = tf.keras.losses.MeanSquaredError()

    def call(self, y_true, y_pred):
        mse_loss = self.mse(y_true, y_pred)

        # Поощряем большие отклонения от среднего
        pred_mean = tf.reduce_mean(y_pred)
        risk_bonus = -self.alpha * tf.reduce_mean(tf.abs(y_pred - pred_mean))

        # Комбинируем: меньше MSE + больше дисперсии предсказаний
        return mse_loss + risk_bonus


class NeuroBrain:
    def __init__(self):
        self.model = None


    def build_model(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape)

        conv3 = tf.keras.layers.Conv1D(32, 3, padding='same', activation='relu')(inputs)
        conv5 = tf.keras.layers.Conv1D(32, 5, padding='same', activation='relu')(inputs)
        conv7 = tf.keras.layers.Conv1D(32, 7, padding='same', activation='relu')(inputs)

        x = tf.keras.layers.Concatenate()([conv3, conv5, conv7])
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)

        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, return_sequences=False, dropout=0.4)
        )(x)

        x = tf.keras.layers.Dense(64, activation='linear')(x)
        x = tf.keras.layers.Dropout(0.2)(x)

        x = tf.keras.layers.Dense(35, activation='linear')(x)
        x = tf.keras.layers.Dropout(0.1)(x)

        x = tf.keras.layers.Dense(10, activation='linear')(x)
        x = tf.keras.layers.Dropout(0.1)(x)

        outputs = tf.keras.layers.Dense(1, activation='linear')(x)

        self.model = tf.keras.Model(inputs, outputs)
        return self.model

    def callbacks(self):
        """
        Универсальные callback'ы работают для любой задачи
        """
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',  # ✅ Всегда доступна
                patience=15,  # ✅ Увеличил терпение
                restore_best_weights=True,
                mode='min',  # ✅ Минимизируем потерю
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,  # ✅ Больше patience перед уменьшением LR
                min_lr=0.000001,  # ✅ Еще меньше минимальный LR
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.h5',  # ✅ .h5 более надежно
                monitor='val_loss',
                save_best_only=True,
                mode='min',  # ✅ Сохраняем при минимальной потере
                verbose=1
            )
        ]
        return callbacks

    def create_sequences(self, X, y, time_steps=100):
        """Создание последовательностей для временных рядов"""
        Xs, ys = [], []
        for i in range(time_steps, len(X)):
            Xs.append(X[i - time_steps:i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)

    def plot_predictions(self, y_true, y_pred, title="Прогноз vs Реальность"):
        """Визуализация прогнозов"""
        plt.figure(figsize=(15, 8))

        # Берем только первые 500 точек для наглядности
        n_points = min(500, len(y_true))
        indices = range(n_points)

        plt.plot(indices, y_true[:n_points], label='Реальные значения', alpha=0.7, linewidth=2)
        plt.plot(indices, y_pred[:n_points], label='Прогноз', alpha=0.7, linewidth=1.5)

        plt.title(title, fontsize=14)
        plt.xlabel('Время')
        plt.ylabel('Целевая переменная')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('predictions_plot.png', dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    # Загрузка данных
    data = pd.read_csv(FILE_NAME).head(VOLUME_DATA)
    neuro = NeuroBrain()

    # Создание фич
    import table
    df0 = table.DataCreate(data)
    df = df0.table
    df["target"] = (df["MACD"].shift(PREDICTION) / df["MACD"]) - 1
    df = df.dropna()

    # Подготовка фич и target
    features = df[df0.list_sign]
    target = df['target']

    # Масштабирование
    feature_scaler = StandardScaler()
    features_scaled = feature_scaler.fit_transform(features)

    # Создание последовательностей
    X_seq, y_seq = neuro.create_sequences(features, target.values, TIMESTEP)

    print(f"\n=== ИНФОРМАЦИЯ О ПОСЛЕДОВАТЕЛЬНОСТЯХ ===")
    print(f"X_seq shape: {X_seq.shape}")
    print(f"y_seq shape: {y_seq.shape}")

    # Разделение на train/val/test с учетом временных рядов
    train_size = int(0.7 * len(X_seq))
    val_size = int(0.15 * len(X_seq))

    X_train = X_seq[:train_size]
    y_train = y_seq[:train_size]

    X_val = X_seq[train_size:train_size + val_size]
    y_val = y_seq[train_size:train_size + val_size]

    X_test = X_seq[train_size + val_size:]
    y_test = y_seq[train_size + val_size:]


    # target_scaler = StandardScaler()
    y_train_scaled = y_train
    y_val_scaled = y_val

    model = neuro.build_model((X_train.shape[1], X_train.shape[2]))

    target_mean = np.mean(target.values)
    target_centered = target.values - target_mean

    print(f"Было среднее: {target_mean:.6f}")
    print(f"Стало среднее: {np.mean(target_centered):.6f}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=['mae', 'mse']
    )
    model.summary()

    # В вашем основном коде ДО обучения добавьте:
    print("=== ПРОВЕРКА СВЯЗИ X-y ===")
    print(f"Пример 1: X[0] последние значения: {features_scaled[95:100, :3]}...")
    print(f"Соответствующий y: {target.values[100]:.4f}")

    import joblib
    # Обучение
    print(f"\n=== НАЧАЛО ОБУЧЕНИЯ ===")

    try:
        os.mkdir(f'models/{MODEL_NAME}')
    except FileExistsError:
        print("have direct")
    try:
        # Основное обучение (строка ~110):
        history = model.fit(
            X_train, y_train_scaled,  # ← y_train_scaled
            batch_size=BATCH_SIZE, epochs=EPOCH,
            validation_data=(X_val, y_val_scaled),  # ← y_val_scaled
            callbacks=neuro.callbacks(), verbose=1
        )

    except KeyboardInterrupt:
        print("Обучение прервано пользователем")
        model.save(f"models/{MODEL_NAME}/model_{VERSION}.keras")

        # joblib.dump(save_data, f'models/{MODEL_NAME}/model_complete_{VERSION}.pkl')
        print("✅ Модель и все настройки сохранены!")
        print(MODEL_NAME)

    # Сохранение модели
    model.save(f"models/{MODEL_NAME}/model_{VERSION}.keras")

    print("✅ Модель и все настройки сохранены!")
    print(MODEL_NAME)
