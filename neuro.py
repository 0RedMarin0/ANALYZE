import os

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler  # ← Изменил на StandardScaler
matplotlib.use('TkAgg')

VERSION = "2.0.3"  # ← Обновил версию
MIN = 10
TIMESTEP = 100
BATCH_SIZE = 32
PREDICTION = -10
VOLUME_DATA = 40000
FILE_NAME = 'BD/SBER_10.csv'

EPOCH = 50  # ← Увеличил эпохи

MODEL_NAME = f"MOEX_model_{MIN}min_step_{TIMESTEP}_pred_{abs(PREDICTION)}_{VOLUME_DATA}_e{EPOCH}"


class NeuroBrain:
    def __init__(self):
        self.model = None


    def build_model(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape)

        # УСИЛЕННЫЕ LSTM СЛОИ
        x = tf.keras.layers.LSTM(128, return_sequences=True,
                               dropout=0.2, recurrent_dropout=0.1)(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LSTM(96, return_sequences=True,
                               dropout=0.15, recurrent_dropout=0.1)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LSTM(64, return_sequences=False,
                               dropout=0.1)(x)

        # УСИЛЕННЫЕ DENSE СЛОИ
        x = tf.keras.layers.Dense(96, activation='tanh')(x)
        x = tf.keras.layers.Dropout(0.25)(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Dense(48, activation='tanh')(x)
        x = tf.keras.layers.Dropout(0.2)(x)

        x = tf.keras.layers.Dense(24, activation='tanh')(x)
        x = tf.keras.layers.Dropout(0.15)(x)

        # Выходной слой
        outputs = tf.keras.layers.Dense(1, activation='tanh')(x)

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
    df["target"] = df["RSI"].shift(PREDICTION)
    df = df.dropna()

    # Подготовка фич и target
    features = df[df0.list_sign]
    target = df['target']

    # Масштабирование
    feature_scaler = StandardScaler()
    features_scaled = feature_scaler.fit_transform(features)

    # Создание последовательностей
    X_seq, y_seq = neuro.create_sequences(features_scaled, target.values, TIMESTEP)

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


    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = target_scaler.transform(y_val.reshape(-1, 1)).flatten()

    model = neuro.build_model((X_train.shape[1], X_train.shape[2]))

    target_mean = np.mean(target.values)
    target_centered = target.values - target_mean

    print(f"Было среднее: {target_mean:.6f}")
    print(f"Стало среднее: {np.mean(target_centered):.6f}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0008),
        loss=tf.keras.losses.Huber(),
        metrics=['mae', 'mse']
    )
    model.summary()

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







    # Сохранение модели
    model.save(f"models/{MODEL_NAME}/model_{VERSION}.keras")

    # save_data = {
    #     'feature_scaler': feature_scaler,
    #     'target_scaler': target_scaler,
    #     'feature_columns': neuro.feature_columns,
    #     'timestep': TIMESTEP,
    # }
    #
    # joblib.dump(save_data, f'models/{MODEL_NAME}/model_complete_{VERSION}.pkl')
    print("✅ Модель и все настройки сохранены!")
    #
    # train_predictions = model.predict(X_train[:1000], verbose=0).flatten()
    # print(f"Среднее прогнозов модели: {np.mean(train_predictions):.6f}")
    # print(f"Стандартное отклонение прогнозов: {np.std(train_predictions):.6f}")
    #
    # output_layer = model.layers[-1]
    # weights, biases = output_layer.get_weights()
    # print(f"Bias выходного слоя: {biases[0]:.6f}")
    #
    # print("=== ДИАГНОСТИКА МОДЕЛИ ===")
    # print(f"1. Среднее таргета: {np.mean(y_train):.6f}")
    # print(f"2. Среднее прогнозов: {np.mean(train_predictions):.6f}")
    # print(f"3. Bias выходного слоя: {biases[0]:.6f}")
    # print(f"4. Loss на трейне: {history.history['loss'][-1]:.6f}")
    # print(f"5. Loss на валидации: {history.history['val_loss'][-1]:.6f}")
