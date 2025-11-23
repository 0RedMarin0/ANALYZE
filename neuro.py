import numpy as np
import pandas as pd
import tensorflow as tf
import talib
import matplotlib
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler  # ← Изменил на StandardScaler
matplotlib.use('TkAgg')

VERSION = "2.5"  # ← Обновил версию
MIN = 10
TIMESTEP = 100
BATCH_SIZE = 32  # ← Увеличил для GPU
PREDICTION = -20  # ← Упростил прогноз
VOLUME_DATA = 100000
FILE_NAME = 'BD/SBER_10.csv'

EPOCH = 50  # ← Увеличил эпохи

MODEL_NAME = f"models/model_{MIN}min_step_{TIMESTEP}_pred_{abs(PREDICTION)}_{VOLUME_DATA}_{VERSION}.keras"


class NeuroBrain:
    def __init__(self):
        self.model = None
        self.feature_columns = [
            'open', 'high', 'low', 'close', 'volume',
            'BB_upper', 'BB_middle', 'BB_lower', 'SMA_20',
            'EMA_20', 'SMA_100', 'EMA_100', 'CCI', 'SAR',
            'RSI', 'MACD', 'MACD_signal', 'MACD_hist'
        ]

    def build_model(self, input_shape):
        """Модель со смещением и улучшенной архитектурой"""
        inputs = tf.keras.Input(shape=input_shape)

        # LSTM слои
        x = tf.keras.layers.LSTM(64, return_sequences=True)(inputs)
        x = tf.keras.layers.LSTM(32, return_sequences=False)(x)

        # Dense слои с bias (важно!)
        x = tf.keras.layers.Dense(32, activation='relu', use_bias=True)(x)
        x = tf.keras.layers.Dense(16, activation='relu', use_bias=True)(x)

        # Выход с bias
        outputs = tf.keras.layers.Dense(1, activation='linear', use_bias=True)(x)

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

    def data_create(self, data):
        """Создание фич с обработкой NaN"""
        data = data.copy()

        # Базовые фичи
        data['RSI'] = talib.RSI(data['close'], timeperiod=14)
        data['MACD'], data['MACD_signal'], data['MACD_hist'] = talib.MACD(
            data['close'], fastperiod=12, slowperiod=26, signalperiod=9)

        # Bollinger Bands
        data['BB_upper'], data['BB_middle'], data['BB_lower'] = talib.BBANDS(data['close'])

        # Moving averages
        data['SMA_20'] = talib.SMA(data['close'], timeperiod=20)
        data['EMA_20'] = talib.EMA(data['close'], timeperiod=20)
        data['SMA_100'] = talib.SMA(data['close'], timeperiod=100)
        data['EMA_100'] = talib.EMA(data['close'], timeperiod=100)

        # Дополнительные индикаторы
        data['CCI'] = talib.CCI(data['high'], data['low'], data['close'])
        data['SAR'] = talib.SAR(data['high'], data['low'])

        # Заполняем NaN значения
        data = data.fillna(method='bfill').fillna(method='ffill')

        return data

    def create_sequences(self, X, y, time_steps=100):
        """Создание последовательностей для временных рядов"""
        Xs, ys = [], []
        for i in range(time_steps, len(X)):
            Xs.append(X[i - time_steps:i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)

    def simple_directional_loss(self, y_true, y_pred):
        """Упрощенная функция потерь с учетом направления"""
        # Основная MSE loss
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))

        # Вычисление правильных направлений
        true_changes = y_true[1:] - y_true[:-1]
        pred_changes = y_pred[1:] - y_pred[:-1]

        # Directional accuracy
        directional_accuracy = tf.reduce_mean(
            tf.cast(tf.sign(true_changes) == tf.sign(pred_changes), tf.float32)
        )

        # Комбинированная loss: 90% MSE + 10% directional
        return 0.9 * mse_loss + 0.1 * (1.0 - directional_accuracy)

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
    df = neuro.data_create(data)

    df["target"] = (df["close"].shift(-20) / df["close"]) - 1

    df = df.dropna()

    # Подготовка фич и target
    features = df[neuro.feature_columns]
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
    print(X_train.shape[1], X_train.shape[2])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', 'mse']
    )
    model.summary()


    # Обучение
    print(f"\n=== НАЧАЛО ОБУЧЕНИЯ ===")
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
        model.save(MODEL_NAME)

    # Сохранение модели
    model.save(MODEL_NAME)
    print(f"Модель сохранена как: {MODEL_NAME}")
