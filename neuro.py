# === Модель ===
# inputs = tf.keras.Input(shape=(X_train_seq.shape[1], X_train_seq.shape[2]))
#
# x = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='causal')(inputs)
# x = tf.keras.layers.Conv1D(64, 5, activation='relu', padding='causal')(x)
# x = tf.keras.layers.BatchNormalization()(x)
# x = tf.keras.layers.Dropout(0.2)(x)
#
# x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True, recurrent_dropout=0.2))(x)
# x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True, recurrent_dropout=0.2))(x)
#
# # === Attention ===
# attn = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)
# attn_out = attn(x, x)
# attn_out = tf.keras.layers.Dense(128, activation='relu')(attn_out)
# x = tf.keras.layers.Add()([x, attn_out])
# x = tf.keras.layers.LayerNormalization()(x)
#
#
# x = tf.keras.layers.GlobalAveragePooling1D()(x)
# x = tf.keras.layers.Dense(128, activation='relu')(x)
# x = tf.keras.layers.Dropout(0.3)(x)
# x = tf.keras.layers.Dense(64, activation='relu')(x)
# outputs = tf.keras.layers.Dense(1, activation='linear')(x)
#
# model = tf.keras.Model(inputs, outputs)
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import talib
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

VERSION = "1.4"
MIN = 10
TIMESTEP = 100
BATCH_SIZE = 16
PREDICTION = -5
VOLUME_DATA = 100000
FILE_NAME = 'BD/SBER_10.csv'

EPOCH = 3

MODEL_NAME = f"models/model_{MIN}min_step_{TIMESTEP}_pred__{abs(PREDICTION)}__{VOLUME_DATA}_{VERSION}.keras"
HISTORY_SAVE_PATH = f'training_history_{MODEL_NAME}.pkl'
print(MODEL_NAME)

data = pd.read_csv(FILE_NAME).iloc[:100000]


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
        # === Модель ===
        inputs = tf.keras.Input(shape=(input_shape))

        x = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='causal')(inputs)
        x = tf.keras.layers.Conv1D(64, 5, activation='relu', padding='causal')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)

        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True, recurrent_dropout=0.2))(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True, recurrent_dropout=0.2))(x)

        # === Attention ===
        attn = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)
        attn_out = attn(x, x)
        attn_out = tf.keras.layers.Dense(128, activation='relu')(attn_out)
        x = tf.keras.layers.Add()([x, attn_out])
        x = tf.keras.layers.LayerNormalization()(x)


        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        outputs = tf.keras.layers.Dense(1, activation='linear')(x)

        self.model = tf.keras.Model(inputs, outputs)
        return self.model

    def callbacks(self):
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',  # Следим за accuracy а не loss
                patience=10,
                restore_best_weights=True,
                mode='max',
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.keras',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        ]
        return callbacks

    def data_create(self, data):
        data['RSI'] = talib.RSI(data['close'], timeperiod=14)
        data['MACD'], data['MACD_signal'], data['MACD_hist'] = talib.MACD(
            data['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        data['BB_upper'], data['BB_middle'], data['BB_lower'] = talib.BBANDS(data['close'])
        data['SMA_20'] = talib.SMA(data['close'], timeperiod=20)
        data['EMA_20'] = talib.EMA(data['close'], timeperiod=20)
        data['SMA_100'] = talib.SMA(data['close'], timeperiod=100)
        data['EMA_100'] = talib.EMA(data['close'], timeperiod=100)
        data['CCI'] = talib.CCI(data['high'], data['low'], data['close'])
        data['SAR'] = talib.SAR(data['high'], data['low'])
        return data

    def create_sequences(self, X, y, time_steps=100):
        """Создание последовательностей для LSTM"""
        Xs, ys = [], []
        for i in range(time_steps, len(X)):
            Xs.append(X[i - time_steps:i])
            ys.append(y[i])
        return np.array(Xs), np.array(ys)

    def combined_price_loss(self, y_true, y_pred):
        """
        Комбинированная функция потерь для прогнозирования цены
        """
        # 1. Основная MSE loss для точности прогноза
        mse_loss = tf.keras.losses.mse(y_true, y_pred)

        # 2. Directional loss с плавным переходом
        # Используем разницу знаков с плавной функцией
        true_sign = tf.sign(y_true[1:] - y_true[:-1])
        pred_sign = tf.sign(y_pred[1:] - y_pred[:-1])

        # Плавный directional penalty (меньше резких скачков)
        directional_penalty = tf.where(
            true_sign != pred_sign,
            2.0,  # Умеренный штраф
            1.0
        )

        # 3. Volatility-adjusted loss (учет волатильности)
        price_range = tf.reduce_max(y_true) - tf.reduce_min(y_true)
        volatility_weight = tf.where(
            price_range > 0,
            1.0 + (tf.math.reduce_std(y_true) / (price_range + 1e-7)),
            1.0
        )

        # Комбинируем все компоненты
        directional_mse = mse_loss * directional_penalty * volatility_weight

        return tf.reduce_mean(directional_mse)

    def plot_training_history(self, history):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        ax1.plot(history.history['loss'], label='Train Loss')
        ax1.plot(history.history['val_loss'], label='Val Loss')
        ax1.set_title('Model Loss')
        ax1.set_ylabel('Loss')
        ax1.set_xlabel('Epoch')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax2.plot(history.history['accuracy'], label='Train Accuracy')
        ax2.plot(history.history['val_accuracy'], label='Val Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_ylabel('Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    data = pd.read_csv(FILE_NAME).tail(VOLUME_DATA)
    neuro = NeuroBrain()

    df = neuro.data_create(data)
    # df["change"] = (df["close"].shift(1) / df["close"]) - 1
    df["target"] = ((df["close"].shift(-1) / df["close"]) - 1) + \
                       ((df["close"].shift(-2) / df["close"].shift(-1)) - 1) + \
                       ((df["close"].shift(-3) / df["close"].shift(-2)) - 1) + \
                       ((df["close"].shift(-4) / df["close"].shift(-3)) - 1) + \
                       ((df["close"].shift(-5) / df["close"].shift(-4)) - 1)
    # df['target'] = df['close'].shift(PREDICTION)
    df = df.dropna()

    features = df[neuro.feature_columns]
    target = df['target']

    feature_scaler = MinMaxScaler()
    features_scaled = feature_scaler.fit_transform(features)
    X_seq, y_seq = neuro.create_sequences(features_scaled, target.values, TIMESTEP)

    X_temp, X_test, y_temp, y_test = train_test_split(X_seq, y_seq, test_size=0.15, random_state=42,
                                                      shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15 / (1 - 0.15),
                                                      random_state=42, shuffle=False)

    model = neuro.build_model((X_train.shape[1], X_train.shape[2]))
    # model.compile(
    #     optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    #     loss=directional_loss,
    #     metrics=['mae']
    # )
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001,  # Можно увеличить благодаря BatchNorm
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        ),
        loss=tf.keras.losses.Huber(),  # Лучше чем MSE для финансовых данных
        metrics=['mae', 'mse', 'accuracy']
    )


    check = 0
    try:
        # with tf.device('/GPU:0'):
        history = model.fit(
            X_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCH,
            validation_data=(X_val, y_val),
            callbacks=neuro.callbacks(),
            verbose=1
        )
    except KeyboardInterrupt:
        model.save(MODEL_NAME)
        check += 1

    if check == 0:
        model.save(MODEL_NAME)

    # test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    # neuro.plot_training_history(history)

