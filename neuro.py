import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import talib

# Загрузка данных
df = pd.read_csv('BD/SBER_10.csv').head(110000)

TIME_STEPS = 100

# Добавляем выбранные индикаторы
df['RSI'] = talib.RSI(df['close'])
df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['close'])
df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['close'])
df['SMA_20'] = talib.SMA(df['close'], timeperiod=20)
df['EMA_20'] = talib.EMA(df['close'], timeperiod=20)
df['CCI'] = talib.CCI(df['high'], df['low'], df['close'])
df['SAR'] = talib.SAR(df['high'], df['low'])
df['ADX'] = talib.ADX(df['high'], df['low'], df['close'])
df['PLUS_DI'] = talib.PLUS_DI(df['high'], df['low'], df['close'])
df['MINUS_DI'] = talib.MINUS_DI(df['high'], df['low'], df['close'])


df['target_close'] = df['close'].shift(-5)

df = df.dropna()

feature_columns = ['open', 'high', 'low', 'close', 'volume',
                   'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
                   'BB_upper', 'BB_middle', 'BB_lower',
                   'SMA_20', 'EMA_20', 'CCI', 'SAR', 'ADX',
                   'PLUS_DI', 'MINUS_DI']

features = df[feature_columns]
target = df['target_close']

print(f"Размерность признаков: {features.shape}")
print(f"Размерность целевой: {target.shape}")

# Создаем скейлеры для признаков и целевой переменной
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

# Нормализуем признаки
features_scaled = feature_scaler.fit_transform(features)

# Нормализуем целевую переменную (важно для LSTM)
target_scaled = target_scaler.fit_transform(target.values.reshape(-1, 1))

# Рассчитываем индексы для разделения
total_samples = len(features_scaled)
train_size = int(0.7 * total_samples)
val_size = int(0.15 * total_samples)

# Разделяем с сохранением временного порядка
X_train = features_scaled[:train_size]
y_train = target_scaled[:train_size]

X_val = features_scaled[train_size:train_size + val_size]
y_val = target_scaled[train_size:train_size + val_size]

X_test = features_scaled[train_size + val_size:]
y_test = target_scaled[train_size + val_size:]

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

def create_sequences(X, y, time_steps=100):
    Xs, ys = [], []
    for i in range(time_steps, len(X)):
        Xs.append(X[i-time_steps:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

# Создаем последовательности для train/val/test
X_train_seq, y_train_seq = create_sequences(X_train, y_train, TIME_STEPS)
X_val_seq, y_val_seq = create_sequences(X_val, y_val, TIME_STEPS)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, TIME_STEPS)

print(f"Train sequences: {X_train_seq.shape}, {y_train_seq.shape}")

model = tf.keras.Sequential()

# Увеличенная архитектура для больших данных
model.add(tf.keras.layers.LSTM(256, return_sequences=True, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.LSTM(128, return_sequences=True))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.LSTM(64, return_sequences=True))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.LSTM(32))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.2))

# Дополнительные полносвязные слои
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dropout(0.1))

model.add(tf.keras.layers.Dense(1))  # Выходной слой

model.summary()


model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mae', 'mse']
)

print("Проверка размерностей:")
print(f"X_train_seq: {X_train_seq.shape}")
print(f"y_train_seq: {y_train_seq.shape}")
print(f"X_val_seq: {X_val_seq.shape}")
print(f"y_val_seq: {y_val_seq.shape}")

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Callbacks для улучшения обучения
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=0.0001,
        verbose=1
    )
]

history = model.fit(
    X_train_seq, y_train_seq,
    batch_size=32,
    epochs=50,
    validation_data=(X_val_seq, y_val_seq),
    callbacks=callbacks,
    verbose=1
)

# Сохраняем архитектуру и веса модели
model.save('models/my_lstm_model_10_minus5_110000.keras')

# Альтернативно можно сохранить так:
# model.save('my_lstm_model', save_format='keras')

print("Модель сохранена")

# import os
# os.system("shutdown /s /t 5")  # Выключение через 5 секунд