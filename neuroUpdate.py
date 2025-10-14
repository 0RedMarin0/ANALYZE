import numpy as np
import pandas as pd
import tensorflow as tf
import talib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# === ПАРАМЕТРЫ ===
FILE = 'BDcrypt/CRYPTO_ETHUSDT_15m_YEAR.csv'  # 👉 Новый актив
MODEL_NAME = 'models/crypto_model_15_delta_FINAL.keras'
NEW_MODEL_NAME = 'models/crypto_model_15_delta_FINAL_CONT.keras'  # Новый файл после дообучения
TIME_STEPS = 350

# === ЗАГРУЖАЕМ СУЩЕСТВУЮЩУЮ МОДЕЛЬ ===
print("🔁 Загружаем обученную модель...")
model = tf.keras.models.load_model(MODEL_NAME)
print("✅ Модель загружена успешно!")

# === ЗАГРУЖАЕМ НОВЫЙ ДАТАСЕТ ===
df = pd.read_csv(FILE).tail(5000)

# === ИНДИКАТОРЫ ===
df['RSI'] = talib.RSI(df['close'])
df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['close'])
df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['close'])
df['SMA_20'] = talib.SMA(df['close'], 20)
df['EMA_20'] = talib.EMA(df['close'], 20)
df['SMA_50'] = talib.SMA(df['close'], 50)
df['SMA_100'] = talib.SMA(df['close'], 100)
df['EMA_100'] = talib.EMA(df['close'], 100)
df['SMA_200'] = talib.SMA(df['close'], 200)
df['EMA_200'] = talib.EMA(df['close'], 200)
df['CCI'] = talib.CCI(df['high'], df['low'], df['close'])
df['ADX'] = talib.ADX(df['high'], df['low'], df['close'])
df['volatility'] = talib.ATR(df['high'], df['low'], df['close'], 14)

# === КОНТЕКСТ ===
df['trend_strength'] = df['SMA_50'] / df['SMA_200'] - 1
df['momentum'] = df['close'] / df['close'].shift(10) - 1
df['vol_ratio'] = df['volume'] / df['volume'].rolling(50).mean()
df['price_pos'] = (df['close'] - df['low'].rolling(100).min()) / (
    df['high'].rolling(100).max() - df['low'].rolling(100).min()
)

df['slope'] = df['close'].diff(5)
df['slope'] = df['slope'] / df['close'].shift(5)

df['candle_body'] = df['close'] - df['open']
df['upper_shadow'] = df['high'] - df[['close', 'open']].max(axis=1)
df['lower_shadow'] = df[['close', 'open']].min(axis=1) - df['low']

df['returns'] = df['close'].pct_change()
df['log_return'] = np.log(df['close'] / df['close'].shift(1))

# === ЦЕЛЕВАЯ ПЕРЕМЕННАЯ ===
df['target_close'] = np.log(df['close'].shift(-20) / df['close'])
df = df.dropna()

# === ПРИЗНАКИ ===
feature_columns = [
    'open', 'high', 'low', 'close', 'volume',
    'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
    'BB_upper', 'BB_middle', 'BB_lower',
    'SMA_20', 'EMA_20', 'SMA_100', 'EMA_100', 'SMA_200', 'EMA_200', 'SMA_50',
    'CCI', 'ADX', 'volatility',
    'trend_strength', 'momentum', 'vol_ratio', 'price_pos', 'slope',
    'candle_body', 'upper_shadow', 'lower_shadow', 'returns', 'log_return'
]


features = df[feature_columns]
target = df['target_close']

# === СКЕЙЛИНГ ===
feature_scaler = StandardScaler()
features_scaled = feature_scaler.fit_transform(features)
target_values = target.values.reshape(-1, 1)

# === РАЗДЕЛЕНИЕ ===
train_size = int(0.85 * len(features_scaled))
X_train = features_scaled[:train_size]
y_train = target_values[:train_size]
X_val = features_scaled[train_size:]
y_val = target_values[train_size:]

def create_sequences(X, y, time_steps=100):
    Xs, ys = [], []
    for i in range(time_steps, len(X)):
        Xs.append(X[i - time_steps:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

X_train_seq, y_train_seq = create_sequences(X_train, y_train, TIME_STEPS)
X_val_seq, y_val_seq = create_sequences(X_val, y_val, TIME_STEPS)

print(f"📈 Данные для дообучения: {X_train_seq.shape}")

# === CALLBACKS ===
callbacks = [
    EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1)
]

# === ПРОДОЛЖАЕМ ОБУЧЕНИЕ ===
print("🚀 Продолжаем обучение на новом активе...")
history = model.fit(
    X_train_seq, y_train_seq,
    epochs=10,                # 👈 Можно поставить 5–10 для мягкой подгонки
    batch_size=128,
    validation_data=(X_val_seq, y_val_seq),
    callbacks=callbacks,
    verbose=1
)

# === СОХРАНЯЕМ ОБНОВЛЁННУЮ МОДЕЛЬ ===
model.save(NEW_MODEL_NAME)
print(f"✅ Модель дообучена и сохранена как {NEW_MODEL_NAME}")

import joblib
feature_scaler = joblib.load("scalers/feature_scaler.pkl")

