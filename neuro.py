from keras.src.saving import register_keras_serializable
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import tensorflow as tf
import talib
from tensorflow.keras.models import load_model



FILE = 'BDcrypt/CRYPTO_BTCUSDT_15m_YEAR.csv'
TIME_STEPS = 350
MODEL_NAME = 'models/crypto_model_15_delta_FINAL.keras'

df = pd.read_csv(FILE).tail(20000)

# === Индикаторы ===
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

# === Контекстные признаки ===
df['trend_strength'] = df['SMA_50'] / df['SMA_200'] - 1
df['momentum'] = df['close'] / df['close'].shift(10) - 1
df['vol_ratio'] = df['volume'] / df['volume'].rolling(50).mean()
df['price_pos'] = (df['close'] - df['low'].rolling(100).min()) / (df['high'].rolling(100).max() - df['low'].rolling(100).min())

df['slope'] = df['close'].diff(5)
df['slope'] = df['slope'] / df['close'].shift(5)

df['candle_body'] = df['close'] - df['open']
df['upper_shadow'] = df['high'] - df[['close','open']].max(axis=1)
df['lower_shadow'] = df[['close','open']].min(axis=1) - df['low']

df['returns'] = df['close'].pct_change()
df['log_return'] = np.log(df['close'] / df['close'].shift(1))

# === Целевая переменная: лог-доходность через 5 свечей ===
df['target_close'] = np.log(df['close'].shift(-6) / df['close']) * 2
df = df.dropna()

# === Признаки ===
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

# === Масштабирование: StandardScaler вместо MinMax ===
from sklearn.preprocessing import StandardScaler
feature_scaler = StandardScaler()
features_scaled = feature_scaler.fit_transform(features)

# === Без масштабирования целевой — она уже в нормальном масштабе (лог-доходность) ===
target_values = target.values.reshape(-1, 1)

# === Разделение ===
train_size = int(0.7 * len(features_scaled))
val_size = int(0.15 * len(features_scaled))

X_train = features_scaled[:train_size]
y_train = target_values[:train_size]
X_val = features_scaled[train_size:train_size + val_size]
y_val = target_values[train_size:train_size + val_size]
X_test = features_scaled[train_size + val_size:]
y_test = target_values[train_size + val_size:]

def create_sequences(X, y, time_steps=100):
    Xs, ys = [], []
    for i in range(time_steps, len(X)):
        Xs.append(X[i - time_steps:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

X_train_seq, y_train_seq = create_sequences(X_train, y_train, TIME_STEPS)
X_val_seq, y_val_seq = create_sequences(X_val, y_val, TIME_STEPS)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, TIME_STEPS)

# === Модель ===
inputs = tf.keras.Input(shape=(X_train_seq.shape[1], X_train_seq.shape[2]))

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

model = tf.keras.Model(inputs, outputs)


@register_keras_serializable(package="Custom", name="directional_loss")
def directional_loss(y_true, y_pred):
    diff = y_pred - y_true
    sign_penalty = tf.where(tf.sign(y_pred) != tf.sign(y_true), 2.0, 1.0)
    return tf.reduce_mean(tf.abs(diff) * sign_penalty)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    loss=directional_loss,
    metrics=['mae']
)


from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-5, verbose=1)
]

try:
    history = model.fit(
        X_train_seq, y_train_seq,
        epochs=10,
        batch_size=128,
        validation_data=(X_val_seq, y_val_seq),
        callbacks=callbacks,
        verbose=1
    )
except KeyboardInterrupt:
    print("\n⛔ Обучение остановлено вручную.")
    model.save(MODEL_NAME)
    print(f"✅ Модель сохранена как {MODEL_NAME}")

# import os
# os.system("shutdown /p")
