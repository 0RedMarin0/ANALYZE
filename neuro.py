# from keras.src.saving import register_keras_serializable
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import tensorflow as tf
import table


FILE = 'BD/SBER_10.csv'
TIME_STEPS = 10
MODEL_NAME = 'models/model_10_version_2.keras'

df_start = table.DataCreate(FILE)
df = df_start.table

for i in range(1, 21):
    df[f'close_plus_{i}'] = df['close'].shift(-i)
# Будет ли рост через 5 свечей? (1 = рост, 0 = падение)
df['future_close'] = df['close_plus_5']
df['target_close'] = (df['future_close'] > df['close']).astype(int)
# Максимум за 20 свечей вперед
close_columns = [f'close_plus_{i}' for i in range(1, 21)]
df['target_close'] = df[close_columns].max(axis=1)
# df['target_close'] = (df['close'].shift(-5) / df['close']) - 1

df = df.dropna()

# === Признаки ===
feature_columns = df_start.list_sign
features = df[feature_columns]
target = df['target_close']

# === Масштабирование: StandardScaler вместо MinMax ===
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

#
# @register_keras_serializable(package="Custom", name="directional_loss")
# def directional_loss(y_true, y_pred):
#     diff = y_pred - y_true
#     sign_penalty = tf.where(tf.sign(y_pred) != tf.sign(y_true), 2.0, 1.0)
#     return tf.reduce_mean(tf.abs(diff) * sign_penalty)
def directional_loss(y_true, y_pred):
    diff = y_pred - y_true
    sign_penalty = tf.where(tf.sign(y_pred) != tf.sign(y_true), 5.0, 1.0)  # было 2.0
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

check = 0
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
    check += 1

if check == 0:
    print("\n⛔ Обучение остановлено вручную.")
    model.save(MODEL_NAME)
    print(f"✅ Модель сохранена как {MODEL_NAME}")
# import os
# os.system("shutdown /p")
