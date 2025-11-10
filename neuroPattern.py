import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# === ДАННЫЕ ===
TIME_STEPS = 10
MODEL_NAME = 'models/model_15_pattern_version_2.keras'
FILE = 'BD/SBER_10.csv'
df = pd.read_csv(FILE).tail(20000)

# === ПРИЗНАКИ ===
feature_columns = ['open', 'high', 'low', 'close', 'volume']

# === ЦЕЛЕВАЯ ПЕРЕМЕННАЯ: будет ли рост через 5 свечей ===
# Создаем столбцы со сдвигом для следующих 5 свечей
for i in range(1, 6):
    df[f'close_plus_{i}'] = df['close'].shift(-i)

# Будет ли рост через 5 свечей? (1 = рост, 0 = падение)
df['future_close'] = df['close_plus_5']
df['target_close'] = (df['future_close'] > df['close']).astype(int)

# Удаляем временные столбцы
df = df.drop(columns=[f'close_plus_{i}' for i in range(1, 6)] + ['future_close'])
df = df.dropna()

features = df[feature_columns]
target = df['target_close']

# === МАСШТАБИРОВАНИЕ ===
feature_scaler = StandardScaler()
features_scaled = feature_scaler.fit_transform(features)
target_values = target.values.reshape(-1, 1)

def create_sequences(X, y, time_steps):
    Xs, ys = [], []
    for i in range(time_steps, len(X)):
        Xs.append(X[i - time_steps:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(features_scaled, target_values, TIME_STEPS)

# === РАЗДЕЛЕНИЕ ===
split = int(0.8 * len(X_seq))
X_train, X_val = X_seq[:split], X_seq[split:]
y_train, y_val = y_seq[:split], y_seq[split:]

# === МОДЕЛЬ ===
def create_classification_model(sequence_length, n_features):
    inputs = tf.keras.Input(shape=(sequence_length, n_features))

    x = tf.keras.layers.LSTM(128, return_sequences=True)(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.LSTM(32)(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    # Выход с сигмоидой для вероятности
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

model = create_classification_model(
    sequence_length=TIME_STEPS,
    n_features=len(feature_columns)
)

# Исправленные метрики для бинарной классификации
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

check = 0
try:
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=5)
        ],
        verbose=1
    )
except KeyboardInterrupt:
    print("\n⛔ Обучение остановлено вручную.")
    model.save(MODEL_NAME)
    print(f"✅ Модель сохранена как {MODEL_NAME}")
    check += 1

if check == 0:
    model.save(MODEL_NAME)
    print(f"✅ Модель сохранена как {MODEL_NAME}")