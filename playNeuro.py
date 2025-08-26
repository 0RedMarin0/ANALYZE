import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# ==================== 1. Загрузка модели ====================
model = tf.keras.models.load_model('stock_price_predictor_final.h5')
print("Модель загружена!")

# ==================== 2. Подготовка данных ====================
# Загрузите новые данные для предсказания
data = pd.read_csv('your_data.csv')  # замените на свой файл

# Выберите признаки, которые использовались при обучении
features = ['close', 'rsi_14', 'volume_change_pct', 'high_low_spread',
            'close_open_spread', 'sma_20', 'ema_12']

# Преобразование типов и удаление пропусков
for col in features:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data = data.dropna().reset_index(drop=True)

# Нормализация (важно использовать те же MinMaxScaler параметры!)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[features])

# Средние значения признаков для генерации новых последовательностей
feature_means = np.mean(scaled_data, axis=0)

# ==================== 3. Создание последовательностей ====================
SEQ_LENGTH = 60
TARGET_LENGTH = 30

def create_sequences(data: np.ndarray, seq_length: int, target_length: int):
    X = []
    for i in range(len(data) - seq_length - target_length + 1):
        X.append(data[i:i + seq_length])
    return np.array(X, dtype=np.float32)

X_all = create_sequences(scaled_data, SEQ_LENGTH, TARGET_LENGTH)
last_sequence = X_all[-1]  # берём последнюю последовательность для предсказания

# ==================== 4. Предсказание ====================
def predict_future_prices(model, last_sequence, target_length, feature_means):
    current_sequence = last_sequence.copy()
    predictions = []

    for _ in range(target_length):
        # Предсказание
        pred_tensor = model(tf.expand_dims(current_sequence, 0))
        next_price = pred_tensor[0, 0].numpy()  # первая цена в выходе

        # Создание новой строки данных
        new_row = np.zeros((1, current_sequence.shape[1]))
        new_row[0, 0] = next_price
        new_row[0, 1:] = feature_means[1:]  # остальные признаки

        # Обновляем последовательность
        current_sequence = np.vstack([current_sequence[1:], new_row])
        predictions.append(next_price)

    return np.array(predictions)

future_predictions = predict_future_prices(model, last_sequence, TARGET_LENGTH, feature_means)

# ==================== 5. Денормализация ====================
# Если при обучении использовался MinMaxScaler
y_pred_actual = future_predictions * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]

# ==================== 6. Вывод результата ====================
print("Прогнозируемые цены на следующие 30 свечей:")
for i, price in enumerate(y_pred_actual, 1):
    print(f"Свеча {i:2d}: {price:.2f}")
