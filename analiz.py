import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
from sklearn.preprocessing import MinMaxScaler, StandardScaler

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import neuro


MODEL_NAME = "models/model_10min_step_100_pred_5_100000_2.5.keras"
FILE_NAME = 'BD/SBER_10_NOW.csv'
VOLUME = 500
TIME_STEPS = 100
NAME_PNG = f"png/png1.3/predictions_step_{TIME_STEPS}_vol_{VOLUME}_b1024.png"

model = tf.keras.models.load_model(MODEL_NAME, compile=False)

new_df = pd.read_csv(FILE_NAME).iloc[-1000:]

df = neuro.NeuroBrain()
new_df = df.data_create(new_df)
new_df = new_df.dropna()

feature_columns = df.feature_columns
#######
target_scaler = StandardScaler()

new_features = new_df[feature_columns]
new_features_scaled = target_scaler.fit_transform(new_df[feature_columns])
#######


def create_prediction_sequences(data, time_steps=100):
    """Создает последовательности для прогнозирования"""
    X_pred = []
    for i in range(time_steps, len(data)):
        X_pred.append(data[i-time_steps:i])
    return np.array(X_pred)


X_pred_seq = create_prediction_sequences(new_features_scaled, TIME_STEPS)

# Для модели классификации используем predict для получения вероятностей
probabilities = model.predict(X_pred_seq, verbose=1).flatten()
# Получаем соответствующие цены закрытия
close_prices = new_df['close'].values[TIME_STEPS:]
# Создаем DataFrame с результатами
results_df = pd.DataFrame({
    'close': close_prices,
    'probability_rise': probabilities,  # Вероятность роста
    'predicted_class': (probabilities > 0.5).astype(int)  # 1 - рост, 0 - падение
})


# Сохраняем в CSV
results_df.to_csv('predictions_results.csv', index=False, header=True)
print("Прогнозы сохранены в predictions_results.csv")
print("\nПервые 10 прогнозов:")
print(results_df.head(10))

# Шаг 8: Визуализация результатов
print("Создание графиков...")

# Создаем фигуру и subplots с синхронизацией по оси X
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

# График 1: Цены закрытия
ax1.plot(results_df.index, results_df['close'], label='Close Price', color='blue', linewidth=2)
ax1.set_title('Цены закрытия (база для прогноза)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Цена', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# График 2: Вероятности роста
ax2.plot(results_df.index, results_df['probability_rise'] * 100, color='green', linewidth=2)
ax2.set_title('Вероятность роста через 5 свечей (%)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Вероятность роста (%)', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)

# График 3: Распределение вероятностей
ax3.hist(results_df['probability_rise'], bins=50, color='purple', alpha=0.7, edgecolor='black')
ax3.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Порог 50%')
ax3.set_title('Распределение вероятностей роста', fontsize=14, fontweight='bold')
ax3.set_xlabel('Вероятность роста', fontsize=12)
ax3.set_ylabel('Количество', fontsize=12)
ax3.legend()
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(NAME_PNG, dpi=300, bbox_inches='tight')
plt.show()
