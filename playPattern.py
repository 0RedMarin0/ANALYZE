import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

print("Загрузка обученной модели...")
model = tf.keras.models.load_model("models/model_15_pattern_version_2.keras", compile=False)
print("Модель успешно загружена!")

print("Загрузка новых данных...")
new_df = pd.read_csv('BD/SBER_10_NOW.csv').tail(1000)

required_columns = ['open', 'high', 'low', 'close', 'volume']
for col in required_columns:
    if col not in new_df.columns:
        raise ValueError(f"Отсутствует обязательная колонка: {col}")

feature_columns = ['open', 'high', 'low', 'close', 'volume']

from sklearn.preprocessing import StandardScaler
feature_scaler = StandardScaler()

new_features = new_df[feature_columns]
new_features_scaled = feature_scaler.fit_transform(new_features)

TIME_STEPS = 10

def create_prediction_sequences(data, time_steps=10):
    """Создает последовательности для прогнозирования"""
    X_pred = []
    for i in range(time_steps, len(data)):
        X_pred.append(data[i-time_steps:i])
    return np.array(X_pred)

X_pred_seq = create_prediction_sequences(new_features_scaled, TIME_STEPS)
print(f"Создано {X_pred_seq.shape[0]} последовательностей для прогноза")

# Создание прогнозов (вероятностей)
print("Создание прогнозов...")
predictions = model.predict(X_pred_seq, verbose=1)

# Получаем вероятности роста (класс 1)
probabilities = predictions.flatten()

# Получаем соответствующие цены закрытия
close_prices = new_df['close'].values[TIME_STEPS:]

# Создаем DataFrame с результатами
results_df = pd.DataFrame({
    'close': close_prices,
    'probability_rise': probabilities,  # Вероятность роста
    'predicted_class': (probabilities > 0.5).astype(int)  # Класс: 1 - рост, 0 - падение
})

# Рассчитываем процент уверенности
results_df['confidence_percentage'] = np.where(
    results_df['predicted_class'] == 1,
    results_df['probability_rise'] * 100,  # Для роста: вероятность * 100
    (1 - results_df['probability_rise']) * 100  # Для падения: (1 - вероятность) * 100
)

results_df.to_csv('predictions_results.csv', index=False, header=True)
print("Прогнозы сохранены в predictions_results.csv")
print("\nПервые 10 прогнозов:")
print(results_df.head(10))

print("Создание графиков...")

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 16), sharex=True)

# График 1: Цены закрытия
ax1.plot(results_df.index, results_df['close'], label='Close Price', color='blue', linewidth=2)
ax1.set_title('Цены закрытия (база для прогноза)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Цена', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# График 2: Вероятность роста
ax2.plot(results_df.index, results_df['probability_rise'] * 100, color='green', linewidth=2)
ax2.axhline(y=50, color='red', linestyle='--', linewidth=1, label='Порог 50%')
ax2.set_title('Вероятность роста через 5 свечей (%)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Вероятность роста (%)', fontsize=12)
ax2.set_ylim(0, 100)
ax2.legend()
ax2.grid(True, alpha=0.3)

# График 3: Прогнозируемый класс (рост/падение)
colors = ['red' if x == 0 else 'green' for x in results_df['predicted_class']]
ax3.bar(results_df.index, results_df['predicted_class'], color=colors, alpha=0.6)
ax3.set_title('Прогнозируемый класс (0=падение, 1=рост)', fontsize=14, fontweight='bold')
ax3.set_ylabel('Класс', fontsize=12)
ax3.set_yticks([0, 1])
ax3.set_yticklabels(['Падение', 'Рост'])
ax3.grid(True, alpha=0.3)

# График 4: Процент уверенности в прогнозе
ax4.plot(results_df.index, results_df['confidence_percentage'], color='purple', linewidth=2)
ax4.set_title('Уверенность в прогнозе (%)', fontsize=14, fontweight='bold')
ax4.set_xlabel('Временные точки', fontsize=12)
ax4.set_ylabel('Уверенность (%)', fontsize=12)
ax4.set_ylim(0, 100)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('predictions_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# Статистика прогнозов
print("\n" + "="*50)
print("СТАТИСТИКА ПРОГНОЗОВ:")
print("="*50)
print(f"Всего прогнозов: {len(results_df)}")
print(f"Прогнозов роста: {len(results_df[results_df['predicted_class'] == 1])} ({len(results_df[results_df['predicted_class'] == 1])/len(results_df)*100:.1f}%)")
print(f"Прогнозов падения: {len(results_df[results_df['predicted_class'] == 0])} ({len(results_df[results_df['predicted_class'] == 0])/len(results_df)*100:.1f}%)")
print(f"Средняя вероятность роста: {results_df['probability_rise'].mean()*100:.2f}%")
print(f"Медианная вероятность роста: {results_df['probability_rise'].median()*100:.2f}%")
print(f"Максимальная вероятность роста: {results_df['probability_rise'].max()*100:.2f}%")
print(f"Минимальная вероятность роста: {results_df['probability_rise'].min()*100:.2f}%")

# Анализ уверенности прогнозов
high_confidence = results_df[results_df['confidence_percentage'] > 70]
medium_confidence = results_df[(results_df['confidence_percentage'] >= 50) & (results_df['confidence_percentage'] <= 70)]
low_confidence = results_df[results_df['confidence_percentage'] < 50]

print(f"\nАНАЛИЗ УВЕРЕННОСТИ:")
print(f"Высокая уверенность (>70%): {len(high_confidence)} прогнозов ({len(high_confidence)/len(results_df)*100:.1f}%)")
print(f"Средняя уверенность (50-70%): {len(medium_confidence)} прогнозов ({len(medium_confidence)/len(results_df)*100:.1f}%)")
print(f"Низкая уверенность (<50%): {len(low_confidence)} прогнозов ({len(low_confidence)/len(results_df)*100:.1f}%)")

print("\nГотово! Проверьте файлы:")
print("- predictions_results.csv - таблица с прогнозами")
print("- predictions_plot.png - графики результатов")