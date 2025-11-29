import matplotlib
import pandas as pd

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from NeuroPredictor import NeuroPredictor

# Константы
MODEL_NAME = "model_complete.pkl"  # теперь используем наш пакет
FILE_NAME = 'BD/SBER_10_NOW.csv'
VOLUME = 500
NAME_PNG = f"png/predictions_vol_{VOLUME}.png"

# 🎯 ВСЕГО 3 ОСНОВНЫЕ СТРОКИ ДЛЯ ПРОГНОЗА!
predictor = NeuroPredictor()
new_df = pd.read_csv(FILE_NAME)
close_prices, probabilities = predictor.predict(new_df)

# Сохраняем результаты
results_df = predictor.save_predictions(close_prices, probabilities)

print("\nПервые 10 прогнозов:")
print(results_df.head(10))

# Визуализация (ваш код без изменений)
print("Создание графиков...")
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