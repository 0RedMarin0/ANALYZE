import numpy as np
import pandas as pd
import tensorflow as tf
import talib
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# def directional_loss(y_true, y_pred):
#     # Ошибка по направлению: если знак разный — штраф сильнее
#     diff = y_pred - y_true
#     sign_penalty = tf.where(tf.sign(y_pred) != tf.sign(y_true), 2.0, 1.0)
#     return tf.reduce_mean(tf.abs(diff) * sign_penalty)



# Шаг 1: Загрузка обученной модели
print("Загрузка обученной модели...")
model = tf.keras.models.load_model("models/model_10_version_2.keras", compile=False)
print("Модель успешно загружена!")

# Шаг 2: Загрузка новых данных для прогноза
print("Загрузка новых данных...")
new_df = pd.read_csv('BD/SBER_10.csv').tail(1000)

# Проверяем наличие необходимых колонок
required_columns = ['open', 'high', 'low', 'close', 'volume']
for col in required_columns:
    if col not in new_df.columns:
        raise ValueError(f"Отсутствует обязательная колонка: {col}")

print(f"Загружено {len(new_df)} строк данных")

# Шаг 3: Добавление индикаторов TA-Lib (таких же как при обучении)
# === Индикаторы ===
new_df['RSI'] = talib.RSI(new_df['close'])
new_df['MACD'], new_df['MACD_signal'], new_df['MACD_hist'] = talib.MACD(new_df['close'])
new_df['BB_upper'], new_df['BB_middle'], new_df['BB_lower'] = talib.BBANDS(new_df['close'])
new_df['SMA_20'] = talib.SMA(new_df['close'], 20)
new_df['EMA_20'] = talib.EMA(new_df['close'], 20)
new_df['SMA_50'] = talib.SMA(new_df['close'], 50)
new_df['SMA_100'] = talib.SMA(new_df['close'], 100)
new_df['EMA_100'] = talib.EMA(new_df['close'], 100)
new_df['SMA_200'] = talib.SMA(new_df['close'], 200)
new_df['EMA_200'] = talib.EMA(new_df['close'], 200)
new_df['CCI'] = talib.CCI(new_df['high'], new_df['low'], new_df['close'])
new_df['ADX'] = talib.ADX(new_df['high'], new_df['low'], new_df['close'])
new_df['volatility'] = talib.ATR(new_df['high'], new_df['low'], new_df['close'], 14)

# === Контекстные признаки ===
new_df['trend_strength'] = new_df['SMA_50'] / new_df['SMA_200'] - 1
new_df['momentum'] = new_df['close'] / new_df['close'].shift(10) - 1
new_df['vol_ratio'] = new_df['volume'] / new_df['volume'].rolling(50).mean()
new_df['price_pos'] = (new_df['close'] - new_df['low'].rolling(100).min()) / (new_df['high'].rolling(100).max() - new_df['low'].rolling(100).min())

new_df['slope'] = new_df['close'].diff(5)
new_df['slope'] = new_df['slope'] / new_df['close'].shift(5)

new_df['candle_body'] = new_df['close'] - new_df['open']
new_df['upper_shadow'] = new_df['high'] - new_df[['close','open']].max(axis=1)
new_df['lower_shadow'] = new_df[['close','open']].min(axis=1) - new_df['low']

new_df['returns'] = new_df['close'].pct_change()
new_df['log_return'] = np.log(new_df['close'] / new_df['close'].shift(1))

# Удаляем строки с NaN (из-за индикаторов)
new_df = new_df.dropna()
print(f"Данные после очистки: {new_df.shape}")

# Шаг 4: Подготовка данных для прогноза
print("Подготовка данных для прогноза...")

# Определяем те же признаки что и при обучении
feature_columns = [
    'open', 'high', 'low', 'close', 'volume',
    'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
    'BB_upper', 'BB_middle', 'BB_lower',
    'SMA_20', 'EMA_20', 'SMA_100', 'EMA_100', 'SMA_200', 'EMA_200', 'SMA_50',
    'CCI', 'ADX', 'volatility',
    'trend_strength', 'momentum', 'vol_ratio', 'price_pos', 'slope',
    'candle_body', 'upper_shadow', 'lower_shadow', 'returns', 'log_return'
]

# Загружаем скейлеры (если сохраняли) или создаем новые
# Если скейлеры не сохранялись, нужно пересчитать на новых данных
from sklearn.preprocessing import StandardScaler
feature_scaler = StandardScaler()
# target_scaler = MinMaxScaler()

# Нормализуем новые данные
new_features = new_df[feature_columns]
new_features_scaled = feature_scaler.fit_transform(new_df[feature_columns])

# Шаг 5: Создание последовательностей для прогноза
TIME_STEPS = 10

def create_prediction_sequences(data, time_steps=100):
    """Создает последовательности для прогнозирования"""
    X_pred = []
    for i in range(time_steps, len(data)):
        X_pred.append(data[i-time_steps:i])
    return np.array(X_pred)

X_pred_seq = create_prediction_sequences(new_features_scaled, TIME_STEPS)
print(f"Создано {X_pred_seq.shape[0]} последовательностей для прогноза")

# Шаг 6: Создание прогнозов
print("Создание прогнозов...")
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

# Вычисляем процент уверенности
results_df['confidence_percentage'] = np.where(
    results_df['predicted_class'] == 1,
    results_df['probability_rise'] * 100,
    (1 - results_df['probability_rise']) * 100
)

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
ax2.axhline(y=50, color='red', linestyle='--', linewidth=1, label='Порог 50%')
ax2.set_title('Вероятность роста через 5 свечей (%)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Вероятность роста (%)', fontsize=12)
ax2.set_ylim(0, 100)
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
plt.savefig('predictions_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# Шаг 9: Статистика прогнозов
print("\n" + "="*50)
print("СТАТИСТИКА ПРОГНОЗОВ:")
print("="*50)
print(f"Всего прогнозов: {len(results_df)}")
print(f"Прогнозов роста: {len(results_df[results_df['probability'] > 0])} ({len(results_df[results_df['probability'] > 0])/len(results_df)*100:.1f}%)")
print(f"Прогнозов падения: {len(results_df[results_df['probability'] < 0])} ({len(results_df[results_df['probability'] < 0])/len(results_df)*100:.1f}%)")
print(f"Медианное изменение: {results_df['probability'].median():.2f}%")
print(f"Среднее изменение: {results_df['probability'].mean():.2f}%")
print(f"Максимальный рост: {results_df['probability'].max():.2f}%")
print(f"Максимальное падение: {results_df['probability'].min():.2f}%")
print(f"Стандартное отклонение: {results_df['probability'].std():.2f}%")

print("\nГотово! Проверьте файлы:")
print("- predictions_results.csv - таблица с прогнозами")
print("- predictions_plot.png - графики результатов")