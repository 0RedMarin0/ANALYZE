import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import talib
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Шаг 1: Загрузка обученной модели
print("Загрузка обученной модели...")
model = tf.keras.models.load_model('models/my_lstm_model_60_minus5.keras')
print("Модель успешно загружена!")

# Шаг 2: Загрузка новых данных для прогноза
print("Загрузка новых данных...")
new_df = pd.read_csv('BD/GAZP_60.csv').iloc[34500:]

# Проверяем наличие необходимых колонок
required_columns = ['open', 'high', 'low', 'close', 'volume']
for col in required_columns:
    if col not in new_df.columns:
        raise ValueError(f"Отсутствует обязательная колонка: {col}")

print(f"Загружено {len(new_df)} строк данных")

# Шаг 3: Добавление индикаторов TA-Lib (таких же как при обучении)
print("Расчет индикаторов TA-Lib...")
new_df['RSI'] = talib.RSI(new_df['close'])
new_df['MACD'], new_df['MACD_signal'], new_df['MACD_hist'] = talib.MACD(new_df['close'])
new_df['BB_upper'], new_df['BB_middle'], new_df['BB_lower'] = talib.BBANDS(new_df['close'])
new_df['SMA_20'] = talib.SMA(new_df['close'], timeperiod=20)
new_df['EMA_20'] = talib.EMA(new_df['close'], timeperiod=20)
new_df['CCI'] = talib.CCI(new_df['high'], new_df['low'], new_df['close'])
new_df['SAR'] = talib.SAR(new_df['high'], new_df['low'])
new_df['ADX'] = talib.ADX(new_df['high'], new_df['low'], new_df['close'])
new_df['PLUS_DI'] = talib.PLUS_DI(new_df['high'], new_df['low'], new_df['close'])
new_df['MINUS_DI'] = talib.MINUS_DI(new_df['high'], new_df['low'], new_df['close'])

# Удаляем строки с NaN (из-за индикаторов)
new_df = new_df.dropna()
print(f"Данные после очистки: {new_df.shape}")

# Шаг 4: Подготовка данных для прогноза
print("Подготовка данных для прогноза...")

# Определяем те же признаки что и при обучении
feature_columns = ['open', 'high', 'low', 'close', 'volume',
                   'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
                   'BB_upper', 'BB_middle', 'BB_lower',
                   'SMA_20', 'EMA_20', 'CCI', 'SAR', 'ADX',
                   'PLUS_DI', 'MINUS_DI']

# Загружаем скейлеры (если сохраняли) или создаем новые
# Если скейлеры не сохранялись, нужно пересчитать на новых данных
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

# Нормализуем новые данные
new_features = new_df[feature_columns]
new_features_scaled = feature_scaler.fit_transform(new_features)

# Шаг 5: Создание последовательностей для прогноза
TIME_STEPS = 100

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
predictions_scaled = model.predict(X_pred_seq, verbose=1)

# Преобразуем прогнозы обратно в нормальные цены
# Для этого нам нужно настроить target_scaler на исходные данные
# Временно используем фиктивные данные для inverse_transform
dummy_target = np.array([new_df['close'].min(), new_df['close'].max()]).reshape(-1, 1)
target_scaler.fit(dummy_target)

predictions = target_scaler.inverse_transform(predictions_scaled)

# Шаг 7: Подготовка результатов
close_prices = new_df['close'].values[TIME_STEPS:]
dates = new_df.index[TIME_STEPS:]  # или new_df['date'] если есть колонка с датами

# Вычисляем "вероятность" - процент ожидаемого изменения
probabilities = ((predictions.flatten() - close_prices) / close_prices) * 100

# Создаем DataFrame с результатами
results_df = pd.DataFrame({
    'close': close_prices,
    'probability': probabilities
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

# График 2: Вероятности роста/падения
colors = ['red' if x < 0 else 'green' for x in results_df['probability']]
ax2.bar(results_df.index, results_df['probability'], color=colors, alpha=0.7)
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
ax2.set_title('Вероятность роста/падения (%)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Процент изменения', fontsize=12)
ax2.grid(True, alpha=0.3)

# График 3: Распределение вероятностей (НЕ синхронизируется, так как это гистограмма)
ax3.hist(results_df['probability'], bins=50, color='purple', alpha=0.7, edgecolor='black')
ax3.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Нейтральная линия')
ax3.set_title('Распределение вероятностей', fontsize=14, fontweight='bold')
ax3.set_xlabel('Процент изменения', fontsize=12)
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