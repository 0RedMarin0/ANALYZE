import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import StandardScaler

import neuro

matplotlib.use('TkAgg')

# Загружаем данные
df = pd.read_csv('BD/SBER_10_NOW.csv').head(1000)

for i in range(1, 21):
    df[f'close_plus_{i}'] = df['close'].shift(-i)

# Будет ли рост через 5 свечей? (1 = рост, 0 = падение)
df['future_close'] = df['close_plus_5']
df['target_close'] = (df['future_close'] > df['close']).astype(int)

# Максимум за 20 свечей вперед
close_columns = [f'close_plus_{i}' for i in range(1, 21)]
df['max_next_20'] = df[close_columns].max(axis=1)
df['ver'] = (df['close'] / df['close'].shift(1)) - 1

df["target"] = (df["close"].shift(-20) / df["close"]) - 1

df = df.dropna()
def create_sequences(X, y, time_steps=100):
    """Создание последовательностей для временных рядов"""
    Xs, ys = [], []
    for i in range(time_steps, len(X)):
        Xs.append(X[i - time_steps:i])
        ys.append(y[i])

    # print(len(Xs), len(ys))
    # print(np.array(Xs), np.array(ys))
    return np.array(Xs), np.array(ys)

# Подготовка фич и target
neu = neuro.NeuroBrain()
features = df['close']
target = df['target']
#
# # Масштабирование
# feature_scaler = StandardScaler()
# features_scaled = feature_scaler.fit_transform(features)
#
X_seq, y_seq = create_sequences(features, target.values, 100)
# print(X_seq, y_seq)

target_scaler = StandardScaler()
yyy = target_scaler.fit_transform(y_seq.reshape(-1, 1)).flatten()
xxx = target_scaler.fit_transform(X_seq.reshape(-1, 1)).flatten()

yyy_s = np.concatenate([
    [np.nan] * 100,  # Добавляем 100 NaN в начало
    yyy[:-100]       # Берем все элементы кроме последних 100
])

print(xxx, yyy)

# xxx_s = np.concatenate([
#     [np.nan] * 100,  # Добавляем 100 NaN в начало
#     xxx[:-100]       # Берем все элементы кроме последних 100
# ])
# # Отображаем свечи
# for i in range(len(df)):
#     color = 'green' if df['close'].iloc[i] >= df['open'].iloc[i] else 'red'
#     plt.plot([i, i], [df['low'].iloc[i], df['high'].iloc[i]], color='black', linewidth=1)
#     plt.plot([i, i], [df['open'].iloc[i], df['close'].iloc[i]], color=color, linewidth=3)

# Настройки графика
# plt.xticks(range(0, len(df), max(1, len(df)//10)),
#            [str(pd.to_datetime(t).time())[:5] for t in df['time'].iloc[::max(1, len(df)//10)]])

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 16), sharex=True)

print(df.index)

# График 1: Цены закрытия
ax1.plot(df.index, df['close'], label='Close Price', color='blue', linewidth=2)
ax1.set_title('Цены закрытия (база для прогноза)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Цена', fontsize=12)
ax1.legend()
# Плотная сетка
ax1.grid(True, which='both', alpha=0.3)
ax1.minorticks_on()  # Включаем минорные деления

ax2.plot(xxx[1], color='green', linewidth=2)
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3.plot(yyy_s, color='green', linewidth=2)
ax3.legend()
ax3.grid(True, alpha=0.3)

ax3.plot(df.index, df['target'], color='green', linewidth=2)
ax3.legend()
ax3.grid(True, which='both', alpha=0.3)
ax3.minorticks_on()  # Включаем минорные деления

plt.tight_layout()
plt.show()
