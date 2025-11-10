import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Загружаем данные
df = pd.read_csv('BD/SBER_10_NOW.csv').tail(1000)

for i in range(1, 21):
    df[f'close_plus_{i}'] = df['close'].shift(-i)

# Будет ли рост через 5 свечей? (1 = рост, 0 = падение)
df['future_close'] = df['close_plus_5']
df['target_close'] = (df['future_close'] > df['close']).astype(int)

# Максимум за 20 свечей вперед
close_columns = [f'close_plus_{i}' for i in range(1, 21)]
df['max_next_20'] = df[close_columns].max(axis=1)
df['ver'] = (df['close'] / df['close'].shift(1)) - 1

# # Отображаем свечи
# for i in range(len(df)):
#     color = 'green' if df['close'].iloc[i] >= df['open'].iloc[i] else 'red'
#     plt.plot([i, i], [df['low'].iloc[i], df['high'].iloc[i]], color='black', linewidth=1)
#     plt.plot([i, i], [df['open'].iloc[i], df['close'].iloc[i]], color=color, linewidth=3)

# Настройки графика
# plt.xticks(range(0, len(df), max(1, len(df)//10)),
#            [str(pd.to_datetime(t).time())[:5] for t in df['time'].iloc[::max(1, len(df)//10)]])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 16), sharex=True)

# График 1: Цены закрытия
ax1.plot(df.index, df['close'], label='Close Price', color='blue', linewidth=2)
ax1.set_title('Цены закрытия (база для прогноза)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Цена', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(df.index, df['max_next_20'], color='green', linewidth=2)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
