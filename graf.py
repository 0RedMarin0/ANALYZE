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

df["target"] = ((df["close"].shift(-1) / df["close"]) - 1) + \
               ((df["close"].shift(-2) / df["close"].shift(-1)) - 1) + \
               ((df["close"].shift(-3) / df["close"].shift(-2)) - 1) + \
               ((df["close"].shift(-4) / df["close"].shift(-3)) - 1) + \
               ((df["close"].shift(-5) / df["close"].shift(-4)) - 1) + \
               ((df["close"].shift(-6) / df["close"].shift(-5)) - 1) + \
               ((df["close"].shift(-7) / df["close"].shift(-6)) - 1) + \
               ((df["close"].shift(-8) / df["close"].shift(-7)) - 1) + \
               ((df["close"].shift(-9) / df["close"].shift(-8)) - 1) + \
               ((df["close"].shift(-10) / df["close"].shift(-9)) - 1) + \
               ((df["close"].shift(-11) / df["close"].shift(-10)) - 1) + \
               ((df["close"].shift(-12) / df["close"].shift(-11)) - 1) + \
               ((df["close"].shift(-13) / df["close"].shift(-12)) - 1) + \
               ((df["close"].shift(-14) / df["close"].shift(-13)) - 1) + \
               ((df["close"].shift(-15) / df["close"].shift(-14)) - 1)

# # Отображаем свечи
# for i in range(len(df)):
#     color = 'green' if df['close'].iloc[i] >= df['open'].iloc[i] else 'red'
#     plt.plot([i, i], [df['low'].iloc[i], df['high'].iloc[i]], color='black', linewidth=1)
#     plt.plot([i, i], [df['open'].iloc[i], df['close'].iloc[i]], color=color, linewidth=3)

# Настройки графика
# plt.xticks(range(0, len(df), max(1, len(df)//10)),
#            [str(pd.to_datetime(t).time())[:5] for t in df['time'].iloc[::max(1, len(df)//10)]])

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 16), sharex=True)

# График 1: Цены закрытия
ax1.plot(df.index, df['close'], label='Close Price', color='blue', linewidth=2)
ax1.set_title('Цены закрытия (база для прогноза)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Цена', fontsize=12)
ax1.legend()
# Плотная сетка
ax1.grid(True, which='both', alpha=0.3)
ax1.minorticks_on()  # Включаем минорные деления

ax2.plot(df.index, df['max_next_20'], color='green', linewidth=2)
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3.plot(df.index, df['target'], color='green', linewidth=2)
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
