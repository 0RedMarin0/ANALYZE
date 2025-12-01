import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import table

matplotlib.use('TkAgg')

# Загружаем данные
df = table.DataCreate(pd.read_csv('BD/SBER_10_NOW.csv').head(20000))
df = df.table

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

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(15, 16), sharex=True)

print(df.index)

# График 1: Цены закрытия
a1 = 'close'
ax1.plot(df.index, df[a1], label='Close Price', color='blue', linewidth=2)
ax1.set_title('Цены закрытия (база для прогноза)', fontsize=14, fontweight='bold')
ax1.set_ylabel(a1, fontsize=12)
ax1.legend()
# Плотная сетка
ax1.grid(True, which='both', alpha=0.3)
ax1.minorticks_on()  # Включаем минорные деления

a2 = 'MACD'
ax2.plot(df.index, df[a2], color='green', linewidth=2)
ax2.plot(df.index, df["MACD_signal"], color='red', linewidth=2)
ax2.set_ylabel(a2, fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)

a3 = 'CCI'
ax3.plot(df.index, df[a3], color='green', linewidth=2)
ax3.set_ylabel(a3, fontsize=12)
ax3.legend()
ax3.grid(True, which='both', alpha=0.3)
ax3.minorticks_on()  # Включаем минорные деления

a4 = 'WILLR'
ax4.plot(df.index, df[a4], color='green', linewidth=2)
ax4.set_ylabel(a4, fontsize=12)
ax4.legend()
ax4.grid(True, which='both', alpha=0.3)
ax4.minorticks_on()  # Включаем минорные деления

a5 = 'ADX'
ax5.plot(df.index, (df['close'].shift(-1) / df['close']) - 1, color='green', linewidth=2)
ax5.set_ylabel(a5, fontsize=12)
ax5.legend()
ax5.grid(True, which='both', alpha=0.3)
ax5.minorticks_on()  # Включаем минорные деления

plt.tight_layout()
plt.show()
