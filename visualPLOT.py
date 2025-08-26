from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Загрузка и подготовка данных (как ранее)
df = pd.read_csv('data10.csv', parse_dates=['end'])
df = df.set_index('end').sort_index()



# Применяем к typical_price

# Создаем признаки
df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3  # среднее арефметич за 10 мин

df['range_hl'] = df['high'] - df['low']

df['price_change'] = df['close'] - df['open']

delta = df['typical_price'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(20).mean()
avg_loss = loss.rolling(20).mean()
rs = avg_gain / avg_loss
df['rsi1'] = 100 - (100 / (1 + rs))

def calculate_rsi(prices, period=10):
    deltas = np.diff(prices)
    seed = deltas[:period + 1]

    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period

    rs = up / down
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100. / (1. + rs)

    for i in range(period, len(prices)):
        delta = deltas[i - 1]

        if delta > 0:
            up_val = delta
            down_val = 0.
        else:
            up_val = 0.
            down_val = -delta

        up = (up * (period - 1) + up_val) / period
        down = (down * (period - 1) + down_val) / period

        rs = up / down
        rsi[i] = 100. - 100. / (1. + rs)

    return rsi


# Улучшенная функция для расчета RSI
def calculate_rsi2(prices, period=10):
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.zeros_like(prices)
    avg_loss = np.zeros_like(prices)

    # Первые значения
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])

    # Сглаживание
    for i in range(period + 1, len(prices)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i - 1]) / period

    rs = avg_gain / (avg_loss + 1e-10)  # избегаем деления на ноль
    rsi = 100 - (100 / (1 + rs))

    return rsi


# Создаем признаки
df['rsi2'] = calculate_rsi(df['close'].values)
df['rsi3'] = calculate_rsi2(df['close'].values)

# Короткие периоды (для 10-минутных свечей)
df['sma_10'] = df['typical_price'].rolling(window=10).mean()    # ~1.5 часа
df['sma_20'] = df['typical_price'].rolling(window=20).mean()    # ~3 часа
df['sma_30'] = df['typical_price'].rolling(window=30).mean()    # ~5 часов
df['sma_50'] = df['typical_price'].rolling(window=50).mean()    # ~8 часов
df['sma_60'] = df['typical_price'].rolling(window=60).mean()    # 10 часов
df['sma_100'] = df['typical_price'].rolling(window=100).mean()  # ~16 часов
df['sma_200'] = df['typical_price'].rolling(window=200).mean()  # ~33 часа

# EMA более чувствительны к последним ценам
df['ema_10'] = df['typical_price'].ewm(span=10, adjust=False).mean()
df['ema_20'] = df['typical_price'].ewm(span=20, adjust=False).mean()
df['ema_30'] = df['typical_price'].ewm(span=30, adjust=False).mean()
df['ema_50'] = df['typical_price'].ewm(span=50, adjust=False).mean()
df['ema_60'] = df['typical_price'].ewm(span=60, adjust=False).mean()


def calculate_wma(series, window):
    """Расчет взвешенной скользящей средней"""
    weights = np.arange(1, window + 1)
    wma = series.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    return wma


df['wma_10'] = calculate_wma(df['typical_price'], 10)
df['wma_20'] = calculate_wma(df['typical_price'], 20)
df['wma_30'] = calculate_wma(df['typical_price'], 30)


# Hull Moving Average (HMA) - мало лагающая MA
def calculate_hma(series, window):
    """Hull Moving Average"""
    wma_half = calculate_wma(series, window // 2)
    wma_full = calculate_wma(series, window)
    hma_series = calculate_wma(2 * wma_half - wma_full, int(np.sqrt(window)))
    return hma_series


df['hma_20'] = calculate_hma(df['typical_price'], 20)


# Kaufman Adaptive Moving Average (KAMA)
def calculate_kama(series, window=10, fast=2, slow=30):
    """Kaufman Adaptive Moving Average"""
    change = series.diff(window).abs()
    volatility = series.diff().abs().rolling(window).sum()
    efficiency_ratio = change / volatility
    smoothing_constant = (efficiency_ratio * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    kama = series.copy()
    for i in range(1, len(series)):
        if not np.isnan(smoothing_constant.iloc[i]):
            kama.iloc[i] = kama.iloc[i-1] + smoothing_constant.iloc[i] * (series.iloc[i] - kama.iloc[i-1])
    return kama


df['kama_20'] = calculate_kama(df['typical_price'], 20)


# ATR (Average True Range) - волатильность
def calculate_atr(df, window=14):
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window).mean()
    return atr


df['atr_14'] = calculate_atr(df, 14)
df['atr_20'] = calculate_atr(df, 20)

# Скользящее стандартное отклонение
df['std_20'] = df['typical_price'].rolling(20).std()
df['std_50'] = df['typical_price'].rolling(50).std()


# Разности между MA разных периодов
df['sma_10_20_diff'] = df['sma_10'] - df['sma_20']
df['ema_10_20_diff'] = df['ema_10'] - df['ema_20']
df['sma_ema_diff'] = df['sma_20'] - df['ema_20']

# Процентные отклонения
df['price_sma20_pct'] = (df['typical_price'] / df['sma_20'] - 1) * 100
df['price_ema20_pct'] = (df['typical_price'] / df['ema_20'] - 1) * 100

# Направление тренда
df['sma_trend'] = np.where(df['sma_10'] > df['sma_20'], 1, -1)
df['ema_trend'] = np.where(df['ema_10'] > df['ema_20'], 1, -1)

# Пересечения MA
df['sma_cross'] = (df['sma_10'] > df['sma_20']).astype(int)
df['ema_cross'] = (df['ema_10'] > df['ema_20']).astype(int)








# Целевая переменная
n_intervals = 1
df['target'] = (df['close'].shift(-n_intervals) > df['close']).astype(int)
df = df.dropna()


# Выделяем признаки и цель
feature_columns = ['typical_price', 'range_hl', 'price_change', 'rsi1', 'rsi2',
                   'sma_10', 'sma_20', 'sma_30', 'sma_50', 'sma_60',
                   'sma_100', 'sma_200', 'ema_10', 'ema_20', 'ema_30',
                   'ema_50', 'ema_60', 'wma_10', 'wma_20', 'wma_30',
                   'hma_20', 'kama_20', 'atr_14', 'atr_20', 'std_20',
                   'std_50', 'sma_10_20_diff', 'ema_10_20_diff', 'sma_ema_diff',
                   'price_sma20_pct', 'price_ema20_pct', 'sma_trend',
                   'ema_trend', 'sma_cross', 'ema_cross']
X = df[feature_columns]
y = df['target']


data_min, data_max = datetime(2025, 4, 5), datetime(2025, 4, 7)

# Настройка стиля графиков
plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(4, 2, figsize=(15, 12))
fig.suptitle('ВИЗУАЛИЗАЦИЯ ПРИЗНАКОВ', fontsize=16, fontweight='bold')

# Удаляем NaN значения которые появились из-за скользящих окон
df_clean = df.dropna().copy()

# 1. Исходные цены (для reference)
axes[0, 0].plot(df_clean.index, df_clean['close'], label='Close Price', linewidth=1, alpha=0.7)
axes[0, 0].set_title('Исходная цена (Close)')
axes[0, 0].set_ylabel('Цена')
axes[0, 0].set_xlim(data_min, data_max)
axes[0, 0].set_ylim(280, 293)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Typical Price
axes[0, 1].plot(df_clean.index, df_clean['typical_price'], label='Typical Price', color='orange', linewidth=1.5)
axes[0, 1].plot(df_clean.index, df_clean['close'], label='Close Price', alpha=0.5, linewidth=0.8)
axes[0, 1].set_title('Typical Price vs Close Price')
axes[0, 1].set_ylabel('Цена')
axes[0, 1].set_ylim(300, 310)
axes[0, 1].set_xlim(data_min, data_max)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Range HL (Волатильность)
axes[2, 1].plot(df_clean.index, df_clean['range_hl'], label='High-Low Range', color='red', linewidth=1)
axes[2, 1].set_title('Волатильность (High - Low Range)')
axes[2, 1].set_ylabel('Диапазон')
axes[2, 1].set_xlim(data_min, data_max)
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

# 4. Price Change
axes[1, 1].plot(df_clean.index, df_clean['price_change'], label='Price Change', color='green', linewidth=1)
axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
axes[1, 1].set_title('Изменение цены за свечу (Close - Open)')
axes[1, 1].set_ylabel('Изменение')
axes[1, 1].set_xlim(data_min, data_max)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# 5. RSI
axes[2, 0].plot(df_clean.index, df_clean['rsi1'], label='RSI', color='purple', linewidth=1.5)
axes[2, 0].axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Перекупленность (70)')
axes[2, 0].axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Перепроданность (30)')
axes[2, 0].axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Центр (50)')
axes[2, 0].set_title('RSI (Index Relative Strength 1)')
axes[2, 0].set_ylabel('RSI')
axes[2, 0].set_ylim(0, 100)
axes[2, 0].set_xlim(data_min, data_max)
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)

# 6. Gain и Loss (тоже нужно пересчитать для очищенных данных)
# 5. RSI
axes[1, 0].plot(df_clean.index, df_clean['rsi2'], label='RSI', color='purple', linewidth=1.5)
axes[1, 0].axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Перекупленность (70)')
axes[1, 0].axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Перепроданность (30)')
axes[1, 0].axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Центр (50)')
axes[1, 0].set_title('RSI (Index Relative Strength 2)')
axes[1, 0].set_ylabel('RSI')
axes[1, 0].set_ylim(0, 100)
axes[1, 0].set_xlim(data_min, data_max)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)


# 5. RSI
axes[3, 0].plot(df_clean.index, df_clean['rsi3'], label='RSI', color='purple', linewidth=1.5)
axes[3, 0].axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Перекупленность (70)')
axes[3, 0].axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Перепроданность (30)')
axes[3, 0].axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Центр (50)')
axes[3, 0].set_title('RSI (Index Relative Strength 3)')
axes[3, 0].set_ylabel('RSI')
axes[3, 0].set_ylim(0, 100)
axes[3, 0].set_xlim(data_min, data_max)
axes[3, 0].legend()
axes[3, 0].grid(True, alpha=0.3)


# Настройка layout
plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.show()


def plot_moving_averages(df, last_n=200):
    """Визуализация различных скользящих средних"""
    plot_data = df.tail(last_n)

    plt.figure(figsize=(15, 12))

    # Цена и основные MA
    plt.subplot(2, 1, 1)
    plt.plot(plot_data.index, plot_data['typical_price'], label='Price', linewidth=1, alpha=0.8)
    plt.plot(plot_data.index, plot_data['sma_20'], label='SMA 20', linewidth=2)
    plt.plot(plot_data.index, plot_data['ema_20'], label='EMA 20', linewidth=2)
    plt.plot(plot_data.index, plot_data['hma_20'], label='HMA 20', linewidth=2)
    plt.xlim(data_min, data_max)
    plt.ylim(300, 310)
    plt.title('Moving Averages Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Разности и отклонения
    plt.subplot(2, 1, 2)
    plt.plot(plot_data.index, plot_data['sma_10_20_diff'], label='SMA 10-20 Diff', alpha=0.8)
    plt.plot(plot_data.index, plot_data['price_sma20_pct'], label='Price/SMA20 %', alpha=0.8)
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.title('MA Differences and Deviations')
    plt.xlim(data_min, data_max)
    plt.ylim(300, 310)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# Запуск визуализации
plot_moving_averages(df, 200)


# Дополнительная статистика
print("📊 СТАТИСТИКА ПРИЗНАКОВ:")
print("=" * 50)
print(f"Typical Price: {df_clean['typical_price'].mean():.2f} ± {df_clean['typical_price'].std():.2f}")
print(f"Range HL:      {df_clean['range_hl'].mean():.2f} ± {df_clean['range_hl'].std():.2f}")
print(f"Price Change:  {df_clean['price_change'].mean():.2f} ± {df_clean['price_change'].std():.2f}")
print(f"RSI:           {df_clean['rsi1'].mean():.2f} ± {df_clean['rsi1'].std():.2f}")
print(f"Размер данных после очистки: {len(df_clean)} записей")