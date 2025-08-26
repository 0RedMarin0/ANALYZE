"""
распределим индикаторы


1. Трендовые индикаторы:
    Скользящие средние (MA) :
        Простая (SMA): SMA = (Sum(Close, n)) / n
        Экспоненциальная (EMA): EMA = (Close * k) + (EMA_prev * (1 - k)), где k = 2 / (n + 1)
        Применение: Определение тренда, уровней поддержки/сопротивления.

    MACD (Moving Average Convergence Divergence) показывает взаимосвязь между двумя EMA:
        Формула: MACD = EMA(Close, 12) - EMA(Close, 26)
        Сигнальная линия: Signal = EMA(MACD, 9)
        Применение: Пересечение линии MACD сигнальной — торговый сигнал. Дивергенция — разворот

    Parabolic SAR точечный индикатор, показывающий возможные точки разворота тренда:
        Формула сложна, рассчитывается итеративно с учетом экстремумов и коэффициента ускорения (AF).
        Применение: Точки выше цены — медвежий тренд, ниже — бычий. Стоп-лосс и трейлинг-стоп.

    ADX (Average Directional Index) — измеряет силу тренда, но не его направление:
        Формула сложна, основана на сравнении текущих макс/мин с предыдущими (+DI и -DI).
        Применение: ADX > 25 — сильный тренд. ADX < 20 — флэт (боковое движение).

2. Осцилляторы (определяют моменты перекупленности/перепроданности):
    RSI (Relative Strength Index) — измеряет скорость изменения цен:
        Формула: RSI = 100 - (100 / (1 + RS)), где RS = Avg Gain / Avg Loss за период n.
        Применение: >70 — перекупленность (зона продаж), <30 — перепроданность (зона покупок). Дивергенция.

    Stochastic Oscillator — сравнивает текущую цену с диапазоном цен за период:
        Формула: %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
        Применение: >80 — перекупленность, <20 — перепроданность. Пересечение линий %K и %D.

    CCI (Commodity Channel Index) — оценивает отклонение цены от своего статистического среднего.
        Формула: CCI = (Typical Price - SMA(TP, n)) / (0.015 * Mean Deviation)
        *Применение: Выход за уровни +100/-100 — сильный тренд. >+200 — перекупленность, <-200 — перепроданность.*

    Williams %R — аналогичен Stochastic, но инвертирован.
        Формула: %R = (Highest High - Close) / (Highest High - Lowest Low) * (-100)
        *Применение: > -20 — перекупленность, < -80 — перепроданность.*

3. Индикаторы объема:
    OBV (On Balance Volume) — кумулятивный индикатор, связывающий объем с изменением цены.
        Формула: Если сегодняшняя Close > вчерашней, то: OBV = OBV_prev + Volume
        Если сегодняшняя Close < вчерашней, то: OBV = OBV_prev - Volume
        Применение: Подтверждение тренда. Если цена растет и OBV растет — тренд сильный. Дивергенция.

    Volume — обычные гистограммы объема под графиком.
        Применение: Рост объема подтверждает движение цены. Низкий объем — отсутствие интереса.

4. Индикаторы волатильности (измеряют размах ценовых колебаний)
    ATR (Average True Range) — показывает средний диапазон движения цены за период.
        Формула: True Range = Max(High - Low, |High - Close_prev|, |Low - Close_prev|)
        ATR = SMA(True Range, n)
        Применение: Оценка волатильности для установки стоп-лоссов.

    Полосы Боллинджера (Bollinger Bands®) — динамический канал волатильности.
        Формула:
            Средняя линия: SMA(20)
            Верхняя полоса: SMA(20) + (2 * Std Dev(Close, 20))
            Нижняя полоса: SMA(20) - (2 * Std Dev(Close, 20))
        Применение: Отскок от полос — продолжение тренда. Сужение ("squeeze") — предвестник сильного движения.


                                        НАМ ПОНРАВИЛОСЬ!!!!!!!
                                        RSI
                                        MACD
                                        Скользящие средние MA (SMA, EMA)
                                        ATR
                                        CCI

Парсер:
1 минута - по 5 часов
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf

# Загрузка данных
df = pd.read_csv('dataNOW.csv', parse_dates=['begin', 'end'], index_col='begin')

# Убедимся, что данные отсортированы по времени
df.sort_index(inplace=True)

# Вычисление индикаторов
# 1. RSI
# 2. Осцилляторы (определяют моменты перекупленности/перепроданности):
#     RSI (Relative Strength Index) — измеряет скорость изменения цен:
#         Формула: RSI = 100 - (100 / (1 + RS)), где RS = Avg Gain / Avg Loss за период n.
#         Применение: >70 — перекупленность (зона продаж), <30 — перепроданность (зона покупок). Дивергенция.
def calculate_rsi(data, window=14):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['RSI'] = calculate_rsi(df)

# 2. MACD
def calculate_macd(data, fast=12, slow=26, signal=9):
    ema_fast = data['close'].ewm(span=fast).mean()
    ema_slow = data['close'].ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    macd_histogram = macd - macd_signal
    return macd, macd_signal, macd_histogram

macd, macd_signal, macd_histogram = calculate_macd(df)
df['MACD'] = macd
df['MACD_signal'] = macd_signal
df['MACD_histogram'] = macd_histogram

# 3. Скользящие средние
df['SMA_20'] = df['close'].rolling(window=20).mean()
df['EMA_20'] = df['close'].ewm(span=20).mean()

# 4. ATR
def calculate_atr(data, window=14):
    high_low = data['high'] - data['low']
    high_close = np.abs(data['high'] - data['close'].shift())
    low_close = np.abs(data['low'] - data['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=window).mean()
    return atr

df['ATR'] = calculate_atr(df)

# 5. CCI
def calculate_cci(data, window=20):
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    sma = typical_price.rolling(window=window).mean()
    mean_deviation = typical_price.rolling(window=window).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    cci = (typical_price - sma) / (0.015 * mean_deviation)
    return cci

df['CCI'] = calculate_cci(df)

# Создание графиков
plt.figure(figsize=(14, 12))

# График цены и скользящих средних
plt.subplot(4, 1, 1)
plt.plot(df.index, df['close'], label='Цена закрытия', linewidth=1)
plt.plot(df.index, df['SMA_20'], label='SMA 20', linestyle='--', alpha=0.8)
plt.plot(df.index, df['EMA_20'], label='EMA 20', linestyle='--', alpha=0.8)
plt.title('График цены и скользящих средних')
plt.legend()
plt.grid(True)

# График MACD
plt.subplot(4, 1, 2)
plt.plot(df.index, df['MACD'], label='MACD', linewidth=1)
plt.plot(df.index, df['MACD_signal'], label='Сигнальная линия', linewidth=1)
plt.bar(df.index, df['MACD_histogram'], label='Гистограмма', alpha=0.3)
plt.title('MACD')
plt.legend()
plt.grid(True)

# График RSI
plt.subplot(4, 1, 3)
plt.plot(df.index, df['RSI'], label='RSI', linewidth=1, color='purple')
plt.axhline(70, linestyle='--', alpha=0.5, color='red')
plt.axhline(30, linestyle='--', alpha=0.5, color='green')
plt.title('RSI (14 периодов)')
plt.legend()
plt.grid(True)

# График ATR и CCI
plt.subplot(4, 1, 4)
plt.plot(df.index, df['ATR'], label='ATR', linewidth=1, color='orange')
plt.plot(df.index, df['CCI'], label='CCI', linewidth=1, color='blue')
plt.axhline(100, linestyle='--', alpha=0.5, color='red')
plt.axhline(-100, linestyle='--', alpha=0.5, color='green')
plt.title('ATR и CCI')
plt.legend()
plt.grid(True)

# plt.savefig('my_plot.png')  # Вот эта строка всё сохраняет

plt.tight_layout()
plt.show()

print(df)
df.to_csv("infaGraf.csv", mode='a')
# Дополнительно: свечной график с mplfinance
apds = [
    mpf.make_addplot(df['SMA_20'], color='blue'),
    mpf.make_addplot(df['EMA_20'], color='orange'),
]

mpf.plot(df, type='candle', addplot=apds,
         style='charles', title='Свечной график с MA',
         ylabel='Цена', volume=True,
         figratio=(12, 8), figscale=1.2)

