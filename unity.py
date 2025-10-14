import pandas as pd
import numpy as np
import talib
from glob import glob

files = glob("BDcrypt/CRYPTO_*_15m_YEAR.csv")

all_dfs = []
for i, f in enumerate(files):
    df = pd.read_csv(f)
    asset_name = f.split('/')[-1].replace('.csv', '')
    print(f"Загружается {asset_name} (ID={i})")

    # Добавляем индикаторы (как ты делал раньше)
    df['RSI'] = talib.RSI(df['close'])
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['close'])
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['close'])
    df['SMA_20'] = talib.SMA(df['close'], timeperiod=20)
    df['EMA_20'] = talib.EMA(df['close'], timeperiod=20)
    df['CCI'] = talib.CCI(df['high'], df['low'], df['close'])
    df['SAR'] = talib.SAR(df['high'], df['low'])
    df['ADX'] = talib.ADX(df['high'], df['low'], df['close'])
    df['PLUS_DI'] = talib.PLUS_DI(df['high'], df['low'], df['close'])
    df['MINUS_DI'] = talib.MINUS_DI(df['high'], df['low'], df['close'])
    df['SMA_200'] = talib.SMA(df['close'], timeperiod=200)
    df['EMA_200'] = talib.EMA(df['close'], timeperiod=200)
    df['trend_strength'] = df['SMA_20'] / df['SMA_200'] - 1
    df['volatility'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)

    # Добавляем целевую переменную
    df['target_close'] = (df['close'].shift(-5) - df['close']) / df['close']

    # Добавляем ID актива
    df['asset_id'] = i

    # Добавляем короткое имя актива (для наглядности)
    # df['asset_name'] = asset_name

    # Убираем NaN из-за индикаторов
    df = df.dropna()
    all_dfs.append(df)
    df.to_csv("oooooooooo.csv", mode='a')

# Объединяем всё в один датасет
combined_df = pd.concat(all_dfs, ignore_index=True)
print(f"✅ Общий датасет: {combined_df.shape}")
