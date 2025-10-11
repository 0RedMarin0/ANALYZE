import os
import time

import requests
import pandas as pd
from datetime import datetime, timedelta


def get_moex_candles(ticker='SBER', params=None):
    """
    Получаем свечи по акциям
    interval: 1 (1 мин), 10 (10 мин), 60 (1 час), 24 (1 день)
    """
    url = f'https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/{ticker}/candles.json'
    response = requests.get(url, params=params)
    data = response.json()
    candles = pd.DataFrame(data['candles']['data'],
                           columns=data['candles']['columns'])
    candles['begin'] = pd.to_datetime(candles['begin'])
    candles.set_index('begin', inplace=True)
    return candles

def start(filename, start_date, end_date, interval, day, ticker):
    if os.path.exists(filename):
        os.remove(filename)

    i = 0
    stop = 0
    while True:
        ppp = {
            'from': start_date.strftime('%Y-%m-%d %H:%M:%S'),
            'till': end_date.strftime('%Y-%m-%d %H:%M:%S'),
            'interval': interval,
            'iss.meta': 'off'
        }

        print(f"Итерация {i + 1}: с {start_date} по {end_date}")

        gazp_candles = get_moex_candles(ticker, params=ppp)  # Исправлено на GAZP вместо MOEX
        print(f"Получено строк: {len(gazp_candles)}")
        if not gazp_candles.empty:
            print(gazp_candles.head())

        if i == 0:
            gazp_candles.to_csv(filename, mode='a')
        else:
            gazp_candles.to_csv(filename, mode='a', header=False)

        start_date += timedelta(days=day)
        end_date += timedelta(days=day)

        i += 1

        if len(gazp_candles) == 0:
            stop += 1
            if stop == 20:
                break


if __name__ == '__main__':
    start_date = datetime(2025, 8, 20, 1, 0, 0)  # 2025-07-01 07:00:00 2025-08-22 17:44:59
    end_date = datetime(2025, 8, 22, 23, 59, 0)  # 2025-07-01 12:00:00
    interval = 10
    day = 3
    ticker = 'MRKZ'
    filename = f'BD/{ticker}_{interval}_NOW.csv'
    start(filename, start_date, end_date, interval, day, ticker)
