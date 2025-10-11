import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta


def get_binance_candles(symbol='BTCUSDT', interval='1m', start_time=None, end_time=None):
    """
    Получаем свечи с Binance
    interval: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d
    """
    url = 'https://api.binance.com/api/v3/klines'

    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': 1000  # Максимум 1000 свечей за запрос
    }

    if start_time:
        params['startTime'] = int(start_time.timestamp() * 1000)
    if end_time:
        params['endTime'] = int(end_time.timestamp() * 1000)

    response = requests.get(url, params=params)
    data = response.json()

    # Конвертируем в DataFrame
    columns = ['open_time', 'open', 'high', 'low', 'close', 'volume',
               'close_time', 'quote_asset_volume', 'number_of_trades',
               'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore']

    df = pd.DataFrame(data, columns=columns)

    # Конвертируем временные метки
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

    # Конвертируем цены и объемы в числа
    numeric_columns = ['open', 'high', 'low', 'close', 'volume',
                       'quote_asset_volume', 'taker_buy_base_volume',
                       'taker_buy_quote_volume']
    df[numeric_columns] = df[numeric_columns].astype(float)

    df.set_index('open_time', inplace=True)
    return df[['open', 'high', 'low', 'close', 'volume']]  # Возвращаем основные колонки


def start_crypto(filename, start_date, end_date, interval, days_chunk, symbol):
    """
    interval: 1m, 5m, 15m, 1h, 4h, 1d и т.д.
    days_chunk: количество дней за один запрос (зависит от интервала)
    """
    if os.path.exists(filename):
        os.remove(filename)

    i = 0
    stop = 0
    current_start = start_date
    current_end = end_date

    while True:
        print(f"Итерация {i + 1}: с {current_start} по {current_end}")

        try:
            candles = get_binance_candles(
                symbol=symbol,
                interval=interval,
                start_time=current_start,
                end_time=current_end
            )

            print(f"Получено строк: {len(candles)}")

            if not candles.empty:
                print(candles.head())

                # Сохраняем в файл
                if i == 0:
                    candles.to_csv(filename, mode='a')
                else:
                    candles.to_csv(filename, mode='a', header=False)

                stop = 0  # Сбрасываем счетчик пустых запросов
            else:
                print("Пустой ответ")
                stop += 1

        except Exception as e:
            print(f"Ошибка: {e}")
            stop += 1
            time.sleep(1)  # Пауза при ошибке

        # Сдвигаем даты
        current_start = current_end + timedelta(seconds=1)
        current_end = current_start + timedelta(days=days_chunk)

        i += 1

        # # Проверяем текущую дату (не заходим в будущее)
        # if current_start > datetime.now():
        #     print("Достигнута текущая дата")
        #     break

        if stop >= 5:  # Меньше пустых запросов для остановки
            print("Слишком много пустых ответов, завершаем")
            break

        time.sleep(0.1)  # Небольшая пауза между запросами


# Дополнительная функция для популярных криптопар
def get_popular_pairs():
    """Список популярных криптопар"""
    return {
        'BTCUSDT': 'Bitcoin',
        'ETHUSDT': 'Ethereum',
        'BNBUSDT': 'Binance Coin',
        'ADAUSDT': 'Cardano',
        'DOTUSDT': 'Polkadot',
        'LTCUSDT': 'Litecoin',
        'LINKUSDT': 'Chainlink',
        'BCHUSDT': 'Bitcoin Cash',
        'XLMUSDT': 'Stellar',
        'DOGEUSDT': 'Dogecoin'
    }


if __name__ == '__main__':
    # Настройки для сбора данных
    start_date = datetime(2023, 2, 20)  # Начальная дата
    end_date = datetime(2023, 2, 22)  # Конец первого чанка
    interval = '15m'  # 5-минутные свечи
    days_chunk = 7  # 7 дней за запрос (для 5m интервала)
    symbol = 'BTCUSDT'  # Биткоин к USDT
    filename = f'BDcrypt/CRYPTO_{symbol}_{interval}_YEAR.csv'

    print(f"Собираем данные для {symbol}")
    print(f"Интервал: {interval}")
    print(f"Период: с {start_date} по текущее время")

    start_crypto(filename, start_date, end_date, interval, days_chunk, symbol)