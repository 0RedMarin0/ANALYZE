# import yfinance as yf
# import matplotlib.pyplot as plt
#
# # Получаем данные по акциям Apple (AAPL)
# data = yf.download('TEVA',
#                    start='2025-08-20',
#                    progress=False,
#                    interval='1m')
#
# print(data)  # Первые 5 строк
# print(f"\nДанные с {data.index[0]} по {data.index[-1]}")
# print(f"Кол-во записей: {len(data)}")
# data.to_csv('apple_data.txt', sep='\t')  # табуляция как разделитель
#
# # Построим график
# plt.figure(figsize=(12, 6))
# plt.plot(data['Close'], label='Apple (AAPL)')
# plt.title('Цены акций Apple')
# plt.xlabel('Дата')
# plt.ylabel('Цена ($)')
# plt.legend()
# plt.grid(True)
# plt.show()
# print(data)  # Первые 5 строк

# pip install apimoex
import requests
import pandas as pd
import matplotlib.pyplot as plt


def get_moex_candles(ticker='EUTR', interval=24):
    """
    Получаем свечи по акциям
    interval: 1 (1 мин), 10 (10 мин), 60 (1 час), 24 (1 день)
    """
    url = f'https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/{ticker}/candles.json'

    params = {
        'from': '2025-05-20',
        'till': '2025-08-21',
        'interval': interval,
        'iss.meta': 'off'
    }

    response = requests.get(url, params=params)
    data = response.json()

    candles = pd.DataFrame(data['candles']['data'],
                           columns=data['candles']['columns'])

    candles['begin'] = pd.to_datetime(candles['begin'])
    candles.set_index('begin', inplace=True)

    return candles



# Получаем дневные свечи Газпрома
gazp_candles = get_moex_candles('GAZP', 24)
print(gazp_candles.head())
gazp_candles.to_csv('apple_data.txt', sep='\t')  # табуляция как разделитель

plt.figure(figsize=(12, 6))
plt.plot(gazp_candles['close'], label='Apple (AAPL)')
plt.title('Цены акций Apple')
plt.xlabel('Дата')
plt.ylabel('Цена ($)')
plt.legend()
plt.grid(True)
plt.show()
