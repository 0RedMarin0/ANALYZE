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
from datetime import datetime, timedelta


def get_moex_candles(ticker='SBER', params=None):
    """
    Получаем свечи по акциям
    interval: 1 (1 мин), 10 (10 мин), 60 (1 час), 24 (1 день)
    """
    if params is None:
        params = {
            'from': '2025-07-10',
            'till': '2025-07-14',
            'interval': 10,
            'iss.meta': 'off'
        }
    url = f'https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR/securities/{ticker}/candles.json'

    response = requests.get(url, params=params)
    data = response.json()

    candles = pd.DataFrame(data['candles']['data'],
                           columns=data['candles']['columns'])

    candles['begin'] = pd.to_datetime(candles['begin'])
    candles.set_index('begin', inplace=True)

    return candles


# Начальные даты
# Начальные даты
start_date = datetime(2024, 8, 11, 1, 0, 0)  # 2025-07-01 07:00:00 2025-08-22 17:44:59
end_date = datetime(2024, 8, 13, 23, 59, 0)   # 2025-07-01 12:00:00
filename = 'data10.csv'

# Очищаем файл перед началом
# open(filename, 'w').close()

for i in range(130):
    # Форматируем даты для запроса
    ppp = {
        'from': start_date.strftime('%Y-%m-%d %H:%M:%S'),
        'till': end_date.strftime('%Y-%m-%d %H:%M:%S'),
        'interval': 10,
        'iss.meta': 'off'
    }

    print(f"Итерация {i + 1}: с {start_date} по {end_date}")

    # Получаем свечи
    gazp_candles = get_moex_candles('SBER', params=ppp)  # Исправлено на GAZP вместо MOEX
    print(f"Получено строк: {len(gazp_candles)}")
    if not gazp_candles.empty:
        print(gazp_candles.head())

    # Записываем в файл
    if i == 0:
        gazp_candles.to_csv(filename, mode='a')
    else:
        gazp_candles.to_csv(filename, mode='a', header=False)

    # Увеличиваем время на 5 часов
    start_date += timedelta(days=3)
    end_date += timedelta(days=3)

print("Все итерации завершены! Данные сохранены в", filename)

#
# plt.title('Цены акций Apple')
# plt.xlabel('Дата')
# plt.ylabel('Цена ($)')
# plt.legend()
# plt.grid(True)
# plt.show()
