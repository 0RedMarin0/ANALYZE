import os
from datetime import datetime, timedelta
import pandas as pd
from tinkoff.invest import Client, CandleInterval

TOKEN = "t.SkOk2hbyag3ShJnUpSNl6TROkEr0f1x9lbRED2f2qLrSe3jTrbJs0RHZrGm60X2Khaniqra1xibXn6Cg1xvWjw"
FIGI = "BBG004730N88"  # Пример: Сбербанк (SBER)

INTERVAL_MAPPING = {
    1: CandleInterval.CANDLE_INTERVAL_1_MIN,
    5: CandleInterval.CANDLE_INTERVAL_5_MIN,
    15: CandleInterval.CANDLE_INTERVAL_15_MIN,
    60: CandleInterval.CANDLE_INTERVAL_HOUR,
    24: CandleInterval.CANDLE_INTERVAL_DAY
}

def get_tinkoff_candles(client, figi, dt_from, dt_to, interval):
    response = client.market_data.get_candles(
        figi=figi,
        from_=dt_from,
        to=dt_to,
        interval=interval
    )
    candles = []
    for c in response.candles:
        candles.append({
            "time": c.time,
            "open": c.open.units + c.open.nano / 1e9,
            "high": c.high.units + c.high.nano / 1e9,
            "low": c.low.units + c.low.nano / 1e9,
            "close": c.close.units + c.close.nano / 1e9,
            "volume": c.volume
        })
    df = pd.DataFrame(candles)
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"])
        df.set_index("time", inplace=True)
    return df


def start_tinkoff(filename, start_date, end_date, day, figi, interval_minutes):
    if os.path.exists(filename):
        os.remove(filename)

    interval = INTERVAL_MAPPING.get(interval_minutes)
    if interval is None:
        raise ValueError("Неподдерживаемый интервал")

    with Client(TOKEN) as client:
        i = 0
        stop = 0
        while True:
            print(f"Итерация {i+1}: {start_date} – {end_date}")

            df = get_tinkoff_candles(client, figi, start_date, end_date, interval)
            print(f"Получено {len(df)} строк")

            if not df.empty:
                mode = 'a' if i > 0 else 'w'
                df.to_csv(filename, mode=mode, header=(mode == 'w'))

            start_date += timedelta(days=day)
            end_date += timedelta(days=day)
            i += 1

            if df.empty:
                stop += 1
                if stop >= 5:
                    print("Данных больше нет, остановка.")
                    break


if __name__ == '__main__':
    filename = "BD/SBER_15m.csv"
    start_date = datetime(2025, 8, 20, 1, 0, 0)
    end_date = datetime(2025, 8, 22, 23, 59, 0)
    day = 7
    figi = "BBG004730N88"  # FIGI Сбербанка
    interval_minutes = 15

    start_tinkoff(filename, start_date, end_date, day, figi, interval_minutes)
