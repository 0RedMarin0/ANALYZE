import numpy as np
import pandas
import talib


class DataCreate:
    def __init__(self, data):
        self.data = data # pandas.read_csv(data)

        self.list_sign = [] # 'open', 'high', 'low', 'close', 'volume'

        self.modif()

        self.table = self.data.dropna()

    def modif(self):
        self.list_sign.extend(['RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'CCI',
                               'ADX', 'STOCH_K', 'STOCH_D', 'ATR',
                               'MFI', 'WILLR', 'percent'])

        self.data['RSI'] = talib.RSI(self.data['close'])
        self.data['MACD'], self.data['MACD_signal'], self.data['MACD_hist'] = talib.MACD(self.data['close'])
        self.data['CCI'] = talib.CCI(self.data['high'], self.data['low'], self.data['close'], timeperiod=20)
        self.data['ADX'] = talib.ADX(self.data['high'], self.data['low'], self.data['close'], timeperiod=14)
        self.data['STOCH_K'], self.data['STOCH_D'] = talib.STOCH(
            self.data['high'], self.data['low'], self.data['close'],
            fastk_period=14,
            slowk_period=3,
            slowd_period=3,
        )
        self.data['ATR'] = talib.ATR(self.data['high'], self.data['low'], self.data['close'], timeperiod=14)
        # self.data['OBV'] = talib.OBV(self.data['close'], self.data['volume'])
        self.data['MFI'] = talib.MFI(self.data['high'], self.data['low'], self.data['close'],
                                        self.data['volume'], timeperiod=14)
        self.data['WILLR'] = talib.WILLR(self.data['high'], self.data['low'], self.data['close'], timeperiod=14)
        self.data['percent'] = (self.data['close'].shift(-1) / self.data['close']) - 1



    def indi_on(self):
        self.default_indi()
        self.second_indi()
        self.premitiv()

    def premitiv(self):
        self.list_sign.extend(['RSI', 'SMA_100', 'EMA_100', 'SMA_20', 'EMA_20'])

        self.data['RSI'] = talib.RSI(self.data['close'])
        self.data['SMA_20'] = talib.SMA(self.data['close'], 20)
        self.data['EMA_20'] = talib.EMA(self.data['close'], 20)
        self.data['SMA_100'] = talib.SMA(self.data['close'], 100)
        self.data['EMA_100'] = talib.EMA(self.data['close'], 100)

    def default_indi(self):
        self.list_sign.extend(['MACD', 'MACD_signal', 'MACD_hist', 'BB_upper', 'BB_lower', 'BB_middle',
                               'SMA_50', 'SMA_200', 'EMA_50', 'EMA_200',
                               'CCI', 'ADX', 'volatility'])

        # self.data['RSI'] = talib.RSI(self.data['close'], period=14)
        self.data['MACD'], self.data['MACD_signal'], self.data['MACD_hist'] = talib.MACD(self.data['close'])
        self.data['BB_upper'], self.data['BB_middle'], self.data['BB_lower'] = talib.BBANDS(self.data['close'])
        # self.data['SMA_20'] = talib.SMA(self.data['close'], 20)
        # self.data['EMA_20'] = talib.EMA(self.data['close'], 20)
        self.data['SMA_50'] = talib.SMA(self.data['close'], 50)
        self.data['EMA_50'] = talib.EMA(self.data['close'], 50)
        # self.data['SMA_100'] = talib.SMA(self.data['close'], 100)
        # self.data['EMA_100'] = talib.EMA(self.data['close'], 100)
        self.data['SMA_200'] = talib.SMA(self.data['close'], 200)
        self.data['EMA_200'] = talib.EMA(self.data['close'], 200)
        self.data['CCI'] = talib.CCI(self.data['high'], self.data['low'], self.data['close'])
        self.data['ADX'] = talib.ADX(self.data['high'], self.data['low'], self.data['close'])
        self.data['volatility'] = talib.ATR(self.data['high'], self.data['low'], self.data['close'], 14)

    def second_indi(self):
        self.list_sign.extend(['trend_strength', 'momentum', 'vol_ratio', 'price_pos', 'slope', 'candle_body',
                               'upper_shadow', 'lower_shadow', 'returns', 'log_return'])

        self.data['trend_strength'] = self.data['SMA_50'] / self.data['SMA_200'] - 1
        self.data['momentum'] = self.data['close'] / self.data['close'].shift(10) - 1
        self.data['vol_ratio'] = self.data['volume'] / self.data['volume'].rolling(50).mean()
        self.data['price_pos'] = (self.data['close'] - self.data['low'].rolling(100).min()) / (
                    self.data['high'].rolling(100).max() - self.data['low'].rolling(100).min())

        self.data['slope'] = self.data['close'].diff(5)
        self.data['slope'] = self.data['slope'] / self.data['close'].shift(5)

        self.data['candle_body'] = self.data['close'] - self.data['open']
        self.data['upper_shadow'] = self.data['high'] - self.data[['close', 'open']].max(axis=1)
        self.data['lower_shadow'] = self.data[['close', 'open']].min(axis=1) - self.data['low']

        self.data['returns'] = self.data['close'].pct_change()
        self.data['log_return'] = np.log(self.data['close'] / self.data['close'].shift(1))

