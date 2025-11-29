import pandas as pd
import numpy as np
import talib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import joblib

# Конфигурация распределенных вычислений
strategy = tf.distribute.MultiWorkerMirroredStrategy()

DATA_FILE = 'BD/SBER_10.csv'
MODEL_FILE = 'model_price_forecast.h5'
SCALER_X_FILE = 'scaler_x.pkl'
SCALER_Y_FILE = 'scaler_y.pkl'
FORECAST_HORIZON = 20
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 100
BATCH_SIZE = 64
PATIENCE = 15

df = pd.read_csv(DATA_FILE)

df['RSI'] = talib.RSI(df['close'], timeperiod=14)
df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
df['SMA_20'] = talib.SMA(df['close'], timeperiod=20)
df['EMA_20'] = talib.EMA(df['close'], timeperiod=20)
df['BBANDS_upper'], df['BBANDS_middle'], df['BBANDS_lower'] = talib.BBANDS(df['close'], timeperiod=20)
df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
df['CCI'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
df['MOM'] = talib.MOM(df['close'], timeperiod=10)
df['OBV'] = talib.OBV(df['close'], df['volume'])

df['target'] = (df['close'].shift(-FORECAST_HORIZON) / df['close']) - 1

df = df.dropna()

feature_columns = ['open', 'close', 'high', 'low', 'volume',
                   'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
                   'SMA_20', 'EMA_20', 'BBANDS_upper', 'BBANDS_middle', 'BBANDS_lower',
                   'ATR', 'ADX', 'CCI', 'MOM', 'OBV']

X = df[feature_columns].values
y = df['target'].values.reshape(-1, 1)

scaler_x = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_x.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=TEST_SIZE, random_state=RANDOM_STATE,
                                                    shuffle=False)

# Создание модели внутри стратегии распределения
with strategy.scope():
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

callback_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
    callbacks.ModelCheckpoint(MODEL_FILE, monitor='val_loss', save_best_only=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7)
]

model.fit(X_train, y_train,
          validation_data=(X_test, y_test),
          epochs=EPOCHS,
          batch_size=BATCH_SIZE * strategy.num_replicas_in_sync,
          callbacks=callback_list,
          verbose=1)

joblib.dump(scaler_x, SCALER_X_FILE)
joblib.dump(scaler_y, SCALER_Y_FILE)