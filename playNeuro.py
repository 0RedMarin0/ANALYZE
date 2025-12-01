import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import table

MODEL_NAME = "models/MOEX_model_10min_step_100_pred_20_20000_e50/model_2.0.2.keras"
FILE_NAME = 'BD/SBER_10_NOW.csv'
TIMESTEP = 100
PREDICTION = -20
VOLUME_DATA = 20000
NAME_PNG = f"predictions_indicator_model.png"

model = tf.keras.models.load_model(MODEL_NAME, compile=False)

new_df = pd.read_csv(FILE_NAME).tail(5000)

df0 = table.DataCreate(new_df)
df = df0.table
print(len(df))
df["target"] = df["RSI"].shift(PREDICTION)

feature_columns = df0.list_sign
features = df[feature_columns]
target = df['target']

def create_prediction_sequences(data, time_steps=100):
    X_pred = []
    for i in range(time_steps, len(data)):
        X_pred.append(data[i-time_steps:i])
    return np.array(X_pred)

X_pred_seq = create_prediction_sequences(features.values, TIMESTEP)
probabilities = model.predict(X_pred_seq, verbose=1).flatten()

close_prices = df['close'].values[TIMESTEP:]
results_df = pd.DataFrame({
    'close': close_prices,
    'rsi_prediction': probabilities,
    'rsi_actual': target.values[TIMESTEP:]
})

results_df.to_csv('predictions_rsi_results.csv', index=False)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 10), sharex=True)

ax1.plot(results_df.index, results_df['close'], color='blue', linewidth=2)
ax1.set_title('Цены закрытия')
ax1.set_ylabel('Цена')
ax1.grid(True, alpha=0.3)

# ax2.plot(results_df.index, results_df['rsi_actual']*100, label='Фактический RSI', alpha=0.7)
ax2.plot(results_df.index, (results_df['rsi_prediction'] + 1) * 50, label='Прогноз RSI', alpha=0.7)
ax2.set_title('RSI: Прогноз vs Факт')
ax2.set_ylabel('RSI chek')
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3.plot(results_df.index, df['RSI'].values[TIMESTEP:], label='RSI', alpha=0.7)
ax3.set_title('RSI')
ax3.set_ylabel('RSI')
ax3.legend()
ax3.grid(True, alpha=0.3)

ax4.plot(results_df.index, results_df['rsi_actual'], label='RSI -20', alpha=0.7)
ax4.set_title('RSI -20')
ax4.set_ylabel('RSI - 20')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(NAME_PNG, dpi=300, bbox_inches='tight')
plt.show()

print("Прогнозы сохранены в predictions_rsi_results.csv")