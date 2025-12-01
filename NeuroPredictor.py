import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib

import neuro
import table

MODEL = "models/MOEX_model_10min_step_100_pred_20_20000_e50/"

class NeuroPredictor:
    def __init__(self):
        """Загружает модель и все настройки одним файлом"""
        data = joblib.load(f"{MODEL}/model_complete_2.0.1.pkl")
        self.model = tf.keras.models.load_model(f"{MODEL}/model_2.0.1.keras", compile=False)
        self.feature_scaler = data['feature_scaler']
        self.target_scaler = data['target_scaler']
        self.feature_columns = data['feature_columns']
        self.timestep = data['timestep']
        ddd = pd.read_csv(f"BD/SBER_10_NOW.csv")
        self.neuro_brain = table.DataCreate(ddd)

        print(f"✅ Модель загружена: {len(self.feature_columns)} признаков, timestep={self.timestep}")

    def predict(self, new_df):
        """Просто передай DataFrame - получи прогноз"""
        # Предобработка данных
        processed_df = self.neuro_brain.table
        processed_df = processed_df.dropna()

        # Масштабирование признаков
        new_features = processed_df[self.feature_columns]
        new_features_scaled = self.feature_scaler.transform(new_features)

        # Создание последовательностей
        X_pred_seq = self.create_prediction_sequences(new_features_scaled, self.timestep)

        # Прогноз
        probabilities = self.model.predict(X_pred_seq, verbose=0).flatten()

        # Получаем соответствующие цены закрытия
        close_prices = processed_df['close'].values[self.timestep:]

        return close_prices, probabilities

    def create_prediction_sequences(self, data, time_steps):
        """Создает последовательности для прогнозирования"""
        X_pred = []
        for i in range(time_steps, len(data)):
            X_pred.append(data[i - time_steps:i])
        return np.array(X_pred)

    def save_predictions(self, close_prices, probabilities, filename='predictions_results.csv'):
        """Сохраняет результаты прогноза"""
        results_df = pd.DataFrame({
            'close': close_prices,
            'probability_rise': probabilities,
            'predicted_class': (probabilities > 0.2).astype(int)
        })

        results_df.to_csv(filename, index=False)
        print(f"✅ Прогнозы сохранены в {filename}")
        return results_df