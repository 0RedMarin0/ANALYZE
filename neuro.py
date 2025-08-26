import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import time
from typing import Tuple, List

def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Расчет RSI индикатора"""
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0.0)
    loss = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.zeros_like(prices)
    avg_loss = np.zeros_like(prices)

    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])

    for i in range(period + 1, len(prices)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i - 1]) / period

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    return rsi


def prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    """Подготовка и очистка данных"""
    numeric_columns = ['open', 'close', 'high', 'low', 'value', 'volume']
    for col in numeric_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    data = data.dropna()

    data['rsi_14'] = calculate_rsi(data['close'].values)
    data['price_change_pct'] = data['close'].pct_change() * 100
    data['volume_change_pct'] = data['volume'].pct_change() * 100
    data['high_low_spread'] = (data['high'] - data['low']) / data['low'] * 100
    data['close_open_spread'] = (data['close'] - data['open']) / data['open'] * 100

    data['sma_20'] = data['close'].rolling(window=20).mean()
    data['sma_50'] = data['close'].rolling(window=50).mean()
    data['ema_12'] = data['close'].ewm(span=12).mean()

    data = data.dropna().reset_index(drop=True)

    return data


def create_sequences(data: np.ndarray, seq_length: int, target_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Создание последовательностей для обучения"""
    X, y = [], []
    n_samples = len(data) - seq_length - target_length + 1

    for i in range(n_samples):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length:i + seq_length + target_length, 0])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ==================== АРХИТЕКТУРА МОДЕЛИ ====================

def create_advanced_model(seq_length: int, num_features: int, target_length: int) -> tf.keras.Model:
    """Создание модели LSTM"""
    model = tf.keras.Sequential([
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                256,
                return_sequences=True,
                kernel_initializer='he_normal',
                recurrent_initializer='orthogonal',
                dropout=0.1,
                recurrent_dropout=0.1
            ),
            input_shape=(seq_length, num_features),
            merge_mode='concat'
        ),

        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                128,
                return_sequences=True,
                kernel_initializer='he_normal',
                recurrent_initializer='orthogonal',
                dropout=0.1,
                recurrent_dropout=0.1
            ),
            merge_mode='concat'
        ),

        tf.keras.layers.LSTM(
            64,
            kernel_initializer='he_normal',
            recurrent_initializer='orthogonal',
            dropout=0.1,
            recurrent_dropout=0.1
        ),

        tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal'),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_normal'),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(32, activation='relu', kernel_initializer='he_normal'),
        tf.keras.layers.Dropout(0.1),

        tf.keras.layers.Dense(target_length, dtype='float32')
    ])

    return model


# ==================== DATA GENERATOR ====================

class StockDataGenerator(tf.keras.utils.Sequence):
    """Генератор данных"""

    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32, shuffle: bool = True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(X))
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, index: int) -> Tuple[tf.Tensor, tf.Tensor]:
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        X_batch = tf.convert_to_tensor(self.X[batch_indices], dtype=tf.float32)
        y_batch = tf.convert_to_tensor(self.y[batch_indices], dtype=tf.float32)
        return X_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


# ==================== КОМПИЛЯЦИЯ И ОБУЧЕНИЕ ====================

def compile_model(model: tf.keras.Model) -> tf.keras.Model:
    """Компиляция модели"""
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.Huber(),
        metrics=['mae', 'mse', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
    )

    return model


def get_callbacks() -> List[tf.keras.callbacks.Callback]:
    """Callbacks"""
    return [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.ModelCheckpoint('best_stock_model.h5', monitor='val_loss', save_best_only=True, verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1)
    ]


# ==================== ПРЕДСКАЗАНИЯ И ОЦЕНКА ====================

@tf.function
def fast_predict(model: tf.keras.Model, data: tf.Tensor) -> tf.Tensor:
    return model(data, training=False)


def predict_future_prices(model: tf.keras.Model, last_sequence: np.ndarray,
                          target_length: int, scaler: MinMaxScaler,
                          feature_means: np.ndarray) -> np.ndarray:
    current_sequence = last_sequence.copy()
    predictions = []

    for _ in range(target_length):
        pred_tensor = fast_predict(model, tf.expand_dims(current_sequence, 0))
        next_price = pred_tensor[0, 0].numpy()
        new_row = np.zeros((1, current_sequence.shape[1]))
        new_row[0, 0] = next_price
        new_row[0, 1:] = feature_means[1:]
        current_sequence = np.vstack([current_sequence[1:], new_row])
        predictions.append(next_price)

    return np.array(predictions)


def evaluate_model(model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray, scaler: MinMaxScaler) -> dict:
    y_pred = model.predict(X_test, verbose=0, batch_size=64)
    y_test_actual = y_test * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]
    y_pred_actual = y_pred * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]

    metrics = {
        'mae': mean_absolute_error(y_test_actual, y_pred_actual),
        'mse': mean_squared_error(y_test_actual, y_pred_actual),
        'rmse': np.sqrt(mean_squared_error(y_test_actual, y_pred_actual)),
        'mape': np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100
    }

    return metrics, y_test_actual, y_pred_actual


# ==================== ВИЗУАЛИЗАЦИЯ ====================

def plot_results(history: tf.keras.callbacks.History, y_test: np.ndarray, y_pred: np.ndarray, metrics: dict):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    axes[0, 0].plot(history.history['loss'], label='Train Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Learning Curve')
    axes[0, 0].legend()

    axes[0, 1].plot(history.history['mae'], label='Train MAE')
    axes[0, 1].plot(history.history['val_mae'], label='Validation MAE')
    axes[0, 1].set_title('MAE Curve')
    axes[0, 1].legend()

    sample_idx = np.random.randint(0, len(y_test))
    axes[1, 0].plot(y_test[sample_idx], label='Actual', marker='o')
    axes[1, 0].plot(y_pred[sample_idx], label='Predicted', marker='x')
    axes[1, 0].set_title(f'Прогноз на 30 свечей (Пример #{sample_idx})')
    axes[1, 0].legend()

    metrics_text = "\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12, va='center')
    axes[1, 1].set_title('Метрики качества')
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300)
    plt.show()


# ==================== ОСНОВНАЯ ФУНКЦИЯ ====================

def main():
    print("🚀 ЗАПУСК МОДЕЛИ НА CPU")
    print("=" * 60)

    SEQ_LENGTH = 60
    TARGET_LENGTH = 30
    BATCH_SIZE = 32
    EPOCHS = 100

    data = pd.read_csv('data10.csv')
    data = prepare_data(data)

    features = ['close', 'rsi_14', 'volume_change_pct', 'high_low_spread', 'close_open_spread', 'sma_20', 'ema_12']

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features])

    feature_means = np.mean(scaled_data, axis=0)

    X, y = create_sequences(scaled_data, SEQ_LENGTH, TARGET_LENGTH)

    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"📈 Train: {len(X_train)}, Test: {len(X_test)}")

    train_generator = StockDataGenerator(X_train, y_train, BATCH_SIZE)
    test_generator = StockDataGenerator(X_test, y_test, BATCH_SIZE, shuffle=False)

    model = create_advanced_model(SEQ_LENGTH, len(features), TARGET_LENGTH)
    model = compile_model(model)
    model.summary()

    start_time = time.time()

    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=test_generator,
        callbacks=get_callbacks(),
        verbose=1
    )

    print(f"✅ Обучение завершено за {time.time() - start_time:.2f} секунд")

    metrics, y_test_actual, y_pred_actual = evaluate_model(model, X_test, y_test, scaler)

    print("\n📈 МЕТРИКИ КАЧЕСТВА:")
    for metric, value in metrics.items():
        print(f"   {metric.upper()}: {value:.6f}")

    plot_results(history, y_test_actual, y_pred_actual, metrics)

    model.save('stock_price_predictor_final.h5')
    print("💾 Модель сохранена как 'stock_price_predictor_final.h5'")

    last_sequence = X_test[-1]
    future_predictions = predict_future_prices(model, last_sequence, TARGET_LENGTH, scaler, feature_means)
    future_prices_actual = future_predictions * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]

    print("\n🔮 ПРИМЕР ПРОГНОЗА:")
    for i, price in enumerate(future_prices_actual, 1):
        print(f"  Свеча {i:2d}: {price:.2f}")

    return model, scaler, history, metrics


if __name__ == "__main__":
    model, scaler, history, metrics = main()
    print("\n" + "=" * 60)
    print("🎉 ПРОГРАММА УСПЕШНО ЗАВЕРШЕНА!")
    print("=" * 60)
