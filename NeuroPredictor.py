import numpy as np
import tensorflow as tf

print("=== ФИНАЛЬНЫЙ ТЕСТ С ЦЕНТРИРОВАНИЕМ ===")

# 1. Данные с ЦЕНТРИРОВАНИЕМ
n_samples = 100
simple_data = np.array([[i, i, i, i, i] for i in range(1, n_samples + 1)], dtype=float)
simple_target = np.array([i + 1 for i in range(1, n_samples)], dtype=float)

# ЦЕНТРИРУЕМ!
data_mean = np.mean(simple_data)
data_std = np.std(simple_data)
simple_data_centered = (simple_data - data_mean) / data_std
simple_target_centered = (simple_target - data_mean) / data_std

print(f"До центрирования: mean={data_mean:.2f}, std={data_std:.2f}")
print(f"После: mean={np.mean(simple_data_centered):.2f}, std={np.std(simple_data_centered):.2f}")

# 2. Последовательности
TIMESTEP = 1
X_seq, y_seq = [], []
for i in range(TIMESTEP, len(simple_data_centered) - 1):
    X_seq.append(simple_data_centered[i - TIMESTEP:i])
    y_seq.append(simple_target_centered[i - 1])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# 3. Модель
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(TIMESTEP, 5)),
    tf.keras.layers.Dense(1, kernel_initializer='zeros', bias_initializer='zeros')
])

# МЕНЬШИЙ learning rate!
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
    loss='mse'
)

print(f"\nНачальные веса: {model.layers[1].get_weights()[0].flatten()}")
print(f"Начальный bias: {model.layers[1].get_weights()[1][0]}")

# 4. Обучение с validation
print("\n=== ОБУЧЕНИЕ ===")
history = model.fit(
    X_seq, y_seq,
    epochs=100,  # БОЛЬШЕ эпох!
    batch_size=16,
    validation_split=0.2,
    verbose=0
)

print(f"Final loss: {history.history['loss'][-1]:.6f}")

# 5. Веса после обучения
weights, bias = model.layers[1].get_weights()
print(f"\nПосле обучения:")
print(f"Веса: {weights.flatten().round(4)}")
print(f"Bias: {bias[0]:.6f}")

# 6. ОБРАТНОЕ ПРЕОБРАЗОВАНИЕ для проверки
print("\n=== ПРОВЕРКА С ОБРАТНЫМ ПРЕОБРАЗОВАНИЕМ ===")

# Тестовый пример: x=2 → должен быть y=3
test_x = np.array([2, 2, 2, 2, 2], dtype=float)
test_x_centered = (test_x - data_mean) / data_std
test_input = test_x_centered.reshape(1, 1, 5)

# Предсказание
pred_centered = model.predict(test_input, verbose=0)[0][0]

# Обратное преобразование
pred_original = pred_centered * data_std + data_mean

print(f"\nТест: x={test_x[0]}, ожидаемый y={test_x[0] + 1}")
print(f"Предсказание: {pred_original:.4f}")
print(f"Ошибка: {abs(pred_original - (test_x[0] + 1)):.4f}")

# 7. Проверка нескольких примеров
print("\n=== ТЕСТ НЕСКОЛЬКИХ ПРИМЕРОВ ===")
test_values = [2, 10, 50, 100]

for x in test_values:
    test_x = np.array([x, x, x, x, x], dtype=float)
    test_x_centered = (test_x - data_mean) / data_std
    test_input = test_x_centered.reshape(1, 1, 5)

    pred_centered = model.predict(test_input, verbose=0)[0][0]
    pred_original = pred_centered * data_std + data_mean

    print(f"x={x:3d}: предсказано {pred_original:6.2f}, должно быть {x + 1:3d}, "
          f"ошибка {abs(pred_original - (x + 1)):5.2f}")