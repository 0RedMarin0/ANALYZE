import pickle

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
matplotlib.use('TkAgg')


# ДОБАВЬ эти функции вместо sklearn:
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0


class StandardScaler:
    """Простая реализация StandardScaler"""

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        return self

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        return (X - self.mean_) / (self.scale_ + 1e-8)

    def inverse_transform(self, X):
        return (X * self.scale_) + self.mean_


class PredictionAnalyzer:
    def __init__(self, model, feature_columns):
        self.model = model
        self.feature_columns = feature_columns

    def comprehensive_analysis(self, X_test, y_test, y_train, n_samples=1000):
        """
        Комплексный анализ качества прогнозов
        """
        print("=" * 70)
        print("🔍 ДЕТАЛЬНЫЙ АНАЛИЗ ПРОГНОЗОВ МОДЕЛИ")
        print("=" * 70)

        # Делаем прогнозы
        print("🤖 Выполняю прогнозирование...")
        predictions = self.model.predict(X_test, verbose=0).flatten()

        # Берем подвыборку для наглядности
        sample_idx = np.random.choice(len(y_test), min(n_samples, len(y_test)), replace=False)
        y_sample = y_test[sample_idx]
        pred_sample = predictions[sample_idx]

        # 1. Базовые метрики качества
        self._print_basic_metrics(y_sample, pred_sample)

        # 2. Сравнение с наивными прогнозами
        self._compare_with_naive(y_sample, pred_sample, y_train)

        # 3. Анализ направлений
        self._analyze_directions(y_sample, pred_sample)

        # 4. Статистика ошибок
        self._error_statistics(y_sample, pred_sample)

        # 5. Анализ распределений
        self._distribution_analysis(y_sample, pred_sample)

        # 6. Детальная визуализация
        self._create_detailed_plots(y_sample, pred_sample, X_test[sample_idx])

        # 7. Диагностика проблем
        self._diagnose_problems(y_sample, pred_sample)

        print("=" * 70)

    def _print_basic_metrics(self, y_true, y_pred):
        """Базовые метрики качества"""
        print("\n📊 БАЗОВЫЕ МЕТРИКИ КАЧЕСТВА:")

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        print(f"MAE:  {mae:.6f}")
        print(f"MSE:  {mse:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"R²:   {r2:.4f}")

        # Интерпретация R²
        if r2 > 0.7:
            print("✅ R² > 0.7 - Отличное качество!")
        elif r2 > 0.5:
            print("⚠️  R² > 0.5 - Хорошее качество")
        elif r2 > 0.3:
            print("🔶 R² > 0.3 - Удовлетворительное")
        else:
            print("❌ R² < 0.3 - Низкое качество")

    def _compare_with_naive(self, y_true, y_pred, y_train):
        """Сравнение с наивными прогнозами"""
        print("\n📈 СРАВНЕНИЕ С НАИВНЫМИ ПРОГНОЗАМИ:")

        mae_model = mean_absolute_error(y_true, y_pred)

        # Наивный прогноз 1: последнее значение
        naive_last = np.roll(y_true, 1)
        naive_last[0] = y_true[0]  # Первый элемент
        mae_naive_last = mean_absolute_error(y_true, naive_last)

        # Наивный прогноз 2: среднее значение
        naive_mean = np.full_like(y_true, np.mean(y_train))
        mae_naive_mean = mean_absolute_error(y_true, naive_mean)

        # Наивный прогноз 3: случайное значение
        naive_random = np.random.normal(np.mean(y_train), np.std(y_train), len(y_true))
        mae_naive_random = mean_absolute_error(y_true, naive_random)

        print(f"MAE модели:          {mae_model:.6f}")
        print(f"MAE (последнее знач): {mae_naive_last:.6f}")
        print(f"MAE (среднее):       {mae_naive_mean:.6f}")
        print(f"MAE (случайное):     {mae_naive_random:.6f}")

        # Процент улучшения
        improvement_vs_last = ((mae_naive_last - mae_model) / mae_naive_last) * 100
        improvement_vs_mean = ((mae_naive_mean - mae_model) / mae_naive_mean) * 100

        print(f"\nУлучшение над 'последним значением': {improvement_vs_last:+.1f}%")
        print(f"Улучшение над 'средним значением':   {improvement_vs_mean:+.1f}%")

        if improvement_vs_last > 0:
            print("✅ Модель лучше наивных прогнозов!")
        else:
            print("❌ Модель хуже наивных прогнозов!")

    def _analyze_directions(self, y_true, y_pred):
        """Анализ правильности направлений"""
        print("\n🎯 АНАЛИЗ НАПРАВЛЕНИЙ:")

        # Изменения между последовательными точками
        actual_changes = np.diff(y_true)
        predicted_changes = np.diff(y_pred)

        # Правильные направления
        correct_directions = np.sign(actual_changes) == np.sign(predicted_changes)
        direction_accuracy = np.mean(correct_directions) * 100

        # Подсчет по категориям
        positive_correct = np.sum((actual_changes > 0) & (predicted_changes > 0))
        negative_correct = np.sum((actual_changes < 0) & (predicted_changes < 0))
        zero_correct = np.sum((actual_changes == 0) & (predicted_changes == 0))

        total_changes = len(actual_changes)

        print(f"Общая точность направлений: {direction_accuracy:.1f}%")
        print(f"Правильно предсказано ростов:  {positive_correct}/{np.sum(actual_changes > 0)}")
        print(f"Правильно предсказано падений: {negative_correct}/{np.sum(actual_changes < 0)}")
        print(f"Случайное угадывание: 50.0%")

        if direction_accuracy > 60:
            print("✅ Отличная точность направлений!")
        elif direction_accuracy > 55:
            print("⚠️  Хорошая точность направлений")
        elif direction_accuracy > 50:
            print("🔶 Слабая, но есть сигнал")
        else:
            print("❌ Точность хуже случайного угадывания!")

    def _error_statistics(self, y_true, y_pred):
        """Статистика ошибок"""
        print("\n📉 СТАТИСТИКА ОШИБОК:")

        errors = y_true - y_pred

        print(f"Средняя ошибка:     {np.mean(errors):.6f} (должна быть ~0)")
        print(f"Std ошибок:         {np.std(errors):.6f}")
        print(f"Медианная ошибка:   {np.median(errors):.6f}")
        print(f"Max положительная:  {np.max(errors):.6f}")
        print(f"Max отрицательная:  {np.min(errors):.6f}")
        print(f"MAPE:               {np.mean(np.abs(errors / (y_true + 1e-8))) * 100:.2f}%")

        # Анализ смещения
        mean_error = np.mean(errors)
        if abs(mean_error) > 0.01 * np.std(y_true):
            print(f"⚠️  Обнаружено смещение: {mean_error:.6f}")
        else:
            print("✅ Смещение в пределах нормы")

    def _distribution_analysis(self, y_true, y_pred):
        """Анализ распределений"""
        print("\n📊 СРАВНЕНИЕ РАСПРЕДЕЛЕНИЙ:")

        print(f"Std реальных значений: {np.std(y_true):.6f}")
        print(f"Std прогнозов:         {np.std(y_pred):.6f}")
        print(f"Среднее реальных:      {np.mean(y_true):.6f}")
        print(f"Среднее прогнозов:     {np.mean(y_pred):.6f}")
        print(f"Медиана реальных:      {np.median(y_true):.6f}")
        print(f"Медиана прогнозов:     {np.median(y_pred):.6f}")

        # Коэффициент вариации
        cv_real = np.std(y_true) / (np.mean(y_true) + 1e-8)
        cv_pred = np.std(y_pred) / (np.mean(y_pred) + 1e-8)
        print(f"CV реальных: {cv_real:.4f}")
        print(f"CV прогнозов: {cv_pred:.4f}")

    def _create_detailed_plots(self, y_true, y_pred, X_sample):
        """Создание детальных графиков"""
        fig = plt.figure(figsize=(20, 15))

        # 1. Реальные vs Прогнозные значения
        plt.subplot(3, 3, 1)
        plt.scatter(y_true, y_pred, alpha=0.6, s=20)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        plt.xlabel('Реальные значения')
        plt.ylabel('Прогнозы')
        plt.title('Реальные vs Прогнозы')
        plt.grid(True, alpha=0.3)

        # 2. Временной ряд (первые 100 точек)
        plt.subplot(3, 3, 2)
        n_plot = min(100, len(y_true))
        plt.plot(range(n_plot), y_true[:n_plot], label='Реальные', linewidth=2)
        plt.plot(range(n_plot), y_pred[:n_plot], label='Прогнозы', linewidth=1.5)
        plt.xlabel('Время')
        plt.ylabel('Значение')
        plt.title('Временной ряд (первые 100 точек)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 3. Распределение ошибок
        plt.subplot(3, 3, 3)
        errors = y_true - y_pred
        plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--')
        plt.xlabel('Ошибка')
        plt.ylabel('Частота')
        plt.title('Распределение ошибок')
        plt.grid(True, alpha=0.3)

        # 4. Распределение реальных и прогнозных значений
        plt.subplot(3, 3, 4)
        plt.hist(y_true, bins=50, alpha=0.5, label='Реальные', edgecolor='black')
        plt.hist(y_pred, bins=50, alpha=0.5, label='Прогнозы', edgecolor='black')
        plt.xlabel('Значение')
        plt.ylabel('Частота')
        plt.title('Сравнение распределений')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 5. Кумулятивная ошибка
        plt.subplot(3, 3, 5)
        cumulative_error = np.cumsum(errors)
        plt.plot(cumulative_error)
        plt.xlabel('Время')
        plt.ylabel('Кумулятивная ошибка')
        plt.title('Кумулятивная ошибка прогноза')
        plt.grid(True, alpha=0.3)

        # 6. Анализ остатков
        plt.subplot(3, 3, 6)
        plt.scatter(y_pred, errors, alpha=0.6, s=20)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('Прогнозы')
        plt.ylabel('Ошибки')
        plt.title('Остатки vs Прогнозы')
        plt.grid(True, alpha=0.3)

        # 7. QQ-plot для нормальности ошибок
        plt.subplot(3, 3, 7)
        # Простая гистограмма вместо QQ-plot
        plt.hist(errors, bins=30, alpha=0.7, density=True, edgecolor='black')
        plt.xlabel('Ошибки')
        plt.ylabel('Плотность')
        plt.title('Распределение ошибок')
        plt.grid(True, alpha=0.3)

        # 8. Корреляция признаков с ошибками
        plt.subplot(3, 3, 8)
        feature_errors = []
        for i in range(min(5, X_sample.shape[2])):  # Первые 5 признаков
            correlation = np.corrcoef(X_sample[:, -1, i].flatten(), errors)[0, 1]
            feature_errors.append(abs(correlation))

        plt.bar(range(len(feature_errors)), feature_errors)
        plt.xlabel('Признак')
        plt.ylabel('|Корреляция с ошибкой|')
        plt.title('Корреляция признаков с ошибками')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('detailed_prediction_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def _diagnose_problems(self, y_true, y_pred):
        """Диагностика проблем модели"""
        print("\n🔧 ДИАГНОСТИКА ПРОБЛЕМ:")

        errors = y_true - y_pred
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        problems = []

        # Проверка 1: Смещение
        if abs(np.mean(errors)) > 0.02 * np.std(y_true):
            problems.append("❌ Систематическое смещение прогнозов")

        # Проверка 2: Низкая дисперсия прогнозов
        if np.std(y_pred) < 0.5 * np.std(y_true):
            problems.append("❌ Прогнозы имеют слишком низкую дисперсию")

        # Проверка 3: Плохая корреляция
        correlation = np.corrcoef(y_true, y_pred)[0, 1]
        if correlation < 0.3:
            problems.append("❌ Низкая корреляция с реальными значениями")

        # Проверка 4: Низкое R²
        if r2 < 0.3:
            problems.append("❌ Низкое R² - модель плохо объясняет дисперсию")

        # Проверка 5: Большие ошибки
        if mae > 0.5 * np.std(y_true):
            problems.append("❌ Слишком большие ошибки прогнозирования")

        if problems:
            print("Обнаружены проблемы:")
            for problem in problems:
                print(f"  {problem}")
        else:
            print("✅ Критических проблем не обнаружено")

        # Рекомендации
        print("\n💡 РЕКОМЕНДАЦИИ:")
        if r2 < 0.5:
            print("  • Улучшите архитектуру модели")
            print("  • Добавьте больше признаков")
            print("  • Увеличьте объем тренировочных данных")

        if np.std(y_pred) < 0.5 * np.std(y_true):
            print("  • Уменьшите регуляризацию")
            print("  • Увеличьте learning rate")
            print("  • Попробуйте другую архитектуру")


# Использование:
def test_model_predictions():
    """Тестирование прогнозов модели"""

    # Загрузка данных
    new_df = pd.read_csv("BD/SBER_10_NOW.csv").iloc[:5000]
    print(f"Загружено данных: {len(new_df)} строк")

    # Создание нейросети и фич
    import neuro as ne
    neuro = ne.NeuroBrain()
    df = neuro.data_create(new_df)

    # Создание целевой переменной
    df["target"] = (df["close"].shift(-5) / df["close"]) - 1
    df = df.dropna()

    print(f"Данные после создания фич: {len(df)} строк")

    # Подготовка фич и target
    features = df[neuro.feature_columns]
    target = df['target']

    # Масштабирование фич
    feature_scaler = StandardScaler()
    features_scaled = feature_scaler.fit_transform(features)

    # Создание последовательностей
    X_seq, y_seq = neuro.create_sequences(features_scaled, target.values, 100)

    print(f"Создано последовательностей: {X_seq.shape}")

    # Разделение на train/test
    train_size = int(0.85 * len(X_seq))  # 85% для обучения, 15% для теста

    X_train = X_seq[:train_size]
    y_train = y_seq[:train_size]

    X_test = X_seq[train_size:]
    y_test = y_seq[train_size:]

    print(f"\n=== РАЗДЕЛЕНИЕ ДАННЫХ ===")
    print(f"Train: {X_train.shape} ({len(X_train)} samples)")
    print(f"Test:  {X_test.shape} ({len(X_test)} samples)")

    # Загрузка обученной модели
    model = tf.keras.models.load_model("models/model_10min_step_100_pred_5_100000_2.5.keras")

    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1))
    # Проверка архитектуры модели
    print(f"Архитектура модели: {model.input_shape} -> {model.output_shape}")

    # Проверка предсказаний перед анализом
    print("\n=== БЫСТРАЯ ПРОВЕРКА МОДЕЛИ ===")
    test_predictions_scaled = model.predict(X_test[:5], verbose=0)
    test_predictions = target_scaler.inverse_transform(test_predictions_scaled.reshape(-1, 1)).flatten()
    print("Примеры предсказаний:")
    for i, (true, pred) in enumerate(zip(y_test[:5], test_predictions.flatten())):
        print(f"  Sample {i}: True={true:.6f}, Pred={pred:.6f}, Error={abs(true - pred):.6f}")

    # Запуск анализатора
    print(f"\n=== ЗАПУСК АНАЛИЗАТОРА ПРОГНОЗОВ ===")
    analyzer = PredictionAnalyzer(model, neuro.feature_columns)
    test_predictions_scaled = model.predict(X_test, verbose=0)
    test_predictions = target_scaler.inverse_transform(test_predictions_scaled.reshape(-1, 1)).flatten()

    analyzer.comprehensive_analysis(
        X_test=X_test,
        y_test=y_test,  # оригинальные y_test
        y_train=y_train_scaled,  # оригинальные y_train
        n_samples=min(2000, len(X_test))
    )

# Добавьте этот вызов после обучения модели:
test_model_predictions()