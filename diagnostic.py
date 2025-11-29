import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve
import warnings
matplotlib.use('TkAgg')
warnings.filterwarnings('ignore')


class NeuroDiagnostic:
    def __init__(self, results_df, actual_returns=None):
        """
        results_df: DataFrame с колонками ['close', 'probability_rise', 'predicted_class']
        actual_returns: Фактические доходности через 5 свечей (если есть)
        """
        self.results_df = results_df.copy()
        self.actual_returns = actual_returns
        self.diagnosis_report = {}

    def calculate_future_returns(self, future_prices_df, period=5):
        """Рассчитывает фактические доходности если передан DataFrame с будущими ценами"""
        if future_prices_df is not None:
            future_closes = future_prices_df['close'].values
            current_closes = self.results_df['close'].values

            if len(future_closes) >= len(current_closes) + period:
                actual_returns = []
                for i in range(len(current_closes)):
                    if i + period < len(future_closes):
                        future_return = (future_closes[i + period] - current_closes[i]) / current_closes[i]
                        actual_returns.append(1 if future_return > 0 else 0)

                self.actual_returns = np.array(actual_returns[:len(self.results_df)])
                print(f"✅ Рассчитаны фактические доходности для {len(self.actual_returns)} примеров")

    def run_complete_diagnosis(self):
        """Запускает полную диагностику модели"""
        print("🔍 ЗАПУСК ПОЛНОЙ ДИАГНОСТИКИ НЕЙРОСЕТИ")
        print("=" * 60)

        self._basic_statistics()
        self._probability_analysis()
        self._calibration_analysis()
        self._temporal_analysis()
        self._risk_analysis()

        if self.actual_returns is not None:
            self._performance_metrics()
            self._profitability_analysis()

        self._generate_report()

        return self.diagnosis_report

    def _basic_statistics(self):
        """Базовая статистика прогнозов"""
        print("\n📊 1. БАЗОВАЯ СТАТИСТИКА ПРОГНОЗОВ")

        total_predictions = len(self.results_df)
        buy_signals = self.results_df['predicted_class'].sum()
        sell_signals = total_predictions - buy_signals

        stats = {
            'total_predictions': total_predictions,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'buy_ratio': buy_signals / total_predictions,
            'avg_probability': self.results_df['probability_rise'].mean(),
            'prob_std': self.results_df['probability_rise'].std(),
            'prob_min': self.results_df['probability_rise'].min(),
            'prob_max': self.results_df['probability_rise'].max()
        }

        print(f"   Всего прогнозов: {stats['total_predictions']}")
        print(f"   Сигналов на покупку: {stats['buy_signals']} ({stats['buy_ratio']:.1%})")
        print(f"   Сигналов на продажу: {stats['sell_signals']} ({1 - stats['buy_ratio']:.1%})")
        print(f"   Средняя вероятность: {stats['avg_probability']:.3f}")
        print(f"   Волатильность вероятностей: {stats['prob_std']:.3f}")
        print(f"   Диапазон: [{stats['prob_min']:.3f}, {stats['prob_max']:.3f}]")

        self.diagnosis_report['basic_stats'] = stats

        # Визуализация распределения
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.hist(self.results_df['probability_rise'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(0.5, color='red', linestyle='--', label='Порог 50%')
        plt.title('Распределение вероятностей')
        plt.xlabel('Вероятность роста')
        plt.ylabel('Частота')
        plt.legend()

        plt.subplot(1, 3, 2)
        signal_counts = [stats['sell_signals'], stats['buy_signals']]
        plt.pie(signal_counts, labels=['SELL', 'BUY'], autopct='%1.1f%%', colors=['lightcoral', 'lightgreen'])
        plt.title('Соотношение сигналов')

        plt.subplot(1, 3, 3)
        plt.plot(self.results_df['close'], alpha=0.7)
        buy_points = self.results_df[self.results_df['predicted_class'] == 1].index
        plt.scatter(buy_points, self.results_df.loc[buy_points, 'close'],
                    color='green', alpha=0.6, label='BUY сигналы', s=10)
        plt.title('Сигналы на графике цен')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def _probability_analysis(self):
        """Анализ качества вероятностных прогнозов"""
        print("\n🎯 2. АНАЛИЗ КАЧЕСТВА ВЕРОЯТНОСТЕЙ")

        probs = self.results_df['probability_rise']

        # Анализ уверенности модели
        high_confidence = len(probs[probs > 0.7]) + len(probs[probs < 0.3])
        low_confidence = len(probs[(probs >= 0.4) & (probs <= 0.6)])

        confidence_stats = {
            'high_confidence_ratio': high_confidence / len(probs),
            'low_confidence_ratio': low_confidence / len(probs),
            'uncertain_predictions': len(probs[(probs > 0.45) & (probs < 0.55)]),
            'extreme_predictions': len(probs[probs > 0.9]) + len(probs[probs < 0.1])
        }

        print(f"   Высокая уверенность (>0.7 или <0.3): {confidence_stats['high_confidence_ratio']:.1%}")
        print(f"   Низкая уверенность (0.4-0.6): {confidence_stats['low_confidence_ratio']:.1%}")
        print(f"   Неопределенные прогнозы (0.45-0.55): {confidence_stats['uncertain_predictions']}")
        print(f"   Экстремальные прогнозы (>0.9 или <0.1): {confidence_stats['extreme_predictions']}")

        self.diagnosis_report['confidence_analysis'] = confidence_stats

    def _calibration_analysis(self):
        """Анализ калибровки вероятностей"""
        print("\n📈 3. АНАЛИЗ КАЛИБРОВКИ ВЕРОЯТНОСТЕЙ")

        if self.actual_returns is not None:
            prob_true, prob_pred = calibration_curve(self.actual_returns,
                                                     self.results_df['probability_rise'],
                                                     n_bins=10)

            # Идеальная калибровка
            perfect_calibration = np.linspace(0, 1, 10)

            plt.figure(figsize=(10, 6))
            plt.plot(prob_pred, prob_true, 's-', label='Модель')
            plt.plot(perfect_calibration, perfect_calibration, '--', label='Идеальная калибровка')
            plt.xlabel('Предсказанная вероятность')
            plt.ylabel('Фактическая доля положительных')
            plt.title('Калибровочная кривая')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

            # Мера калибровки
            calibration_error = np.mean((prob_true - prob_pred) ** 2)
            print(f"   Ошибка калибровки: {calibration_error:.4f}")

            self.diagnosis_report['calibration_error'] = calibration_error

    def _temporal_analysis(self):
        """Анализ временных паттернов"""
        print("\n⏰ 4. ВРЕМЕННОЙ АНАЛИЗ")

        # Анализ кластеризации сигналов
        signals = self.results_df['predicted_class'].values
        signal_changes = np.diff(signals)
        volatility = np.sum(signal_changes != 0) / len(signal_changes)

        # Поиск длинных серий
        from itertools import groupby
        series_lengths = [len(list(group)) for _, group in groupby(signals)]
        avg_series_length = np.mean(series_lengths)
        max_series_length = max(series_lengths)

        temporal_stats = {
            'signal_volatility': volatility,
            'avg_series_length': avg_series_length,
            'max_series_length': max_series_length,
            'total_series': len(series_lengths)
        }

        print(f"   Волатильность сигналов: {temporal_stats['signal_volatility']:.3f}")
        print(f"   Средняя длина серии: {temporal_stats['avg_series_length']:.1f} свечей")
        print(f"   Максимальная длина серии: {temporal_stats['max_series_length']} свечей")
        print(f"   Всего серий сигналов: {temporal_stats['total_series']}")

        self.diagnosis_report['temporal_analysis'] = temporal_stats

    def _risk_analysis(self):
        """Анализ рисков"""
        print("\n⚠️  5. АНАЛИЗ РИСКОВ")

        probs = self.results_df['probability_rise']

        risk_stats = {
            'high_risk_buy': len(probs[(probs > 0.5) & (probs < 0.6)]),
            'high_risk_sell': len(probs[(probs < 0.5) & (probs > 0.4)]),
            'ambiguous_signals': len(probs[(probs >= 0.48) & (probs <= 0.52)])
        }

        print(f"   Рискованные покупки (0.5-0.6): {risk_stats['high_risk_buy']}")
        print(f"   Рискованные продажи (0.4-0.5): {risk_stats['high_risk_sell']}")
        print(f"   Неоднозначные сигналы (0.48-0.52): {risk_stats['ambiguous_signals']}")

        self.diagnosis_report['risk_analysis'] = risk_stats

    def _performance_metrics(self):
        """Метрики производительности если есть фактические данные"""
        print("\n🏆 6. МЕТРИКИ ПРОИЗВОДИТЕЛЬНОСТИ")

        y_true = self.actual_returns
        y_pred = self.results_df['predicted_class'].values[:len(y_true)]
        y_prob = self.results_df['probability_rise'].values[:len(y_true)]

        # Основные метрики
        cm = confusion_matrix(y_true, y_pred)
        accuracy = np.mean(y_true == y_pred)
        precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0
        recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        try:
            auc_roc = roc_auc_score(y_true, y_prob)
        except:
            auc_roc = 0.5

        performance_stats = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'confusion_matrix': cm.tolist()
        }

        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   Precision: {precision:.3f}")
        print(f"   Recall: {recall:.3f}")
        print(f"   F1-Score: {f1:.3f}")
        print(f"   AUC-ROC: {auc_roc:.3f}")
        print(f"   Матрица ошибок:")
        print(f"      [[TN={cm[0, 0]}, FP={cm[0, 1]}]")
        print(f"       [FN={cm[1, 0]}, TP={cm[1, 1]}]]")

        # ROC кривая
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_roc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        self.diagnosis_report['performance_metrics'] = performance_stats

    def _profitability_analysis(self):
        """Анализ прибыльности"""
        print("\n💰 7. АНАЛИЗ ПРИБЫЛЬНОСТИ")

        if self.actual_returns is not None:
            # Простая стратегия: покупать при сигнале > 0.5
            signals = self.results_df['predicted_class'].values[:len(self.actual_returns)]
            returns = self.actual_returns

            # Доходность стратегии
            strategy_returns = returns * (signals == 1)
            buy_hold_returns = returns  # Buy & Hold для сравнения

            total_strategy_return = np.sum(strategy_returns)
            total_buy_hold_return = np.sum(buy_hold_returns)

            profitability_stats = {
                'total_strategy_return': total_strategy_return,
                'total_buy_hold_return': total_buy_hold_return,
                'excess_return': total_strategy_return - total_buy_hold_return,
                'win_rate': np.mean(strategy_returns > 0),
                'avg_return_per_trade': np.mean(strategy_returns[strategy_returns != 0]) if np.any(
                    strategy_returns != 0) else 0
            }

            print(f"   Общая доходность стратегии: {profitability_stats['total_strategy_return']:.4f}")
            print(f"   Общая доходность Buy&Hold: {profitability_stats['total_buy_hold_return']:.4f}")
            print(f"   Превышение над рынком: {profitability_stats['excess_return']:.4f}")
            print(f"   Win Rate: {profitability_stats['win_rate']:.3f}")
            print(f"   Средняя доходность сделки: {profitability_stats['avg_return_per_trade']:.4f}")

            self.diagnosis_report['profitability'] = profitability_stats

    def _generate_report(self):
        """Генерация итогового отчета с рекомендациями"""
        print("\n" + "=" * 60)
        print("📋 ИТОГОВЫЙ ДИАГНОСТИЧЕСКИЙ ОТЧЕТ")
        print("=" * 60)

        basic = self.diagnosis_report.get('basic_stats', {})
        confidence = self.diagnosis_report.get('confidence_analysis', {})
        temporal = self.diagnosis_report.get('temporal_analysis', {})
        risk = self.diagnosis_report.get('risk_analysis', {})
        performance = self.diagnosis_report.get('performance_metrics', {})
        profitability = self.diagnosis_report.get('profitability', {})

        # Анализ проблем и рекомендации
        issues = []
        recommendations = []

        # Проверка баланса сигналов
        if basic.get('buy_ratio', 0.5) > 0.7 or basic.get('buy_ratio', 0.5) < 0.3:
            issues.append("Дисбаланс сигналов покупки/продажи")
            recommendations.append("Настроить порог классификации или сбалансировать данные")

        # Проверка уверенности
        if confidence.get('low_confidence_ratio', 0) > 0.4:
            issues.append("Много неопределенных прогнозов")
            recommendations.append("Увеличить сложность модели или улучшить признаки")

        # Проверка волатильности сигналов
        if temporal.get('signal_volatility', 0) > 0.5:
            issues.append("Высокая волатильность сигналов")
            recommendations.append("Добавить фильтрацию или сглаживание прогнозов")

        # Проверка производительности
        if performance:
            if performance.get('accuracy', 0) < 0.55:
                issues.append("Низкая точность прогнозов")
                recommendations.append("Пересмотреть признаки или архитектуру модели")

            if performance.get('precision', 0) < 0.5:
                issues.append("Много ложных срабатываний")
                recommendations.append("Повысить порог для сигналов покупки")

        # Вывод проблем
        if issues:
            print("\n🚨 ВЫЯВЛЕННЫЕ ПРОБЛЕМЫ:")
            for i, issue in enumerate(issues, 1):
                print(f"   {i}. {issue}")

        # Рекомендации
        if recommendations:
            print("\n💡 РЕКОМЕНДАЦИИ ПО УЛУЧШЕНИЮ:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")

        if not issues:
            print("\n✅ Критических проблем не обнаружено!")

        print(f"\n📊 ОБЩАЯ ОЦЕНКА: {'⚠️ ТРЕБУЕТСЯ ДОРАБОТКА' if issues else '✅ СТАБИЛЬНАЯ РАБОТА'}")

        # Сохраняем полный отчет
        self.diagnosis_report['issues'] = issues
        self.diagnosis_report['recommendations'] = recommendations

        return self.diagnosis_report


# Пример использования
if __name__ == "__main__":
    # Загрузите ваш results_df
    results_df = pd.read_csv('predictions_results.csv')

    # Создайте диагностику
    diagnostic = NeuroDiagnostic(results_df)

    # Запустите диагностику
    report = diagnostic.run_complete_diagnosis()

    print("📋 Загрузите ваш results_df и запустите диагностику!")