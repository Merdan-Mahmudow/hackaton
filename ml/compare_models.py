"""Сравнение эффективности Baseline и RuBERT моделей по Macro F1-score."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd


def load_baseline_metrics(metadata_path: Path) -> Optional[Dict]:
    """Загружает метрики baseline модели из metadata.json."""
    if not metadata_path.exists():
        return None
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        metrics = metadata.get('metrics', {})
        macro_f1 = metrics.get('macro avg', {}).get('f1-score', None)
        accuracy = metrics.get('accuracy', None)
        
        if macro_f1 is None:
            return None
        
        return {
            'model_name': 'Baseline (TF-IDF + Logistic Regression)',
            'macro_f1': macro_f1,
            'accuracy': accuracy,
            'metrics': metrics,
            'model_path': metadata.get('model_path', ''),
            'algorithm': metadata.get('algorithm', ''),
            'vectorizer': metadata.get('vectorizer', ''),
        }
    except Exception as e:
        print(f"Ошибка при загрузке метрик baseline: {e}")
        return None


def load_transformer_metrics(metrics_path: Path) -> Optional[Dict]:
    """Загружает метрики transformer модели из transformer_metrics.json."""
    if not metrics_path.exists():
        return None
    
    try:
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        
        macro_f1 = metrics.get('eval_macro_f1', None)
        accuracy = metrics.get('eval_accuracy', None)
        
        if macro_f1 is None:
            return None
        
        # Пытаемся загрузить информацию о модели из metadata
        metadata_path = Path("models/transformer/metadata.json")
        base_model = "cointegrated/rubert-tiny"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                base_model = metadata.get('base_model', base_model)
            except Exception:
                pass
        
        return {
            'model_name': f'RuBERT ({base_model.split("/")[-1]})',
            'macro_f1': macro_f1,
            'accuracy': accuracy,
            'metrics': metrics,
            'model_path': str(metadata_path.parent),
            'base_model': base_model,
        }
    except Exception as e:
        print(f"Ошибка при загрузке метрик transformer: {e}")
        return None


def compare_models(
    baseline_metadata_path: Path = Path("models/metadata.json"),
    transformer_metrics_path: Path = Path("reports/transformer_metrics.json"),
    output_path: Optional[Path] = None,
) -> Dict:
    """Сравнивает модели Baseline и RuBERT по Macro F1-score."""
    
    baseline_metrics = load_baseline_metrics(baseline_metadata_path)
    transformer_metrics = load_transformer_metrics(transformer_metrics_path)
    
    if baseline_metrics is None:
        print("⚠️ Не удалось загрузить метрики Baseline модели")
        return {}
    
    if transformer_metrics is None:
        print("⚠️ Не удалось загрузить метрики RuBERT модели")
        return {}
    
    # Создаем сравнение
    comparison = {
        'baseline': baseline_metrics,
        'transformer': transformer_metrics,
        'comparison': {
            'macro_f1_diff': transformer_metrics['macro_f1'] - baseline_metrics['macro_f1'],
            'macro_f1_improvement_percent': (
                (transformer_metrics['macro_f1'] - baseline_metrics['macro_f1']) 
                / baseline_metrics['macro_f1'] * 100
            ),
            'accuracy_diff': transformer_metrics['accuracy'] - baseline_metrics['accuracy'],
            'accuracy_improvement_percent': (
                (transformer_metrics['accuracy'] - baseline_metrics['accuracy']) 
                / baseline_metrics['accuracy'] * 100
            ),
        }
    }
    
    # Вывод результатов
    print("=" * 80)
    print("СРАВНЕНИЕ МОДЕЛЕЙ: Baseline vs RuBERT")
    print("=" * 80)
    print(f"\n📊 Baseline (TF-IDF + Logistic Regression):")
    print(f"   Macro F1-score: {baseline_metrics['macro_f1']:.6f}")
    print(f"   Accuracy:       {baseline_metrics['accuracy']:.6f}")
    print(f"   Алгоритм:       {baseline_metrics.get('algorithm', 'N/A')}")
    print(f"   Векторизатор:   {baseline_metrics.get('vectorizer', 'N/A')}")
    
    print(f"\n🤖 RuBERT:")
    print(f"   Macro F1-score: {transformer_metrics['macro_f1']:.6f}")
    print(f"   Accuracy:       {transformer_metrics['accuracy']:.6f}")
    print(f"   Базовая модель: {transformer_metrics.get('base_model', 'N/A')}")
    
    print(f"\n📈 СРАВНЕНИЕ:")
    macro_f1_diff = comparison['comparison']['macro_f1_diff']
    macro_f1_improvement = comparison['comparison']['macro_f1_improvement_percent']
    accuracy_diff = comparison['comparison']['accuracy_diff']
    accuracy_improvement = comparison['comparison']['accuracy_improvement_percent']
    
    print(f"   Macro F1-score:")
    print(f"     Разница:      {macro_f1_diff:+.6f}")
    print(f"     Улучшение:    {macro_f1_improvement:+.2f}%")
    
    print(f"   Accuracy:")
    print(f"     Разница:      {accuracy_diff:+.6f}")
    print(f"     Улучшение:    {accuracy_improvement:+.2f}%")
    
    if macro_f1_diff > 0:
        print(f"\n✅ RuBERT превосходит Baseline на {macro_f1_improvement:.2f}% по Macro F1-score")
    else:
        print(f"\n⚠️ Baseline показывает лучшие результаты")
    
    print("=" * 80)
    
    # Сохранение результатов
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        print(f"\n📄 Результаты сохранены в: {output_path}")
    
    return comparison


def create_visualization(comparison: Dict, output_path: Optional[Path] = None) -> None:
    """Создает визуализацию сравнения моделей."""
    if not comparison:
        print("Нет данных для визуализации")
        return
    
    baseline = comparison['baseline']
    transformer = comparison['transformer']
    
    # Создаем график
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # График Macro F1-score
    models = [baseline['model_name'], transformer['model_name']]
    macro_f1_scores = [baseline['macro_f1'], transformer['macro_f1']]
    
    bars1 = axes[0].bar(models, macro_f1_scores, color=['#3498db', '#e74c3c'], alpha=0.8)
    axes[0].set_ylabel('Macro F1-score', fontsize=12)
    axes[0].set_title('Сравнение Macro F1-score', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, 1.1])
    axes[0].grid(axis='y', alpha=0.3)
    
    # Добавляем значения на столбцы
    for i, (bar, score) in enumerate(zip(bars1, macro_f1_scores)):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{score:.4f}',
                    ha='center', va='bottom', fontweight='bold')
    
    # Добавляем линию улучшения
    improvement = comparison['comparison']['macro_f1_improvement_percent']
    axes[0].annotate(f'Улучшение: +{improvement:.2f}%',
                    xy=(1, transformer['macro_f1']),
                    xytext=(0.5, transformer['macro_f1'] + 0.15),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2),
                    fontsize=10, fontweight='bold', color='green',
                    ha='center')
    
    # График Accuracy
    accuracy_scores = [baseline['accuracy'], transformer['accuracy']]
    bars2 = axes[1].bar(models, accuracy_scores, color=['#3498db', '#e74c3c'], alpha=0.8)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Сравнение Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_ylim([0, 1.1])
    axes[1].grid(axis='y', alpha=0.3)
    
    # Добавляем значения на столбцы
    for i, (bar, score) in enumerate(zip(bars2, accuracy_scores)):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{score:.4f}',
                    ha='center', va='bottom', fontweight='bold')
    
    # Добавляем линию улучшения
    acc_improvement = comparison['comparison']['accuracy_improvement_percent']
    axes[1].annotate(f'Улучшение: +{acc_improvement:.2f}%',
                    xy=(1, transformer['accuracy']),
                    xytext=(0.5, transformer['accuracy'] + 0.15),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2),
                    fontsize=10, fontweight='bold', color='green',
                    ha='center')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"📊 График сохранен в: {output_path}")
    
    plt.show()


def create_detailed_comparison_table(comparison: Dict) -> pd.DataFrame:
    """Создает детальную таблицу сравнения."""
    baseline = comparison['baseline']
    transformer = comparison['transformer']
    
    # Извлекаем метрики по классам для baseline
    baseline_class_metrics = {}
    if 'metrics' in baseline and isinstance(baseline['metrics'], dict):
        for class_name in ['negative', 'neutral', 'positive']:
            if class_name in baseline['metrics']:
                baseline_class_metrics[class_name] = baseline['metrics'][class_name]
    
    # Создаем DataFrame
    data = {
        'Модель': ['Baseline', 'RuBERT'],
        'Macro F1-score': [baseline['macro_f1'], transformer['macro_f1']],
        'Accuracy': [baseline['accuracy'], transformer['accuracy']],
    }
    
    # Добавляем метрики по классам для baseline если есть
    if baseline_class_metrics:
        for class_name in ['negative', 'neutral', 'positive']:
            if class_name in baseline_class_metrics:
                f1 = baseline_class_metrics[class_name].get('f1-score', 0)
                data[f'F1 {class_name}'] = [f1, None]
    
    df = pd.DataFrame(data)
    
    # Добавляем строку с разницей
    diff_row = {
        'Модель': 'Разница (RuBERT - Baseline)',
        'Macro F1-score': comparison['comparison']['macro_f1_diff'],
        'Accuracy': comparison['comparison']['accuracy_diff'],
    }
    
    if baseline_class_metrics:
        for class_name in ['negative', 'neutral', 'positive']:
            if f'F1 {class_name}' in diff_row:
                diff_row[f'F1 {class_name}'] = None
    
    df = pd.concat([df, pd.DataFrame([diff_row])], ignore_index=True)
    
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Сравнение эффективности Baseline и RuBERT моделей по Macro F1-score"
    )
    parser.add_argument(
        "--baseline-metadata",
        type=Path,
        default=Path("models/metadata.json"),
        help="Путь к metadata.json baseline модели",
    )
    parser.add_argument(
        "--transformer-metrics",
        type=Path,
        default=Path("reports/transformer_metrics.json"),
        help="Путь к transformer_metrics.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/model_comparison.json"),
        help="Путь для сохранения результатов сравнения",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=Path("reports/model_comparison.png"),
        help="Путь для сохранения графика",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Не показывать график",
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="Показать детальную таблицу сравнения",
    )
    
    args = parser.parse_args()
    
    # Сравнение моделей
    comparison = compare_models(
        baseline_metadata_path=args.baseline_metadata,
        transformer_metrics_path=args.transformer_metrics,
        output_path=args.output,
    )
    
    if not comparison:
        return
    
    # Визуализация
    if not args.no_plot:
        try:
            create_visualization(comparison, output_path=args.plot)
        except Exception as e:
            print(f"⚠️ Ошибка при создании графика: {e}")
    
    # Детальная таблица
    if args.table:
        try:
            df = create_detailed_comparison_table(comparison)
            print("\n📋 Детальная таблица сравнения:")
            print(df.to_string(index=False))
        except Exception as e:
            print(f"⚠️ Ошибка при создании таблицы: {e}")


if __name__ == "__main__":
    main()

