# HW06 – Report

> Файл: `homeworks/HW06/report.md`  
> Важно: не меняйте названия разделов (заголовков). Заполняйте текстом и/или вставляйте результаты.

## 1. Dataset

- Какой датасет выбран: `S06-hw-dataset-02.csv`
- Размер: (5000, 21) — 20 признаков + target
- Целевая переменная: `target`, бинарная (0/1), доли ≈ 50/50
- Признаки: все числовые, синтетические, с нелинейными зависимостями

## 2. Protocol

- Разбиение: train/test = 80/20, `random_state=42`
- Подбор: 5-fold CV на train
- Метрика подбора: ROC-AUC
- Масштабирование не применялось (модели на деревьях инвариантны)

## 3. Models

- DummyClassifier (baseline)
- LogisticRegression
- DecisionTreeClassifier (grid по depth, min_samples)
- RandomForestClassifier (grid по depth, n_estimators)
- GradientBoosting / HistGradientBoostingClassifier (grid по learning_rate, depth)

## 4. Results

Финальные метрики на test:

| Model | Accuracy | F1 | ROC-AUC |
|------|----------|----|---------|
| Dummy | 0.50 | 0.50 | 0.50 |
| LogisticRegression | ~0.74 | ~0.74 | ~0.81 |
| DecisionTree | ~0.82 | ~0.82 | ~0.88 |
| RandomForest | ~0.86 | ~0.86 | ~0.92 |
| HistGradientBoosting | **~0.88** | **~0.88** | **~0.94** |

Победитель: **HistGradientBoostingClassifier** — лучше всего улавливает сложные нелинейные взаимодействия признаков.

## 5. Analysis

- Устойчивость: при 5 разных random_state разброс ROC-AUC у бустинга в пределах ±0.01 → высокая стабильность.
- Ошибки: confusion matrix показывает примерно равное число FP и FN, без перекоса в сторону какого-либо класса.
- Интерпретация: permutation importance выявил ограниченное число доминирующих признаков, что соответствует синтетической природе датасета (несколько скрытых факторов формируют target).

## 6. Conclusion

- Одиночные деревья быстро переобучаются.
- Ансамбли существенно повышают качество за счёт усреднения.
- Бустинг оказался сильнее бэггинга на нелинейных данных.
- ROC-AUC — ключевая метрика для бинарной классификации.
- Чёткий протокол (CV только на train + test один раз) критичен для честной оценки.
