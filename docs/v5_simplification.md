# v5.0 — Упрощение модели для борьбы с переобучением

## Проблема

Модель v4.0 (`FatigueCNN_Attention`, dual-branch CNN + Temporal Attention) переобучалась:
- Train loss снижался, val loss расходился
- Подбор гиперпараметров (lr, weight_decay, dropout, patience) не решал проблему
- Overfitting gap нарастал с каждой эпохой

## Диагностика причин

### 1. Physio-ветка как источник шума

Physio-каналы (BVP, EDA, TEMP, HR) доступны **только для PhysioNet** (~47% окон). Для Zenodo, 4TU и WSD4FEDSRM physio-входы обнулены, а `has_physio=False` зануляет вектор physio_feat.

**Проблема:** модель учится использовать наличие/отсутствие physio-данных как shortcut-признак домена, а не как полезные физиологические паттерны. Это ведёт к переобучению на domain identity.

### 2. WSD4FEDSRM — высокий domain shift при малом объёме

| Показатель | WSD4FEDSRM | Другие домены |
|-----------|------------|---------------|
| Окон на субъект | ~33 | 316–2358 |
| Доля в датасете | 2.3% | 97.7% |
| Тип упражнения | Изолированное (вращение плеча) | Циклическое (бег, cycling) |
| Сегмент тела | Рука/плечо | Таз/поясница/запястье |

При domain-weighted sampling WSD4FEDSRM получает вес 1/4, несмотря на минимальный объём и максимальный domain shift. Модель тратит ёмкость на подстройку под нерепрезентативный домен.

### 3. Избыточная ёмкость модели

| Версия | Параметры | Окон | Ratio params/samples |
|--------|-----------|------|---------------------|
| v4.0 (dual-branch) | ~80K | ~49K | 1.63 |
| **v5.0 (IMU-only)** | **~27K** | **~48K** | **0.56** |

Правило: для табличных / 1D-данных ratio < 1.0 обычно менее подвержен переобучению.

---

## Изменения v5.0

### Архитектура

**Было (v4.0):** `FatigueCNN_Attention` (~80K параметров)
```
IMU (6ch) → IMUEncoderWithAttention → 64d ─┐
                                             ├── Concat (96d) → FC(96→48) → FC(48→1)
Physio (4ch) → PhysioEncoderWithAttention → 32d ─┘
```

**Стало (v5.0):** `FatigueCNN_Simple` (~27K параметров)
```
IMU (6ch) → IMUEncoderWithAttention → 64d → FC(64→32) → FC(32→1)
```

**Удалено:**
- `PhysioEncoder`, `PhysioEncoderWithAttention` — CNN-энкодер physio-каналов
- `FatigueCNN_LSTM` — baseline модель с BiLSTM
- Physio-аугментации (`physio_channel_mask`, physio-ветка в `augment_sample`)
- `has_physio` маска из Dataset и forward pass

### Датасеты

| Домен | v4.0 | v5.0 | Обоснование |
|-------|------|------|-------------|
| Zenodo | ✅ | ✅ | Бег, полный IMU (acc+gyro) |
| 4TU | ✅ | ✅ | Бег, полный IMU (acc+gyro) |
| PhysioNet | ✅ | ✅ | Cycling/спринты, ACC (gyro=0) |
| WSD4FEDSRM | ✅ | ❌ | Удалён: мало окон, другое упражнение |

### Конвейер данных

```python
# v5.0: загрузка composite_full.npz с фильтрацией
data = np.load('composite_full.npz')
keep_mask = np.isin(data['domains'], ['zenodo', '4tu', 'physionet'])
X_imu_all = data['X_imu'][keep_mask]  # (N, 100, 6)
y_all = data['y'][keep_mask]
```

Physio-массив (`X_physio`) не загружается. `has_physio` не используется.

### Аугментация

| Техника | v4.0 | v5.0 |
|---------|------|------|
| Gaussian noise | ✅ IMU+Physio | ✅ IMU only |
| Time warp | ✅ IMU+Physio | ✅ IMU only |
| Magnitude scale | ✅ IMU+Physio | ✅ IMU only |
| Window slice | ✅ IMU | ✅ IMU |
| Time reverse | ✅ IMU | ✅ IMU |
| IMU SO(3) rotate | ✅ IMU | ✅ IMU |
| Channel dropout | ✅ IMU | ✅ IMU |
| **Physio channel mask** | ✅ Physio | ❌ Удалена |

### Обучение

Гиперпараметры сохранены из v4.1:

| Параметр | Значение |
|----------|----------|
| batch_size | 32 |
| lr | 0.001 |
| weight_decay | 3e-4 |
| patience | 4 |
| label_smoothing | 0.05 |
| dropout (classifier) | 0.4 |
| dropout (encoder) | 0.2 |
| optimizer | AdamW |
| scheduler | CosineAnnealingLR |

`FatigueDataset` теперь возвращает 2-tuple `(X_imu, y)` вместо 4-tuple `(X_imu, X_physio, y, has_physio)`.

`train_epoch()` и `validate()` обновлены для 2-tuple батчей.

### Оценка

- **Hold-out**: 60/20/20 subject-level split, стратифицировано по домену
- **LOSO**: до 30 фолдов, `FatigueCNN_Simple`, early stopping
- **Персонализация**: `freeze encoder → train classifier → unfreeze conv3+attention → fine-tune`
- **Overfitting analysis**: отслеживание train-val gap по эпохам, ratio params/samples

---

## Затронутые файлы

| Файл | Изменения |
|------|-----------|
| `notebooks/diplom_lstm_augm_fix.ipynb` | Полная переработка: модель, dataset, train/validate, LOSO, personalization |
| `docs/v5_simplification.md` | Этот документ |

**Не затронуты** (но могут потребовать обновления):
- `afc/models_cnn.py` — содержит оригинальные `FatigueCNN1D`, `FatigueTrainer` (используются в `train_deep.py`)
- `scripts/build_dataset.py` — по-прежнему собирает все 4 домена в NPZ, фильтрация на этапе загрузки
- `config.yaml` — без изменений

---

## Ожидаемый эффект

1. **Уменьшение overfitting gap** — модель в ~3× меньше, ratio params/samples < 1
2. **Устранение domain shortcut** — без physio-ветки модель не может различать домены по наличию/отсутствию physio
3. **Чистый IMU-сигнал** — фокус на инерциальных паттернах усталости (изменение биомеханики, амплитуд, частот)
4. **Стабильность LOSO** — без WSD4FEDSRM нет фолдов с 3–9 окнами

## Следующие шаги (при необходимости)

Если переобучение сохраняется:
- Уменьшить `out_channels` в IMU encoder (64 → 32)
- Добавить weight decay на attention слой
- Попробовать `DropConnect` вместо `Dropout`
- Уменьшить число Conv-блоков (3 → 2)
- Увеличить `label_smoothing` (0.05 → 0.10)
