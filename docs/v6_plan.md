# План v6.0: Решение проблем модели и переход к новой архитектуре

## Текущее состояние (v5.1)

| Метрика | Значение | Оценка |
|---------|----------|--------|
| Val F1-macro | 0.73 | Приемлемо |
| Test ROC-AUC | 0.651 | **Плохо** (близко к случайному 0.5) |
| Test PR-AUC | 0.529 | **Плохо** |
| Confusion Matrix | Норма: хорошо, Усталость: почти не ловит | **Критическая проблема** |
| Train Loss | ~0.42 | Высокий |
| Val Loss | ~0.55 | Высокий |
| Параметры модели | ~39K | Избыточно для 22K окон |

### Корневые причины

1. **Дисбаланс классов: 69.4% норма / 30.6% усталость.** Модель "ленится" и предсказывает "норма" для всех — это уже даёт accuracy ~70%. BCE + pos_weight недостаточно компенсирует.

2. **Модель не различает паттерны усталости.** AUC 0.65 означает, что ранжирование вероятностей почти случайное — модель не выучила реальные признаки.

3. **Один IMU-датчик (поясница/таз)** даёт слабый сигнал усталости. Усталость проявляется в **комплексе**: изменение каденса, асимметрия шагов, изменение ускорений в конечностях. Один датчик на торсе теряет эту информацию.

4. **Архитектура слишком "глубокая" для задачи.** 3 свёрточных слоя + attention + classifier при `output_dim=32` и окне 100 точек — градиенты размываются, модель не конвергирует.

5. **RuntimeError при загрузке state_dict** — несоответствие ключей между моделью с одним `output_dim` и сохранённой с другим.

---

## Часть 1: Немедленные исправления (v5.2)

### 1.1 Единый CONFIG для всех гиперпараметров

**Проблема:** размеры слоёв вычисляются через `output_dim // 2`, `channels // 4` — при нечётных числах ломается архитектура, нет единого места управления.

**Решение:** вынести ВСЕ размеры слоёв в CONFIG:

```python
CONFIG = {
    # --- Обучение ---
    'batch_size': 32,
    'epochs': 50,
    'lr': 1e-4,
    'weight_decay': 5e-3,
    'patience': 4,
    'min_delta': 2e-5,
    'label_smoothing': 0.015,

    # --- Архитектура: Encoder ---
    'imu_channels': 6,
    'encoder_channels': 32,       # ранее output_dim / out_channels
    'encoder_dropout': 0.375,
    'kernel_sizes': [7, 5, 3],    # размеры ядер conv1, conv2, conv3

    # --- Архитектура: Classifier ---
    'classifier_dropout': 0.525,

    # --- Focal Loss ---
    'focal_gamma': 2.0,
}
```

Модули выполняют деление внутри `__init__`, а при вызове получают значения из CONFIG:

```python
class TemporalAttention(nn.Module):
    def __init__(self, channels):  # деление внутри
        super().__init__()
        self.score = nn.Sequential(
            nn.Conv1d(channels, channels // 4, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(channels // 4, 1, kernel_size=1),
        )

class IMUEncoderWithAttention(nn.Module):
    def __init__(self, in_channels, out_channels, dropout, kernel_sizes):
        super().__init__()
        k1, k2, k3 = kernel_sizes
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=k1, padding=k1 // 2),
            nn.BatchNorm1d(out_channels), nn.ReLU(inplace=True),
            nn.MaxPool1d(2), nn.Dropout(dropout),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=k2, padding=k2 // 2),
            nn.BatchNorm1d(out_channels), nn.ReLU(inplace=True),
            nn.MaxPool1d(2), nn.Dropout(dropout),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=k3, padding=k3 // 2),
            nn.BatchNorm1d(out_channels), nn.ReLU(inplace=True),
        )
        self.attention = TemporalAttention(out_channels)  # деление channels // 4 внутри

class FatigueCNN_Simple(nn.Module):
    def __init__(self, cfg):  # принимает CONFIG, деление внутри модулей
        super().__init__()
        enc_ch = cfg['encoder_channels']
        self.imu_encoder = IMUEncoderWithAttention(
            in_channels=cfg['imu_channels'],
            out_channels=enc_ch,
            dropout=cfg['encoder_dropout'],
            kernel_sizes=cfg['kernel_sizes'],
        )
        self.classifier = nn.Sequential(
            nn.Linear(enc_ch, enc_ch // 2),              # деление внутри
            nn.ReLU(inplace=True),
            nn.Dropout(cfg['classifier_dropout']),
            nn.Linear(enc_ch // 2, 1),                   # деление внутри
        )

# --- Создание модели из CONFIG ---
model = FatigueCNN_Simple(CONFIG).to(DEVICE)
```

### 1.2 Исправление RuntimeError state_dict

**Проблема:** модель создаётся с текущим CONFIG, но `best_model_v5.pth` сохранён с другим `output_dim` → ключи не совпадают.

**Решение:**

```python
# --- При сохранении: записываем CONFIG рядом с весами ---
torch.save({
    'model_state_dict': model.state_dict(),
    'config': CONFIG,
    'best_threshold': best_threshold,
    'best_val_f1': best_val_f1,
}, RESULTS_DIR / 'best_model_v5.pth')

# --- При загрузке: восстанавливаем CONFIG из чекпоинта ---
checkpoint = torch.load(RESULTS_DIR / 'best_model_v5.pth', weights_only=False)
saved_cfg = checkpoint['config']
base_model = FatigueCNN_Simple(saved_cfg).to(DEVICE)
base_model.load_state_dict(checkpoint['model_state_dict'])
best_threshold = checkpoint['best_threshold']
```

### 1.3 Focal Loss вместо BCE (борьба с дисбалансом)

Модель игнорирует класс "усталость" потому что BCE с pos_weight всё равно даёт малый градиент на хорошо классифицированных "лёгких" примерах нормы. **Focal Loss** фокусируется на "трудных" примерах:

$$FL(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

```python
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction='none'
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()
```

- `gamma=2.0` — стандартное значение, уменьшает вклад лёгких примеров в ~25 раз
- `pos_weight` сохраняется для дополнительной компенсации дисбаланса
- `label_smoothing` **убирается** — Focal Loss и smoothing плохо совместимы

---

## Часть 2: Новая стратегия данных и архитектура (v6.0)

### 2.1 Смена фокуса: от поясницы к запястью

**Ключевое решение:** целевое устройство — браслет/часы на запястье. Датасеты Zenodo и 4TU содержат датчики только на поясе/ногах и **не релевантны** для запястного устройства.

**Изменение состава датасетов (v6.0):**

| Датасет | v5.1 | v6.0 | Причина |
|---------|------|------|---------|
| Zenodo (бег, поясница) | Включён | **Исключён** | Нет датчика на запястье |
| 4TU (бег, торс+ноги) | Включён | **Исключён** | Нет датчика на запястье |
| PhysioNet (cycling/спринты, Empatica E4 запястье) | Исключён | **Включён** | Empatica E4 — запястье (ACC + BVP + EDA + TEMP + HR) |
| WSD4FEDSRM (вращения плеча, предплечье/кисть) | Исключён | **Включён** | IMU forearm/hand — ближайшее к запястью + PPG |

### 2.2 Доступные данные по сегментам запястья/руки

#### PhysioNet (до 31 субъектов, AEROBIC + ANAEROBIC)
- **Датчик:** Empatica E4 на **запястье**
- Модальности: ACC 3-осевой (32 Hz) + BVP (64 Hz) + EDA (4 Hz) + TEMP (4 Hz) + HR (1 Hz)
- Гироскоп отсутствует → gyro-каналы заполняются нулями
- Протокол: велотренажёр (AEROBIC) + Wingate-спринты (ANAEROBIC)

#### WSD4FEDSRM (34 субъекта, вращения плеча до отказа)
- **Датчики:** IMU на `forearm` (предплечье) и `hand` (кисть) — ближайшие к запястью
- Модальности: ACC (3) + Gyro (3) + PPG на запястье
- Borg RPE разметка (RPE ≤ 11 → норма, RPE ≥ 14 → усталость)

### 2.3 Архитектура v6.0: Wrist-IMU + Physio (двухветочная)

```
┌─────────────────────────────────────────────────────────────────┐
│            FatigueWrist v6.0                                     │
│                                                                  │
│  ═══ IMU-ветка (acc запястья, ± gyro) ═══                       │
│  Input: (B, 100, 3..6)                                          │
│  Conv1D(C_in→C, k) → BN → ReLU → GAP → (B, C)                 │
│                                          │                       │
│  ═══ Physio-ветка (BVP, EDA, TEMP, HR) ═══                     │
│  Input: (B, 100, 4)   [mask: has_physio]                        │
│  Conv1D(4→C_p, k) → BN → ReLU → GAP → (B, C_p)                │
│                                          │                       │
│  ══════════ Слияние ══════════                                  │
│  Concat(IMU_emb, Physio_emb) → (B, C + C_p)                    │
│  Linear(C+C_p → hidden) → ReLU → Dropout → Linear(hidden → 1)  │
└─────────────────────────────────────────────────────────────────┘
```

**Ключевые отличия от v5.1:**
- Physio-ветка **возвращается** (BVP, EDA, TEMP, HR — доступны на Empatica E4)
- IMU-ветка принимает 3 канала (ACC only) для PhysioNet, 6 каналов (ACC + Gyro) для WSD4FEDSRM
- Модель ориентирована на данные запястья → прямая применимость к реальным часам/браслетам

### 2.4 Мультисегментная стратегия (отложена, v7.0+)

> **На будущее:** 4TU содержит 8 IMU-датчиков на разных сегментах тела (sternum, pelvis, lul/rul, lll/rll, lfoot/rfoot). Мультисегментная модель с параллельными энкодерами для каждого сегмента позволит "видеть" асимметрию шагов — главный биомеханический признак усталости при беге. Эта стратегия отложена до появления данных с запястным IMU в совокупности с другими сегментами.

---

## Часть 3: Решение проблемы "не ловит усталость"

### 3.1 Диагностика: почему модель не предсказывает "усталость"

При дисбалансе 70/30 модель получает **минимум loss просто предсказывая "норма" всем**:
- F1-macro при "всем 0" ≈ 0.41 (macro считает оба класса)
- F1-macro `0.73` — модель немного лучше рандома, но очень мало

Проблема в **пороге**: даже если модель выдаёт вероятности 0.2–0.4 для подозрительных случаев, при пороге 0.5 они классифицируются как "норма".

### 3.2 Конкретные шаги

#### Шаг A: Oversampling класса "усталость"

Текущий `WeightedRandomSampler` взвешивает по **доменам**, но не по **классам**. Нужно комбинированное взвешивание:

```python
def compute_sample_weights(y, domains):
    """Вес = 1/(n_class * n_domain) для данного sample."""
    weights = np.zeros(len(y), dtype=np.float64)
    for dom in np.unique(domains):
        for cls in [0, 1]:
            mask = (domains == dom) & (y == cls)
            n = mask.sum()
            if n > 0:
                weights[mask] = len(y) / (2 * len(np.unique(domains)) * n)
    return weights
```

#### Шаг B: Focal Loss (описан в 1.3)

#### Шаг C: Пересмотр порога

Вместо фиксированного 0.5 — выбирать порог максимизирующий **recall класса 1** при минимально допустимом precision:

```python
def find_recall_optimized_threshold(y_true, y_prob, min_precision=0.3):
    """Найти порог с максимальным recall при precision >= min_precision."""
    best_thresh, best_recall = 0.5, 0.0
    for t in np.linspace(0.05, 0.95, 181):
        preds = (y_prob >= t).astype(int)
        tp = ((preds == 1) & (y_true == 1)).sum()
        fp = ((preds == 1) & (y_true == 0)).sum()
        fn = ((preds == 0) & (y_true == 1)).sum()
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        if prec >= min_precision and rec > best_recall:
            best_recall = rec
            best_thresh = t
    return best_thresh, best_recall
```

#### Шаг D: Asymmetric Loss

Альтернатива Focal Loss — штрафной вес за пропуск усталости:

```python
# Вес FN (пропуск усталости) в 3x больше чем FP (ложная тревога)
pos_weight = torch.tensor([3.0]).to(DEVICE)
```

---

## Часть 4: План реализации

### Фаза 1 — Быстрые исправления (v5.2)

| # | Задача | Файл | Приоритет |
|---|--------|------|-----------|
| 1 | Единый CONFIG без `// 2`, `// 4` | Ячейки 14–16 ноутбука | Высокий |
| 2 | Сохранение CONFIG в checkpoint | Ячейка обучения | Высокий |
| 3 | Focal Loss вместо LabelSmoothingBCE | Ячейка утилит обучения | Высокий |
| 4 | Комбинированный class+domain sampler | Ячейка подготовки данных | Высокий |
| 5 | Recall-optimized threshold selection | Ячейка оценки | Средний |

### Фаза 2 — Новая архитектура (v6.0)

| # | Задача | Файл | Приоритет |
|---|--------|------|-----------|
| 6 | Загрузчик 4TU по сегментам | `afc/io_4tu.py` / `build_dataset.py` | Высокий |
| 7 | Мультисегментная модель | Новый ноутбук `diplom_v6.ipynb` | Высокий |
| 8 | FatigueMiniCNN (1-conv fallback) | В том же ноутбуке | Средний |
| 9 | Сравнительный эксперимент v5.2 vs v6.0 | Ноутбук | Средний |

### Фаза 3 — Оценка и документация

| # | Задача | Файл | Приоритет |
|---|--------|------|-----------|
| 10 | LOSO с новой моделью | Ноутбук | Высокий |
| 11 | Анализ per-segment importance | Ноутбук | Средний |
| 12 | Обновление ТЕХНИЧЕСКАЯ_ДОКУМЕНТАЦИЯ | `docs/` | Низкий |

---

## Часть 5: Ожидаемые результаты

| Изменение | Влияние на AUC | Влияние на Recall(усталость) |
|-----------|---------------|------------------------------|
| Focal Loss + class weighting | +0.05–0.10 | **+15–25%** |
| Recall-optimized threshold | +0.00 (AUC не меняется) | **+10–20%** |
| Мультисегментная модель (4TU 8 IMU) | +0.08–0.15 | **+10–20%** |
| FatigueMiniCNN (антипереобучение) | +0.03–0.07 | +5–10% |
| **Совокупно** | **0.75–0.85** | **Recall ≥ 0.6** |

### Реалистичные цели v6.0

| Метрика | v5.1 (текущий) | v6.0 (цель) |
|---------|----------------|-------------|
| ROC-AUC | 0.651 | **≥ 0.78** |
| PR-AUC | 0.529 | **≥ 0.65** |
| Recall (усталость) | ~0.15 | **≥ 0.55** |
| F1-macro | 0.73 | **≥ 0.72** (может упасть из-за роста FP, это нормально) |
| Loss | 0.42–0.55 | **0.30–0.45** |

> **Примечание:** рост recall неизбежно увеличит FP (ложные тревоги). Для медицинского/спортивного домена **пропуск усталости опаснее ложной тревоги**, поэтому это приемлемый компромисс.
