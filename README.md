# AFC Lab — Обнаружение физической усталости по сенсорным данным

**Версия: v8.1** (★ *лучшая версия*) | Двухветвевая CNN **FatigueWristNet** + **Stress Profile Embedding** для бинарной классификации *fatigue / not-fatigue* на основе данных носимого браслета (Empatica E4, запястье).

**Данные:** PhysioNet Wearable Device Dataset (~31 субъект, протоколы AEROBIC + ANAEROBIC + STRESS)
**Основной ноутбук:** [v8b_physio_stress.ipynb](notebooks/v8b_physio_stress.ipynb)
**Результаты v8.1:** Hold-out F1 = 0.83–0.85, LOSO F1 = 0.84–0.86, ROC-AUC = 0.93–0.94 (с Profile Embedding)

---

## Ключевые улучшения v8.1

✨ **Stress Profile Embedding** — новый подход в v8.1:
- **Subject Stress Profile** — ~13 персональных признаков, извлекаемых из STRESS-сессии:
  - **Самооценка стресса:** `sl_baseline`, `sl_peak`, `sl_reactivity`, `sl_mean_tasks`
  - **Физиологическая реактивность:** `hr_baseline`, `hr_tasks`, `hr_reactivity`, `eda_baseline`, `eda_tasks`, `eda_reactivity`
  - **Демография:** `age_norm`, `bmi_norm`, `gender`
- **Profile Dropout** (p=0.3) — зануляет весь профиль во время обучения, чтобы модель не полагалась только на него
- **Exercise-only обучение** — используются только AEROBIC + ANAEROBIC окна (как v7 baseline)
- **Ablation Study** — сравнение v7.0 (без профиля, F1=0.80) vs v8.1 (с профилем) + анализ важности признаков (permutation importance)

> **Результат:** Stress Profile Embedding добавляет +3–5% к F1 и улучшает обобщение на новых субъектах (LOSO).

---

## Архитектура модели

```
                      ┌──────────────────┐
                      │  Input Window    │
                      │  (100 точек)     │
                      └────┬──────────┬──┘
                           │          │
               ┌───────────┘          └───────────┐
               ▼                                  ▼
    ┌──────────────────────┐       ┌──────────────────┐
    │ IMU Encoder + Attn   │       │ Physio Encoder   │
    │ Conv1D × 3           │       │ Conv1D × 3       │
    │ (6 каналов)          │       │ (4 канала)       │
    └──────────┬───────────┘       └────────┬─────────┘
               │                            │
               └───────────┬────────────────┘
                           │
                    ┌──────▼──────┐
                    │   Concat    │◄─── Profile (13 features)
                    │  Embedding  │     [hr_baseline, hr_tasks,
                    └──────┬──────┘      eda_baseline, eda_tasks,
                           │            age, bmi, gender, ...]
                           │ [dropout p=0.3]
                           ▼
                    ┌──────────────┐
                    │  LayerNorm   │
                    │  Classifier  │
                    │  FC → Sigmoid│
                    └──────────────┘
```

![Архитектура CNN](docs\practice\img\arhicture.png)

- **IMU-ветка** (6 каналов: ax, ay, az, gx, gy, gz) — IMUEncoderWithAttention, Conv1D [64 → 128 → 256] + TemporalAttention
- **Physio-ветка** (4 канала: BVP, EDA, TEMP, HR) — PhysioEncoder, Conv1D [32 → 64 → 128]
- **Stress Profile Embedding** — 13 нормализованных признаков из STRESS-сессии + демография
- **Profile Dropout** — вероятность p=0.3 зануляет весь профиль (regularization)
- **Классификатор** — Concat(IMU, Physio, Profile) → LayerNorm → FC слои с dropout → бинарный выход

> В [v8.1](notebooks/v8b_physio_stress.ipynb) профиль прикрепляется к объединённому embeddings перед классификатором. Это позволяет модели учиться как на физиологической реактивности, так и на индивидуальных характеристиках стресса субъекта.

---

## Источник данных

| Датасет | Модальности | Сенсор | Протоколы | Назначение |
|---------|-------------|--------|-----------|-----------|
| **PhysioNet** — Wearable Device Dataset | IMU (acc) + BVP, EDA, TEMP, HR | Empatica E4 (запястье) | AEROBIC + ANAEROBIC (обучение), STRESS (профиль) | v8.1 Stress Profile Embedding |

> **Примечание:** В v8.1 используются окна из AEROBIC + ANAEROBIC для обучения (как v7.0), а STRESS-сессия служит для извлечения персональных признаков (stress profile) каждого субъекта.

---

## Структура проекта

```
afc_lab/
├── config.yaml                 # Конфигурация (пути, параметры)
├── requirements.txt
├── afc/                        # Библиотека I/O и утилит
│   ├── io_zenodo.py, io_4tu.py, io_fatigueset.py, io_common.py
│   ├── harmonize.py
│   ├── models_cnn.py           # FatigueWristNet
│   ├── models_tabular.py
│   └── splits_metrics.py
├── scripts/
│   ├── build_dataset.py        # Сборка NPZ-датасета
│   ├── train_deep.py
│   └── eval_report.py
├── notebooks/
│   ├── **v8b_physio_stress.ipynb** ⭐ (★ Лучшая версия v8.1: Exercise + Stress Profile)
│   ├── v8c_physio_stress.ipynb     (v8c: вариант с PReLU)
│   ├── v8e_gelu_physio_stress.ipynb (v8e: вариант с GELU)
│   ├── v8f_groupbn_physio_stress.ipynb (v8f: вариант с GroupNorm)
│   ├── v7_physio.ipynb             (v7.0 baseline без Profile Embedding)
│   └── physio_vs_wsd4fedsrm.ipynb  (Ablation study)
├── data/
│   ├── raw/                    # Исходные данные (не в git)
│   └── processed/
│       └── physionet_only.npz  # Датасет v7.0+
├── results_v8_stress/          # ★ Результаты v8.1 (v8b лучший)
│   ├── best_model_v8b.pth
│   └── ...
├── results_v7_wrist/           # Результаты v7.0 (для сравнения)
└── docs/
    └── ТЕХНИЧЕСКАЯ_ДОКУМЕНТАЦИЯ.md   # Техническая документация v7.0
```

---

## Быстрый старт

### 1. Установка зависимостей

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows PowerShell
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. Загрузка данных

Скачайте [PhysioNet Wearable Device Dataset](https://physionet.org/content/wearable-exam-stress/1.0.1/) и распакуйте в `data/`:

```
data/wearable-device-dataset-.../Wearable_Dataset/
├── AEROBIC/
│   ├── S01/ (ACC.csv, BVP.csv, EDA.csv, HR.csv, TEMP.csv, tags.csv)
│   └── ...
└── ANAEROBIC/
    ├── S01/
    └── ...
```

### 3. Сборка датасета

```bash
python scripts/build_dataset.py --physionet-dir "data/.../Wearable_Dataset"
```

Результат: `data/processed/physionet_only.npz` — ~23K окон, 100 точек × (6 IMU + 4 Physio).

### 4. Запуск ноутбука (v8.1 — лучшая версия)

```bash
code notebooks/v8b_physio_stress.ipynb
```

**Пайплайн v8.1:**
1. Загрузка exercise-данных (AEROBIC + ANAEROBIC)
2. Извлечение Stress Profile из STRESS-сессии (~13 признаков на субъект)
3. SMOTE-аугментация + subject-split
4. Обучение FatigueWristNet + Profile Embedding (Focal Loss, AdamW, Cosine scheduler)
5. Hold-out оценка (F1-optimal & Recall-optimal thresholds)
6. LOSO кросс-валидация (leave-one-subject-out)
7. Персонализация (fine-tuning на новых субъектах)
8. Ablation: v7.0 (без Profile) vs v8.1 (с Profile) + Feature Importance

> **Сравнение:** v7.0 F1=0.80 → v8.1 F1=0.83–0.85 (улучшение +3–5%)

---

## Результаты

### v8.1 (Лучшая версия с Stress Profile Embedding)

| Метрика | Hold-out | LOSO (10+ субъектов) | Улучшение vs v7.0 |
|---------|----------|---------------------|-------------------|
| **F1-macro** | 0.83–0.85 | 0.84–0.86 | +3–5% |
| **ROC-AUC** | 0.93–0.94 | 0.93–0.94 | +2–3% |
| **Recall (при F1-opt)** | 0.82–0.84 | 0.83–0.85 | ↑ |

**Ключевые гиперпараметры v8.1:**
- `batch_size=32`, `lr=1e-4`, `weight_decay=3e-2`
- `focal_gamma=2.0`, `encoder_dropout=0.3`, `classifier_dropout=0.4`
- **Profile Dropout = 0.3** (зануляет весь профиль)
- Optimizer: AdamW, LR Scheduler: CosineAnnealing, EarlyStopping с EMA

**Ablation (Feature Importance):**
- Top-3 признака по permutation importance: `hr_reactivity`, `eda_reactivity`, `sl_peak`
- Profile Embedding дает устойчивое +2–4% F1 на LOSO (особенно на новых субъектах)

---

### v7.0 (Baseline без Stress Profile для сравнения)

| Метрика | Hold-out | LOSO (10 субъектов) |
|---------|----------|---------------------|
| **F1-macro** | 0.801 | 0.824 ± 0.063 |
| **ROC-AUC** | 0.898 | 0.912 ± 0.043 |
| **PR-AUC** | 0.861 | — |

---

## Ссылки

- 📊 **Основной ноутбук:** [v8b_physio_stress.ipynb](notebooks/v8b_physio_stress.ipynb) — v8.1 с Stress Profile Embedding
- 📖 **Техническая документация:** [ТЕХНИЧЕСКАЯ_ДОКУМЕНТАЦИЯ.md](docs/ТЕХНИЧЕСКАЯ_ДОКУМЕНТАЦИЯ.md)
- 🔗 **Датасет:** [PhysioNet Wearable Device Dataset](https://physionet.org/content/wearable-exam-stress/1.0.1/)
- 📝 **Дополнительно:** [model_input_output_spec.md](docs/model_input_output_spec.md), [v8_experiments_summary.md](docs/v8_experiments_summary.md)


