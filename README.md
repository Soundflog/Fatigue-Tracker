# AFC Lab — Обнаружение физической усталости по сенсорным данным

**Версия: v7.0** | Двухветвевая CNN **FatigueWristNet** для бинарной классификации *fatigue / not-fatigue* на основе данных носимого браслета (Empatica E4, запястье).

**Данные:** PhysioNet Wearable Device Dataset (~31 субъект, протоколы AEROBIC + ANAEROBIC)
**Результаты:** Hold-out F1 = 0.80, LOSO F1 = 0.82 ± 0.06, ROC-AUC = 0.91

---

## Архитектура модели

```
                      ┌─────────────────┐
                      │   Input Window   │
                      │   (100 точек)    │
                      └────┬────────┬────┘
                           │        │
               ┌───────────┘        └───────────┐
               ▼                                ▼
    ┌──────────────────────┐       ┌─────────────────┐
    │ IMU Encoder + Attn   │       │  Physio Encoder  │
    │ Conv1D × 3           │       │  Conv1D × 3      │
    │ (6 каналов)          │       │  (4 канала)      │
    └──────────┬───────────┘       └────────┬────────┘
               │                            │
               └───────────┬────────────────┘
                           ▼
                    ┌──────────────┐
                    │  LayerNorm   │
                    │  Classifier  │
                    │  FC → Sigmoid│
                    └──────────────┘
```

- **IMU-ветка** (6 каналов: ax, ay, az, gx, gy, gz) — IMUEncoderWithAttention, Conv1D [64 → 128 → 256] + TemporalAttention
- **Physio-ветка** (4 канала: BVP, EDA, TEMP, HR) — PhysioEncoder, Conv1D [32 → 64 → 128]
- **Классификатор** — LayerNorm → FC слои с dropout → бинарный выход

> В v7.0 все окна имеют `has_physio = True` (только PhysioNet), обе ветки активны всегда.

---

## Источник данных

| Датасет | Модальности | Сенсор | Субъектов | Ссылка |
|---------|-------------|--------|-----------|--------|
| **PhysioNet** — Wearable Device Dataset | IMU (acc) + BVP, EDA, TEMP, HR | Empatica E4 (запястье) | ~31 | [physionet.org](https://physionet.org/content/wearable-exam-stress/1.0.1/) |

> Датасеты Zenodo, 4TU и WSD4FEDSRM исследованы в ablation study и исключены из v7.0 — см. [ТЕХНИЧЕСКАЯ_ДОКУМЕНТАЦИЯ.md](docs/ТЕХНИЧЕСКАЯ_ДОКУМЕНТАЦИЯ.md), раздел 3.

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
│   ├── v7_physio.ipynb         # ★ Основной ноутбук v7.0 (PhysioNet only)
│   └── physio_vs_wsd4fedsrm.ipynb  # Ablation study
├── data/
│   ├── raw/                    # Исходные данные (не в git)
│   └── processed/
│       └── physionet_only.npz  # ★ Датасет v7.0
├── results_v7_wrist/           # Результаты v7.0
│   └── best_model_v7.pth
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

### 4. Запуск ноутбука

```bash
code notebooks/v7_physio.ipynb
```

Пайплайн: загрузка → SMOTE-аугментация → обучение (FocalLoss, AdamW, cosine scheduler) → hold-out оценка → LOSO → персонализация.

---

## Результаты v7.0

| Метрика | Hold-out | LOSO (10 субъектов) |
|---------|----------|---------------------|
| **F1-macro** | 0.801 | 0.824 ± 0.063 |
| **ROC-AUC** | 0.898 | 0.912 ± 0.043 |
| **PR-AUC** | 0.861 | — |

Ключевые гиперпараметры: `batch_size=32`, `lr=1e-4`, `weight_decay=3e-2`, `focal_gamma=2.0`, `encoder_dropout=0.3`, `classifier_dropout=0.4`.


