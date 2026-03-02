# AFC Lab — Обнаружение физической усталости по сенсорным данным

Двухветвевая нейросеть **FatigueCNN_LSTM** (CNN + BiLSTM, ≈ 474 K параметров) для бинарной классификации *fatigue / not-fatigue* на основе IMU и физиологических сигналов.

Проект объединяет **три открытых датасета** в единый композитный набор и предоставляет полный пайплайн: загрузка → предобработка → обучение → оценка (hold-out, LOSO, персонализация).

---

## Содержание

- [AFC Lab — Обнаружение физической усталости по сенсорным данным](#afc-lab--обнаружение-физической-усталости-по-сенсорным-данным)
  - [Содержание](#содержание)
  - [Архитектура модели](#архитектура-модели)
  - [Структура проекта](#структура-проекта)
  - [Источники данных](#источники-данных)
  - [Быстрый старт](#быстрый-старт)
    - [1. Установка зависимостей](#1-установка-зависимостей)
    - [2. Загрузка данных](#2-загрузка-данных)
      - [Zenodo](#zenodo)
      - [4TU](#4tu)
      - [PhysioNet](#physionet)
    - [3. Сборка композитного датасета](#3-сборка-композитного-датасета)
    - [4. Запуск ноутбука](#4-запуск-ноутбука)
  - [Конфигурация](#конфигурация)
  - [Результаты](#результаты)
  - [Лицензия](#лицензия)

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
    ┌─────────────────┐              ┌─────────────────┐
    │   IMU Encoder   │              │  Physio Encoder  │
    │  Conv1D × 3     │              │  Conv1D × 3      │
    │  (6 каналов)    │              │  (4 канала)       │
    │  → 256-dim      │              │  → 128-dim × mask │
    └────────┬────────┘              └────────┬─────────┘
             │                                │
             └───────────┬────────────────────┘
                         ▼
                  ┌──────────────┐
                  │   BiLSTM     │
                  │  hidden=128  │
                  └──────┬───────┘
                         ▼
                  ┌──────────────┐
                  │  Classifier  │
                  │  FC → Sigmoid│
                  └──────────────┘
```

- **IMU-ветка** (6 каналов: ax, ay, az, gx, gy, gz) — свёрточный энкодер [64 → 128 → 256]
- **Physio-ветка** (4 канала: BVP, EDA, TEMP, HR) — свёрточный энкодер [32 → 64 → 128], маскируется `has_physio`-флагом для окон без физиологии
- **BiLSTM** — моделирует временные зависимости поверх объединённых CNN-признаков
- **Классификатор** — полносвязные слои с dropout → бинарный выход (fatigue / not-fatigue)

---

## Структура проекта

```
afc_lab/
├── config.yaml                 # Основная конфигурация (пути, параметры, модель)
├── requirements.txt            # Python-зависимости
├── README.md
│
├── afc/                        # Библиотека I/O и утилит
│   ├── io_zenodo.py
│   ├── io_4tu.py
│   ├── io_fatigueset.py
│   ├── io_common.py
│   ├── harmonize.py
│   ├── models_tabular.py
│   └── splits_metrics.py
│
├── scripts/
│   ├── build_dataset.py        # ★ Сборка композитного NPZ из 3 источников
│   ├── make_composite.py       # Старый скрипт (2 источника)
│   ├── train_benchmarks.py     # Табличные бейзлайны (RF, XGB, SVM)
│   ├── train_deep.py           # Обучение CNN1D (устаревший)
│   └── eval_report.py          # Генерация отчётов
│
├── notebooks/
│   ├── diplom_lstm.ipynb       # ★ Основной ноутбук (CNN+LSTM, полный пайплайн)
│   ├── diplom_improved.ipynb   # CNN1D (предыдущая версия)
│   ├── diplom.ipynb            # Базовый ноутбук
│   └── practice_winter_ML.ipynb
│
├── data/
│   ├── raw/                    # Исходные данные (не в git)
│   │   ├── zenodo/             #   CSV-файлы акселерометра/гироскопа
│   │   ├── 4tu/                #   MAT-файлы с шагами при ходьбе
│   │   └── fatigueset/         #   (не используется — ментальная усталость)
│   ├── wearable-device-…/      # PhysioNet Empatica E4 (не в git)
│   └── processed/              # Результаты предобработки
│       └── composite_full.npz  #   Композитный датасет (после build_dataset.py)
│
├── results/                    # Результаты экспериментов
├── results_demo/
├── results_improved/
│
└── docs/
    └── ТЕХНИЧЕСКАЯ_ДОКУМЕНТАЦИЯ.md   # Полная техническая документация v3.0
```

---

## Источники данных

| # | Датасет | Модальности | Формат | Кол-во субъектов | Ссылка |
|---|---------|-------------|--------|------------------|--------|
| 1 | **Zenodo** — Accelerometer Dataset for Physical Activity Monitoring | IMU (acc + gyro) | CSV, 180 pts @ 256 Hz | 30 | [doi:10.5281/zenodo.11065275](https://zenodo.org/records/11065275) |
| 2 | **4TU** — Fall Detection Accelerometer Data | IMU (acc + gyro) | MAT, 150 pts @ 240 Hz | 8 | [doi:10.4121/5b7bd112](https://data.4tu.nl/datasets/5b7bd112-b27b-488e-8df9-98e5b09e0724) |
| 3 | **PhysioNet** — Wearable Device Dataset (Stress & Exercise) | IMU (acc) + Physio (BVP, EDA, TEMP, HR) | Empatica E4 CSV | ~60 | [physionet.org/content/wearable-exam-stress](https://physionet.org/content/wearable-exam-stress/1.0.1/) |

Все окна приводятся к единой длине **100 точек** (ресемплинг). Для Zenodo/4TU физиологические каналы заполняются нулями, а флаг `has_physio = False`.

---

## Быстрый старт

### 1. Установка зависимостей

```bash
# Создание виртуального окружения
python -m venv .venv

# Активация (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
# Активация (Linux/macOS)
# source .venv/bin/activate

# Установка базовых пакетов
pip install -r requirements.txt

# Установка PyTorch (для CNN+LSTM ноутбука)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# Для CPU-only варианта:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 2. Загрузка данных

Скачайте три датасета и разместите их в соответствующих папках:

#### Zenodo
Скачайте CSV-файлы с [Zenodo](https://zenodo.org/records/11065275) и поместите в `data/raw/zenodo/`:
```
data/raw/zenodo/
├── Accel_X_all.csv
├── Accel_Y_all.csv
├── Accel_Z_all.csv
├── Gyro_X_all.csv
├── Gyro_Y_all.csv
└── Gyro_Z_all.csv
```

#### 4TU
Скачайте MAT-файлы с [4TU](https://data.4tu.nl/datasets/5b7bd112-b27b-488e-8df9-98e5b09e0724) и поместите в `data/raw/4tu/`:
```
data/raw/4tu/
├── p001_HDSL_CCW_0-2k.mat
├── p001_HDSL_CW_2-4k.mat
├── p001_HDSL_postfatigue1200m.mat
├── p001_strides.mat
├── ...
└── TableFeats.mat
```

#### PhysioNet
Скачайте архив с [PhysioNet](https://physionet.org/content/wearable-exam-stress/1.0.1/) и распакуйте **внутрь `data/`**. Итоговая структура:
```
data/wearable-device-dataset-from-induced-stress-and-structured-exercise-sessions-1.0.1/
  └── wearable-device-dataset-from-induced-stress-and-structured-exercise-sessions-1.0.1/
      └── Wearable_Dataset/
          ├── AEROBIC/
          │   ├── S01/ (ACC.csv, BVP.csv, EDA.csv, HR.csv, TEMP.csv, tags.csv)
          │   ├── S02/
          │   └── ...
          └── ANAEROBIC/
              ├── S01/
              └── ...
```

> **Примечание:** путь к PhysioNet указан в `config.yaml` → `paths.physionet_root`. Если вы разместили данные иначе, обновите этот параметр или используйте `--physionet-dir` при запуске скрипта.

### 3. Сборка композитного датасета

```bash
python scripts/build_dataset.py
```

Скрипт загружает все три источника, нормализует каждый на единую длину (100 точек), и сохраняет результат в `data/processed/composite_full.npz`.

**Ключевые опции:**

| Флаг | Описание |
|------|----------|
| `--raw-dir <path>` | Путь к `data/raw` (по умолчанию из `config.yaml`) |
| `--physionet-dir <path>` | Путь к PhysioNet `Wearable_Dataset/` |
| `--out <path>` | Путь к выходному NPZ-файлу |
| `--no-physionet` | Исключить PhysioNet (собрать только Zenodo + 4TU) |

**Примеры:**

```bash
# Стандартный запуск (все 3 источника)
python scripts/build_dataset.py

# Без PhysioNet (только IMU)
python scripts/build_dataset.py --no-physionet

# Свой путь к PhysioNet
python scripts/build_dataset.py --physionet-dir "D:/data/PhysioNet/Wearable_Dataset"
```

**Выходной файл** `composite_full.npz` содержит:

| Массив | Размер | Описание |
|--------|--------|----------|
| `X_imu` | (N, 100, 6) | IMU-сигналы: ax, ay, az, gx, gy, gz |
| `X_physio` | (N, 100, 4) | Физиология: BVP, EDA, TEMP, HR |
| `y` | (N,) | Метки: 0 = not-fatigue, 1 = fatigue |
| `pids` | (N,) | Идентификатор субъекта |
| `domains` | (N,) | Источник: `zenodo`, `4tu`, `physionet` |
| `has_physio` | (N,) | `True` — есть физиоданные, `False` — нули |
| `imu_channels` | (6,) | Названия IMU-каналов |
| `physio_channels` | (4,) | Названия Physio-каналов |

### 4. Запуск ноутбука

Откройте и запустите основной ноутбук:

```bash
# В VS Code — просто откройте файл:
code notebooks/diplom_lstm.ipynb

# Или через Jupyter:
jupyter notebook notebooks/diplom_lstm.ipynb
```

Ноутбук `diplom_lstm.ipynb` выполняет полный пайплайн:

1. **Загрузка данных** — чтение `composite_full.npz`
2. **Разведочный анализ** — статистика, распределения, визуализация сигналов
3. **Модель** — определение `FatigueCNN_LSTM` (двухветвевая CNN + BiLSTM)
4. **Обучение** — hold-out 60/20/20, early stopping, cosine scheduler
5. **Оценка на тесте** — precision, recall, F1, ROC-AUC, confusion matrix
6. **LOSO** — Leave-One-Subject-Out кросс-валидация
7. **Персонализация** — двухэтапный fine-tuning на конкретном субъекте
8. **Итоговая сводка** — сохранение результатов в `results_lstm/`

> **Важно:** перед запуском ноутбука убедитесь, что файл `data/processed/composite_full.npz` существует (шаг 3).

---

## Конфигурация

Все параметры проекта находятся в `config.yaml`:

- **paths** — пути к сырым данным, обработанным данным, результатам
- **processing** — длина окна (100 пт.), каналы IMU/Physio, нормализация
- **zenodo / ftu / physionet** — параметры каждого источника
- **model** — архитектура CNN+LSTM (фильтры, ядра, LSTM, classifier)
- **training** — гиперпараметры обучения (lr, epochs, patience, scheduler)
- **splits** — схемы оценки (LOSO, cross-dataset)

---

## Результаты

Результаты экспериментов сохраняются в:

- `results_lstm/` — основные результаты CNN+LSTM ноутбука
  - `best_model_lstm.pth` — веса лучшей модели
  - `experiment_summary.json` — итоговая сводка
- `results/` — результаты скриптовых экспериментов
- `results_demo/` — демонстрационные результаты

---

## Лицензия

Учебный проект. Данные предоставлены под оригинальными лицензиями датасетов.
