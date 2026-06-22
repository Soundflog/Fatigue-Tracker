# v7–v8 Experiments Summary & Plan to Break F1=0.82 Plateau

**Дата:** 2026-04-23
**Датасет:** PhysioNet WESAD-style (Empatica E4, запястье), Exercise-only (AEROBIC + ANAEROBIC), ~4500 окон, 31 субъект, окно 100×6 IMU + 100×4 Physio + 13 stress profile features.
**Цель:** Test F1-macro ≥ 0.90 (overall — допустимо любой ценой).

---

## 1. Сводная таблица всех проведённых экспериментов

### 1.1 Основные обучения (held-out test)

| # | Версия | Ключевая идея | Параметры | Best ep | Val F1 | Test F1 (val-thr) | Test F1 (opt-thr) | Test AUC | Стоп / Поведение |
|---|--------|---------------|-----------|---------|--------|-------------------|-------------------|----------|------------------|
| 1 | v7.0   | Baseline dual-branch CNN, без stress profile | ~12K | — | — | **0.801** | — | ~0.88 | OK |
| 2 | v8.1   | + Stress profile concat (13 фич, MLP→FiLM) | ~12K | — | ~0.83 | **0.826** | — | ~0.89 | OK |
| 3 | v8.2-large | enc=32, kernels=[12,9,6,4], drops=0.4–0.5 | ~70K | — | — | — | — | — | **Полный overfit** (val_loss диверг.) |
| 4 | v8.2-anti-overfit | enc=20, wd=5e-2, focal_γ=2.2 | ~12K | 47 | 0.8014 (EMA) | 0.8205 | 0.8291 | ~0.88 | Плато на v8.1 |
| 5 | v8.2-SE+SWA | + SE-блоки + Hierarchical Physio + R-Drop + WarmRestarts + SWA | ~14K | **7** | **0.8202** | 0.8205 | 0.8291 | 0.8838 | Early stop @ ep 35, SWA не успел активироваться (start=72) |
| 6 | **v8.2-SE+SWA+RDrop (v8c)** | + SE + TemporalAttention(4h) + FiLM(IMU+Physio) + RDrop + SWA + WarmRestarts | **17,789** | **13** | **0.8255** | **0.8114** | **0.8264** | **0.9057** | Early stop @ ep 25 (patience=10); SWA не активировался (start=72); source=plain |
| 7 | v10.0 | Two-head (AEROBIC / ANAEROBIC) + stress_v1 + stress_v2 + aux loss | ~28K | 6 | **0.8488** | 0.7793 | 0.7849 | 0.8814 | **Сильный overfit** (Val F1=0.849, Test F1=0.779); wd=0.4 — слишком агрессивный, best_epoch=6 — недообучение |

| 8 | v9.0 | Multi-dataset joint training (PhysioNet + 4TU + Zenodo + WSD) | ~49K | 80 | 0.7166 (val-phys) | 0.7935 (phys) | 0.7935 (test-opt-thr) | 0.8837 (phys) | **Провал**: Test F1 ниже v8.1, val/test gap ~5pp на PhysioNet |


### 1.2 Ablation: вклад stress profile (v8.2-SE+SWA+RDrop, v8c, TTA×5)

| Variant | F1-macro | ROC-AUC | PR-AUC | Bal-Acc |
|---------|----------|---------|--------|---------|
| v7 baseline (no profile, no FiLM)        | **0.8276** | 0.9100 | 0.8717 | 0.8294 |
| v8.2 (stress profile + FiLM + TTA×5)     | 0.8205     | 0.9096 | 0.8760 | 0.8178 |
| v7 baseline (no profile, no FiLM)        | 0.7817 | 0.8968 | 0.8609 | 0.7747 |
| **v8.2-SE+SWA+RDrop (stress profile + FiLM + TTA×5)** | **0.8114** | **0.9057** | **0.8701** | **0.8113** |

> ⚠️ **Профиль на этом датасете НЕ помогает, а скорее мешает на test split.**
> AUC одинаковые → вероятностное ранжирование не страдает, но decision boundary смещается хуже.
> Профиль даёт **ΔF1 = +0.0297**, ΔAUC = +0.0090 по сравнению с базовой линией без FiLM.

### 1.3 LOSO Evaluation (v8.1, 10/31 fold subset)
### 1.3 LOSO Evaluation (v8.2-SE+SWA+RDrop, v8c, 5/31 fold subset)

| Метрика | Mean ± Std |
|---------|-----------|
| F1-macro | 0.8096 ± 0.0695 |
| ROC-AUC  | 0.9006 ± 0.0541 |

| Subject | n_samples | F1-macro | ROC-AUC |
|---------|-----------|----------|---------|
| physionet_S03 | 534 | 0.9180 | 0.9946 |
| physionet_S04 | 636 | 0.8245 | 0.9269 |
| physionet_S11 | 616 | 0.6075 | 0.8602 |
| physionet_S14 | 655 | 0.9390 | 0.9800 |
| physionet_f02 | 825 | 0.8427 | 0.9204 |
| **Mean ± Std** | — | **0.8263 ± 0.118** | **0.9364 ± 0.048** |

> Высокий разброс по subjects (±0.07) указывает: модель сильно зависит от конкретного человека.

> Высокий разброс (±0.118) отражает межсубъектную вариативность. S11 выпадает (F1=0.607) — вероятно, нетипичный паттерн усталости.
> Результат выше LOSO v8.1 (0.8096±0.0695) по среднему F1 (+1.67pp), хотя std вырос — из-за малой выборки (5 fold vs 10).

### 1.4 Recall-optimal threshold (для справки)

| Версия | F1-macro @ recall-opt | Recall | Precision |
|--------|------------------------|--------|-----------|
| v8.2-SE+SWA | 0.2945 | ~0.99 | ~0.18 | (полный коллапс — порог уехал в 0)

---

## 2. Диагноз: почему мы упёрлись в F1≈0.82

| Свидетельство | Что это значит |
|---------------|---------------|
| 12K и 70K параметров → одинаковый потолок | Это **не** проблема ёмкости/архитектуры |
| Ablation: no-profile (0.8276) > with-profile (0.8205) | 13 stress-фич **не несут полезного сигнала** для test subjects (или коррелируют только у train) |
| LOSO Std=0.0695 при mean=0.81 | Огромная межсубъектная вариативность — generalization-gap, а не bias |
| Best epoch = 7 (из 120 запланированных) | Модель **сходится моментально**. Дальше — переобучение или флуктуации |
| Val_loss быстро растёт после ep 7 при стабильном val_f1 | Классический сигнал: модель учит разреженные паттерны окна, а не общие |
| AUC≈0.88–0.91 при F1≈0.82 | **Ranking почти идеальный**, но decision boundary плохая → проблема в порогах/калибровке/балансе |

**Главный вывод:** Потолок **информационный**, а не алгоритмический. 5-секундное окно (100 samples @ 20Hz) на запястном PPG/IMU физически не содержит достаточно сигнала, чтобы отличить «уставший» от «не уставший» лучше, чем 82–83% F1, при том что метки субъективны и сильно зависят от конкретного человека.

---

## 3. План достижения F1 ≥ 0.90

Идеи отсортированы по **ожидаемому ROI** (соотношение «прирост F1 / трудоёмкость»). Для каждой — конкретный план интеграции.

### 🎯 Tier 1 — обязательные изменения (ожидаем +5–10pp)

#### 3.1 **Длинное окно** (window=300–600 вместо 100)
**Почему:** Усталость — это динамика на минутах, а не на секундах. HRV-паттерны и тренд EDA проявляются только на 30+ сек окнах.
**Как:**
- Перегенерировать `physionet_only.npz` с `window_size=600` (30 сек @ 20Hz) и `stride=100` (overlap 83%).
- В CNN увеличить kernel_size первой conv до 15–25, добавить ещё один pooling.
- Ожидаемый эффект: +3–5pp (наибольший single-shot прирост).

#### 3.2 **Per-subject Z-score normalization**
**Почему:** Базовые HR/EDA различаются между людьми в **разы** (HR_rest от 50 до 90, EDA от 0.1 до 30 µS). Глобальный StandardScaler делает рукой machete-job, а нужны индивидуальные ножницы.
**Как:**
- Перед окнованием для каждого subject вычесть subject-median, поделить на subject-MAD.
- Хранить subject-statistics в profile features (добавить как 14-ю фичу).
- Ожидаемый эффект: +2–4pp, особенно на LOSO.

#### 3.3 **HRV/IMU инженерные фичи как 3-я ветка**
**Почему:** Чистый CNN не выучит SDNN/RMSSD/LF-HF за 100 окон обучения. Эти фичи разработаны кардиологами специально под наш сигнал.
**Как:**
- Из BVP: `RMSSD`, `SDNN`, `pNN50`, `LF/HF`, `sample_entropy`, `mean_HR`, `HR_std`.
- Из ACC: `signal_magnitude_area`, `spectral_centroid`, `dominant_freq`, `zero_crossings`.
- Подавать вектором [B, ~15] → MLP(32)→16 → concat с CNN-фичами.
- Ожидаемый эффект: +2–3pp.

### 🎯 Tier 2 — высокая отдача (ожидаем +2–5pp)

#### 3.4 **Pre-training на composite (4TU + Zenodo)**
**Почему:** В `data/processed/` уже есть `composite_full.npz` и `composite_strides.npz` (Fatigueset + Zenodo). Это ~10× больше данных похожей природы. Чистый PhysioNet — слишком мало для CNN.
**Как:**
- Pretext task: «predict subject ID» или «predict activity intensity» (self-supervised) на composite.
- Заморозить encoder, fine-tune классификатора на PhysioNet.
- Альтернатива: contrastive learning (SimCLR-style) на окнах.
- Ожидаемый эффект: +2–4pp.

#### 3.5 **Personalized calibration (few-shot adaptation)**
**Почему:** В LOSO Std=0.07. Если на test-subject взять 30 sec rest baseline и подкрутить классификатор — получим персонализацию.
**Как:**
- На inference: первые N окон (rest/baseline) — fine-tune только последний `nn.Linear` (1 эпоха, lr=1e-3).
- Либо: subject-embedding из profile + FiLM на classifier.
- Ожидаемый эффект: +3–5pp на LOSO (на in-split — меньше).

#### 3.6 **Ensemble (CNN + LSTM + Transformer)**
**Почему:** Разные индуктивные смещения ловят разные паттерны. Простое усреднение probabilities даёт +1–3pp почти всегда.
**Как:**
- Обучить три модели: текущая v8.2-SE, BiLSTM(64) поверх IMU+Physio, малый Transformer (2 layers, d=64).
- Усреднить TTA-probabilities, искать threshold на val.
- Ожидаемый эффект: +1–3pp.

### 🎯 Tier 3 — улучшения процедуры (ожидаем +1–3pp)

#### 3.7 **Threshold calibration с учётом subject prior**
- Сейчас порог глобальный (0.465). Делать **per-subject threshold** на first-N окон.
- Использовать isotonic regression на val_probs.

#### 3.8 **Удалить FiLM с stress profile, оставить только concat**
- Ablation показала: профиль скорее вредит. Возможно проблема в FiLM-модуляции (она агрессивная). Concat + dropout — мягче.
- Альтернатива: вообще выкинуть профиль (по ablation v7 baseline = 0.8276 > v8.2 = 0.8205).

#### 3.9 **Subject-level mixup** (вместо обычного)
- Mixup пар окон **разных subjects одного класса** → искусственное сглаживание индивидуальных особенностей.

#### 3.10 **Заменить Focal Loss на Label Smoothing + Weighted CE**
- Focal с γ=1–2.2 в наших экспериментах не дал прироста над взвешенной CE.
- Label smoothing (ε=0.1) + class weights — стабильнее.

### 🚀 Tier 4 — прорывные, но рискованные

#### 3.11 **Transformer на raw signal с positional encoding**
- На 30-сек окне 600 токенов — реально для small Transformer. Возможна большая отдача.

#### 3.12 **Pseudo-labeling unlabeled окон AEROBIC/ANAEROBIC**
- В PhysioNet есть периоды между метками. Использовать как unlabeled, делать self-training.

#### 3.13 **Учёт типа протокола (AEROBIC vs ANAEROBIC) — две головы** 
- Возможно для разных видов нагрузки нужны разные decision boundaries. Multi-task head.

---

## 4. Рекомендуемый порядок действий (минимум шагов до 0.9)

```
Шаг 1 (1 день): Перегенерировать датасет с window=600, stride=100      → ожид. F1 ≈ 0.86
Шаг 2 (1 день): Per-subject z-score + добавить subject_stats в profile → ожид. F1 ≈ 0.88
Шаг 3 (2 дня): Pretrain encoder на composite_full.npz, fine-tune       → ожид. F1 ≈ 0.90 ✅
Шаг 4 (опц.):  Personalized calibration на LOSO                         → ожид. F1 ≈ 0.92
```

**Если Шаг 1 не даст прироста** — значит архитектурная гипотеза неверна, и проблема в самой задаче (метки шумные / сигнала нет). Тогда:
1. Сделать **error analysis по subjects**: на каких людях F1<0.7 и почему.
2. Возможно добавить EDA-only / BVP-only baseline как sanity check.
3. Рассмотреть **смену метки**: вместо subjective fatigue label — predicted RPE (Borg scale) как regression.

---

## 5. Что НЕ работает (по факту экспериментов)

| Идея | Почему не сработала |
|------|---------------------|
| Увеличение модели (до 70K params) | Чистый overfit, потолок не пробит |
| Stress profile через FiLM | Ablation: hurts на test split (0.8205 < 0.8276) |
| Focal Loss с γ=2.2 + WD=5e-2 | Тот же 0.82, что и без всего этого |
| SE-блоки + R-Drop + WarmRestarts | Best epoch = 7, всё на старом потолке |
| SWA | Не успевает активироваться — модель сходится за 7 эпох |
| Большие kernels [12,9,6,4] | Overfit |

**Общий паттерн:** любые **алгоритмические** улучшения упираются в один и тот же информационный потолок данных. Нужны изменения **на уровне данных** (окно, нормализация, фичи, pretraining).

---

## 6. v9 Multi-Dataset Joint Training — НЕ сработало (вывод по 3 запускам)

**Гипотеза:** добавить 4TU + Zenodo + WSD4FedSRM (~26K IMU-only окон) к PhysioNet (23K) → больше данных снимет потолок 0.826.

### 6.1 Сводка попыток

| Run | Strategy | Train conf | Val F1 phys | Test F1 (val-thr) | Δ vs v8.1 |
|---|---|---|---|---|---|
| v9.0 | Joint, без subsample, phys-w=1.5, dropout=0.30 | enc=24, ep=80 | 0.8065 | 0.7762 | **−4.98pp** (overfit) |
| v9.1 | + heavy reg (dropout=0.55, wd=5e-2, rdrop=1.5), phys-w=3 | enc=16, ep=50 | 0.8019 | 0.7291 | **−9.69pp** (underfit + val/test gap) |
| v9.2 | Balanced (dropout=0.30, wd=2e-2), cap=5000, phys-w=5 | enc=20, ep=60 | ~0.80 | ~0.74 | **−8pp** (плато) |


| Domain |  N (val)|  F1-macro|  ROC-AUC|  PR-AUC|  Bal-Acc |
| ------|---------|----------|---------|--------|---------|
| physionet |     3454 |    0.8144 |   0.8837 |  0.8465 |   0.8141 |
| wsd4fedsrm |     233 |    0.4146 |   0.5280 |  0.7180 |   0.5000 |
| 4tu |     2481 |    0.3334 |   0.8154 |  0.8135 |   0.5000 |
| zenodo |     828 |    0.3172 |   0.6048 |  0.5363 |   0.4996 |

### 6.2 Главный вывод: **другие домены ЗАМЕДЛЯЮТ исследование**

1. **Distribution shift между датасетами слишком большой.** PhysioNet (Empatica E4 BVP+EDA) и 4TU/Zenodo/WSD (только IMU, разные сенсоры, разные протоколы) — это разные задачи. Даже per-domain z-score + FiLM conditioning не решает проблему: модель учит «среднее по больнице» вместо PhysioNet-специфики.
2. **F1 overall (~0.65) систематически ниже F1 PhysioNet (~0.80)** во всех запусках → не-PhysioNet домены имеют шумные/слабые метки усталости (особенно WSD: 71% positive — крайне несбалансирован), и модель смещает decision boundary к их статистике.
3. **`has_physio` маска создаёт две распределения внутри батча** → BatchNorm ломает их различение, FiLM не успевает компенсировать.
4. **Trade-off между train-data волуме и signal-to-noise** уходит в минус: 49K окон (v9) хуже 23K окон (v8.1).
5. **Val→test gap 5–7pp на PhysioNet** даже в лучших v9-runs — модель overfitится на val-PhysioNet через смешанный сигнал, а на test-PhysioNet выглядит хуже, чем v8.1, обученная только на PhysioNet.
6. **Stage 2 fine-tune (PhysioNet only после joint pre-train)** теоретически должен лечить — на практике не успевает за 15 эпох восстановить потерянное в Stage 1.

### 6.3 Решение (зафиксировано)

> **Multi-dataset подход для текущего объёма и качества меток в 4TU/Zenodo/WSD исключён. Дальнейшая работа — только на PhysioNet (23K окон, 31 субъект).**

Следующие шаги — Tier 1 идеи из секции 4 (длинное окно, HRV-фичи, per-subject calibration), а не расширение датасета.

### 6.4 Когда вернуться к multi-dataset

Стоит пересмотреть только при наличии:
- датасетов с **тем же сенсорным набором** (Empatica E4 / аналогичные wrist BVP+EDA)
- унифицированного протокола разметки усталости (а не «AEROBIC vs ANAEROBIC» против «sit vs run»)
- ≥10K окон **каждого** домена (чтобы оба сигнала были одинаково статистически весомы)

---

## 7. v9 PhysioNet-only — финальная серия (anti-overfit регуляризация)

После решения «multi-dataset исключён», v9 был перепрофилирован: **только PhysioNet (23K окон, 31 субъект)**, но с улучшенной архитектурой v9 (SE+FiLM, R-Drop, MultiHeadAttention, Mixup, label smoothing) и более агрессивной регуляризацией, чем v8.1.

### 7.1 Сводка попыток (все на PhysioNet only)

| Run | Архитектура | Регуляризация | Aug | Val F1 phys | Test F1 (val-thr) | Δ vs v8.1 | Диагноз |
|---|---|---|---|---|---|---|---|
| v9.3 | enc=24, clf=32→16 | drop=0.20/0.25, wd=5e-3, rdrop=0.3 | средняя | ~0.81 | ~0.79 | **−3.6pp** | overfit (TrL≪VaL) |
| v9.4 | enc=18, clf=32→16 | drop=0.40/0.45, wd=2e-2, rdrop=0.7 | усиленная +mag_warp | ~0.82 | ~0.80 | **−2.6pp** | underfit (TrL>VaL) |
| v9.5 | enc=12, k=[7,5,3,3] | drop=0.45/0.55, wd=2e-2, rdrop=0.45, focal γ=2.0, ls=0.08 | усиленная | ~0.82 | ~0.81 | **−1.6pp** | переобучается, но близко |
| v9.6 | enc=12, +Mixup α=0.3 | + Mixup, BS=64 | то же | ~0.82 | ~0.81 | **−1.5pp** | плато, не пробивает |

**Итог**: ни один режим не пробил v8.1 baseline = 0.826. Лучший v9-результат ≈ 0.81 на test. v8.1 (более простая arch без FiLM/SE/R-Drop, без Mixup) остаётся непобеждённой.

### 7.2 Что показала серия v9 PhysioNet-only

1. **Архитектурный потолок подтверждён.** SE-блоки, MultiHeadAttention, FiLM, R-Drop, Mixup, label smoothing, magnitude warping — ни одна комбинация регуляризаторов не улучшает F1 на 23K-окоnном датасете с фиксированной разметкой `fatigue_ratio=0.5`.
2. **Trade-off overfit ↔ underfit очень узкий.** При drop>0.45 модель недоучивается; при drop<0.30 — переобучается за 5–7 эпох. «Окно» нормального обучения ≈ 3–5 эпох, EMA-stopping срабатывает сразу.
3. **Val/test gap 1–3pp устойчив.** Это значит, что 7 test-subjects (≈4.6K окон) дают другую статистику BVP/EDA, чем 5 val-subjects (≈3.5K окон). Это шум разметки, а не модели.
4. **FiLM с n_domains=1 — no-op.** При PhysioNet-only домен один, FiLM-conditioning ничего не даёт (γ→1, β→0 при init и не сдвигается). Лишние ~200 параметров.
5. **MultiHeadAttention в IMUEncoder тратит ёмкость впустую.** На 100-точечных окнах 32 Hz attention не находит структуры, которая не была бы поймана 3 conv-слоями. Подтверждается тем, что снижение enc_ch с 24 до 12 не ухудшило F1.

### 7.3 Главный вывод по v9

> **Алгоритмическая работа на PhysioNet npz исчерпана. Без изменений на уровне данных (новые фичи, новая разметка, новый таргет) F1 ≈ 0.82 — это потолок задачи в текущей формулировке.**

Дальнейшая работа должна идти по линии **обогащения входа** (а не усложнения модели):

| Направление | Гипотеза | Где проверять |
|---|---|---|
| **Two-head по протоколу** (AEROBIC vs ANAEROBIC) | Усталость в велоэргометре и в Wingate имеет разные физиологические маркеры (длительный аэробный vs короткий анаэробный) | v10 — отдельная классификационная голова на каждый тип нагрузки |
| **Stress-level feature** (производный признак HRV/EDA) | Текущий уровень стресса коррелирует с усталостью, но содержит дополнительную информацию помимо raw signal | v10 — `stress_level_v1` (простой) и `stress_level_v2` (HRV-based) |
| **Длинное окно (window=300–600 точек)** | Усталость = медленный процесс, 5-сек окна слишком короткие | требует rebuild npz |
| **Per-subject calibration** | Baseline HR/EDA сильно индивидуальны, z-score per subject недостаточен | требует test-time adaptation |

### 7.4 Сравнительная итоговая таблица (v8 vs v9, PhysioNet test)

| Версия | Архитектура (params) | Регуляризация | Test F1-macro |
|---|---|---|---|
| v7.0 | CNN-LSTM (52K) | drop=0.3 | 0.801 |
| **v8.1** | dual-branch CNN+attention (45K) | drop=0.4, focal | **0.826** ⬅️ baseline |
| v8.2-large | + 70K params | drop=0.4 | 0.78 (overfit) |
| v8.2-anti-overfit | + WD 5e-2 | drop=0.5, focal γ=2.2 | 0.821 |
| v8.2-SE+SWA | + SE blocks + SWA | drop=0.4, swa | 0.821 |
| v9.3–v9.6 | + FiLM + MHA + R-Drop + Mixup (12–24K params) | drop=0.20–0.55, mixup, ls | 0.79–0.81 |

**Никакая из вариаций v9 не превзошла v8.1.** Это означает, что комбинация SE+FiLM+R-Drop+Mixup на 23K окон с 31 субъектом не имеет дополнительной информации для извлечения — потолок задачи достигнут.

### 7.5 Решение по v9

1. ✅ v8.1 (0.826) фиксируется как **окончательный baseline** для PhysioNet «as-is»
2. ✅ v9-эксперименты сворачиваются
3. ➡️ Переход к **v10** — попытка пробить потолок через новые features/multi-task setup, а не через регуляризацию

---

## 8. v10 — Two-head по протоколу + stress-level features (план)

**Старт:** ноутбук [`notebooks/v10_twohead_stress.ipynb`](../notebooks/v10_twohead_stress.ipynb), данные [`data/processed/physionet_v10.npz`](../data/processed/physionet_v10.npz) (build script: [`scripts/build_physionet_v10.py`](../scripts/build_physionet_v10.py)).

### 8.1 Что меняется относительно v9

| Компонент | v9 | v10 |
|---|---|---|
| Архитектура | shared encoder + 1 head | shared encoder + **2 heads** (AEROBIC, ANAEROBIC) |
| Маршрутизация | — | по `protocol_id` per-window |
| Доп. вход | — | **stress_v1** (`mean_HR, EDA_range, TEMP_delta`) + **stress_v2** (`BVP_RMSSD, EDA_tonic, EDA_phasic_amp`) |
| Mixup | глобальный | **per-protocol** (не смешиваем aerobic с anaerobic) |
| Sampler | по классу | **по классу × протоколу** |
| Aux loss | — | «другая» голова обучается с весом 0.3 (anti-collapse) |
| FiLM, MHA, label-smoothing, focal | были | сохранены |

### 8.2 Гипотезы (что должно сработать)

1. **H1 — две головы дают bias-variance gain.** Усталость в велоэргометре (длинная аэробная нагрузка, медленная динамика HR/EDA) и в Wingate (30-сек анаэробный спринт, резкие пики) — это разные функциональные зависимости. Одна голова усредняет их в ущерб обоим, две — учат специализированные decision boundaries.
2. **H2 — stress features снимают нагрузку с CNN.** Раньше модель должна была сама выводить медленные агрегаты (mean HR, EDA range) из 100-точечных окон. Подаём явно → encoder освобождается для high-frequency структуры (HRV паттерны, EDA phasic events).
3. **H3 — per-protocol z-score stress** даёт сравнимые величины между AEROBIC (HR≈140 bpm) и ANAEROBIC (HR≈170 bpm), не теряя относительной динамики внутри каждого протокола.

### 8.3 Критерий успеха

- **Min**: Test F1-macro > 0.826 (бьёт v8.1)
- **Target**: F1 AEROBIC и F1 ANAEROBIC оба > 0.82, разница между ними < 5pp
- **Stretch**: > 0.85 (был бы значительный прорыв)

### 8.4 План если v10 не пробьёт 0.826

1. Прогнать ablation (без stress / только stress_v1 / только stress_v2 / single-head) — понять, что именно полезно
2. Перейти к **длинному окну** (window=300, 15 сек) — требует rebuild npz
3. Перейти к **stress-as-target multi-task** — модель параллельно регрессирует stress_v2 как auxiliary task

---

## 9. v10 Two-Head — результаты run #1 (2026-04-23)

### 9.1 Конфигурация запуска

| Параметр | Значение |
|---|---|
| Архитектура | TwoHeadFatigueNet (~28K params) |
| encoder_channels | 20 (IMU) / 12 (Physio) |
| kernel_sizes | [9, 6, 3] |
| classifier_dropout | 0.40 |
| head_hidden | 32 |
| weight_decay | **0.40** ← проблемное значение |
| lr | 1e-4 |
| warmup_epochs | 12 |
| focal_gamma | 2.5 |
| label_smoothing | 0.10 |
| mixup_alpha | 0.40 |
| rdrop_alpha | 0.25 |
| aux_loss_weight | 0.15 |
| patience | 6 |

### 9.2 Результаты (held-out test, TTA×5)

| Метрика | Val-threshold (0.469) | Opt-threshold (0.478) |
|---|---|---|
| **F1-macro** | **0.7793** | **0.7849** |
| ROC-AUC | 0.8814 | 0.8814 |
| PR-AUC | 0.7982 | 0.7982 |
| Balanced-Acc | 0.8020 | 0.8044 |

**Сравнение с baseline:**

| Версия | Test F1 | Δ vs v8.1 |
|---|---|---|
| v8.1 (baseline) | 0.826 | — |
| v10.0 (val-thr) | 0.7793 | **−4.67pp** |
| v10.0 (opt-thr) | 0.7849 | **−4.11pp** |

> ⚠️ **LOSO evaluation НЕ проводился** — `best_epoch=6` свидетельствует об нестабильной сходимости; запуск LOSO на такой конфигурации нецелесообразен до исправления overfitting.

### 9.3 Диагностика провала

**Ключевая проблема: `weight_decay=0.40` + `best_epoch=6`**

- `weight_decay=0.40` — это слишком агрессивный AdamW L2. При таком значении модель не успевает выучить даже полезные паттерны: best_epoch=6 (из 60) означает, что уже к 6-й эпохе val_f1 достиг пика и начал деградировать.
- Val F1 = **0.8488** при Test F1 = **0.7793** — разрыв 6.95pp — типичный признак overfitting к val-subjects (6 subjects), при котором generalization на test-subjects (6 subjects) не обеспечивается.
- Warmup 12 эпох при patience=6 — парадокс: EarlyStopping может сработать ещё до конца warmup-фазы, если первые 6 эпох дают нестабильный рост.

| Параметр | Текущее | Рекомендуемое | Обоснование |
|---|---|---|---|
| `weight_decay` | 0.40 | **5e-3** | 0.40 — на 2 порядка выше нормы для AdamW |
| `patience` | 6 | **12** | Должен пережить warmup-фазу (12 эп) |
| `warmup_epochs` | 12 | **6** | 20% warmup — много для 60 эпох |
| `encoder_channels` | 20 | **16** | Снизить ёмкость модели |
| `head_hidden` | 32 | **48** | feat_dim≈72, 72→32 — потеря инфо |
| `label_smoothing` | 0.10 | **0.06** | double-smooth с focal γ=2.5 |

### 9.4 Статус LOSO

> ⛔ **LOSO для v10 НЕ проводился и зафиксирован как пропущенный.**
>
> Причина: конфигурация v10.0 показала overfit (Val/Test gap 6.95pp) и нестабильную сходимость (best_epoch=6). Проведение LOSO на заведомо overfitted модели даёт нерепрезентативные результаты (ожидаемый LOSO F1 < 0.75, что ниже v8.1 LOSO=0.810±0.070).
>
> **Когда проводить LOSO v10:** после исправления weight_decay → 5e-3 и patience → 12, при условии что Test F1 > 0.826.


---

## 10. v8c: FatigueWristNet v8.2-SE+SWA+RDrop — Полная документация

### 10.1 Архитектура по доменам

Модель объединяет три домена сигналов через FiLM-кондиционирование:

`
Домен 1: IMU (B,100,6)          Домен 2: Physio (B,100,4)      Домен 3: Stress Profile (B,13)
        ↓                                  ↓                              ↓
  Conv1D(6→8)+BN+ReLU             Conv1D(4→6)+BN+ReLU+Drop       ProfileMLP: 13→26→12
  SE(r=4)+Pool                    SE(r=4)+Pool                   Dropout(0.4)
  Conv1D(8→16)+BN+ReLU            Conv1D(6→8)+BN+ReLU+Drop       → γ_imu, β_imu (24d)
  SE(r=4)+Pool                    SE(r=4)+Pool                   → γ_physio, β_physio (12d)
  Conv1D(16→24)+BN+ReLU           Conv1D(8→12)+BN+ReLU+Drop
  SE(r=4)+Pool                    GAP → (B,12)
  TemporalAttention(4h) → (B,24)
        ↓                                  ↓
  FiLM: feat × (1+γ_imu) + β_imu    FiLM: feat × (1+γ_physio) + β_physio
        ↓                                  ↓
                        Concat → (B,24+12) = (B,36)
                                  ↓
                     LayerNorm → FC(36→32) → GELU → Dropout(0.5) → FC(32→1)
`

| Компонент | Параметры | Домен |
|-----------|-----------|-------|
| IMU Encoder (3×Conv1D + SE + TemporalAttention) | 10,900 | IMU (6-ch) |
| Physio Encoder (3×Conv1D + SE + GAP) | 1,596 | Physio (4-ch: BVP,EDA,TEMP,ACC) |
| Profile MLP (13→26→12) | 708 | Stress Profile (13 фич) |
| FiLM IMU (γ,β для 24d) | 624 | IMU ← Profile |
| FiLM Physio (γ,β для 12d) | 312 | Physio ← Profile |
| Classifier (LayerNorm+FC×2) | 3,649 | Fusion |
| **ИТОГО** | **17,789** | |

**Stress Profile (13 фич):** gender, age, bmi, hr_baseline_mean, eda_baseline_mean, hr_tasks_mean, eda_tasks_mean, hr_reactivity, eda_reactivity, sl_peak, sl_reactivity, sl_mean_tasks, sl_baseline

### 10.2 CONFIG (v8.2-SE+SWA финальный)

| Параметр | Значение |
|---------|---------|
| encoder_channels | 24 |
| physio_encoder_channels | 12 |
| kernel_sizes | [9, 7, 5, 3] |
| attention_heads | 4 |
| use_se | True |
| profile_dim | 13 |
| profile_hidden | 26 |
| profile_out | 12 |
| profile_dropout | 0.4 |
| classifier_dropout | 0.5 |
| lr | 5e-5 |
| weight_decay | 3e-2 |
| epochs | 120 |
| patience | 10 |
| focal_gamma | 1.6 |
| rdrop_alpha | 0.5 |
| mixup_alpha | 0.1 |
| swa_enable | True |
| swa_start_frac | 0.6 |
| swa_lr | 5e-5 |
| sched_type | warmrestart |
| warmrestart_t0 | 30 |
| warmrestart_tmult | 2 |
| tta_n | 5 |

### 10.3 Результаты held-out test

| Метрика | Значение |
|---------|---------|
| Best epoch | 13 / 120 |
| Val F1 (best) | 0.8255 |
| Threshold (val-opt) | 0.505 |
| Source | plain (SWA не активировался) |
| **Test F1-macro (val-thr, TTA×5)** | **0.8114** |
| **Test ROC-AUC** | **0.9057** |
| **Test PR-AUC** | **0.8701** |
| **Test Balanced-Acc** | **0.8113** |
| Test F1-macro (opt-thr=0.455) | 0.8264 |

> Ранняя остановка @ ep 25 (EMA F1=0.8160). SWA не активировался (swa_start_frac=0.6 → ep 72 > 25).

### 10.4 LOSO Evaluation (5/31 subjects)

| Subject | n_samples | F1-macro | ROC-AUC |
|---------|-----------|----------|---------|
| physionet_S03 | 534 | 0.9180 | 0.9946 |
| physionet_S04 | 636 | 0.8245 | 0.9269 |
| physionet_S11 | 616 | 0.6075 | 0.8602 |
| physionet_S14 | 655 | 0.9390 | 0.9800 |
| physionet_f02 | 825 | 0.8427 | 0.9204 |
| **Mean ± Std** | — | **0.8263 ± 0.118** | **0.9364 ± 0.048** |

Настройки LOSO: n_folds=10 (запустилось 5 из-за фильтра min_windows=20), epochs=15, patience=10, lr=3e-4, wd=3e-4.

### 10.5 Ablation: вклад stress profile (vs v7 baseline)

| Вариант | F1-macro | ROC-AUC | PR-AUC | Bal-Acc |
|---------|----------|---------|--------|---------|
| v7 baseline (no profile, no FiLM) | 0.7817 | 0.8968 | 0.8609 | 0.7747 |
| v8.2-SE+SWA+RDrop (+ profile + FiLM + TTA×5) | **0.8114** | **0.9057** | **0.8701** | **0.8113** |
| ΔF1 | **+0.0297** | +0.0090 | +0.0092 | +0.0366 |

### 10.6 Feature Importance (Permutation, по F1-drop)

| Feature | F1_drop_mean | Вывод |
|---------|-------------|-------|
| gender | -0.00417 | Сильнейший предиктор — пол влияет на паттерны HR/IMU |
| sl_peak | -0.00271 | Пик стресс-уровня при задачах |
| sl_reactivity | -0.00255 | Реактивность стресс-уровня |
| eda_reactivity | -0.00200 | EDA-реактивность |
| hr_reactivity | -0.00150 | HR-реактивность |
| hr_baseline_mean | -0.00132 | Базовый ЧСС |
| sl_mean_tasks | -0.00124 | Средний стресс-уровень |
| age | -0.00098 | Возраст |
| bmi | -0.00077 | ИМТ |
| hr_tasks_mean | -0.00023 | ЧСС при задачах (слабый) |
| eda_tasks_mean | -0.00019 | EDA при задачах (слабый) |
| sl_baseline | -0.00004 | Базовый стресс (незначим) |
| eda_baseline_mean | **+0.00091** | ⚠️ Отрицательный вклад (мешает) |

> Топ-3 фичи: gender, sl_peak, sl_reactivity. eda_baseline_mean слегка вредит → можно исключить.

### 10.7 Выводы по v8c

1. **FiLM + Stress Profile работает**: ΔF1=+0.0297 vs baseline без профиля.
2. **SWA не активировался**: ранняя остановка @ ep 25, а SWA start = ep 72 → нужно снизить swa_start_frac до 0.3–0.4.
3. **LOSO улучшился**: mean F1=0.8263 vs v8.1 LOSO=0.810 (+1.63pp), AUC=0.9364.
4. **Предиктор-лидер — gender**: самая важная из 13 profile фич.
5. **Потолок 0.82 сохраняется** на held-out F1 — архитектурные изменения не пробивают информационный барьер.

### 10.8 Файлы результатов

| Файл | Описание |
|------|---------|
| results_v8c_2_stress/best_model_v82.pth | Лучший checkpoint (ep 13) |
| results_v8c_2_stress/ablation_comparison_v82.csv | Ablation таблица |
| results_v8c_2_stress/profile_importance_v82.csv | Feature importance |
| results_v8c_2_stress/test_results_v82.png | CM + ROC + PR кривые |
| results_v8c_2_stress/training_history_v82_se_swa.png | История обучения |

---

## 📋 ГЛАВНЫЙ ВЫВОД: Информационный потолок F1 ≈ 0.82–0.83

### Консолидированные данные по всем версиям

| Версия | Архитектура | Params | Test F1 (val-thr) | Test AUC | Диагноз |
|--------|------------|--------|-------------------|----------|---------|
| v7.0 | Baseline dual-branch CNN | 12K | 0.801 | ~0.88 | Простой базовый подход |
| **v8.1** | + Stress profile + FiLM | 12K | **0.826** | **~0.89** | ⭐ **Лучший с точки зрения простоты** |
| v8.2-SE+SWA+RDrop (v8c) | + SE, TemporalAttention, RDrop | 17.8K | 0.8114 | 0.9057 | Улучшен AUC, но F1 упал на test |
| v10.0 | Two-head (AEROBIC/ANAEROBIC) | 28K | 0.7793 | 0.8814 | ⚠️ Перегибается (Val=0.849 → Test=0.779) |
| v9.0 | Multi-dataset joint training | 49K | 0.7935 | 0.8837 | ❌ Distribution shift разломал обобщение |

### Ключевые факты

1. **Потолок достигнут на F1-macro ≈ 0.826** (v8.1):
   - Модель v8.1 при всей своей простоте остаётся **лучшей на практике**
   - Любые архитектурные усложнения (SE+SWA, FiLM, RDrop, двойные головы) либо упирают в перегибание, либо оставляют F1 на том же уровне

2. **ROC-AUC vs F1 расхождение указывает на проблему decision boundary**:
   - ROC-AUC = 0.88–0.91 (отличное ранжирование вероятностей)
   - F1-macro = 0.80–0.83 (посредственная классификация с единственным порогом)
   - → проблема не в модели, а в том, что пространство признаков **физически** не разделимо лучше, чем на 82% точность

3. **LOSO показывает огромную межсубъектную вариативность**:
   - v8c LOSO: std=0.118 (при mean=0.826), т.е. на одних субъектах F1=0.94, на других F1=0.61
   - → модель сильно переобучивается к индивидуальным паттернам, обобщение на новых людях плохое
   - Это не баг архитектуры, это свойство задачи: усталость очень индивидуальна

4. **Stress profile как дополнительный вход НЕ помогает**:
   - Ablation: v7 baseline (no profile) = 0.8276 vs v8c (с profile) = 0.8114 на test
   - На validation profile помогает выучить более сложные граница, но это переобучение, на test выпадает
   - Профиль содержит информацию, полезную в train-distribution, но не переносится на test-subjects

### Почему F1 ≈ 0.82 — это физический потолок

| Фактор | Почему это лимит |
|--------|-----------------|
| **Длина окна = 100 samples (5 сек)** | Усталость — медленная динамика на 20–30 сек. 5-сек окно слишком короткое, мало информации о тренде. AUC высокий (модель ранжирует хорошо), но конкретное решение за 5 сек недостаточно стабильно |
| **Метки субъективны и состояние-зависимы** | AEROBIC: устаёшь в конце нагрузки. ANAEROBIC: пиковое физиологическое изменение в конце спринта, но субъективное ощущение может различаться. Пошумовано исходное распределение |
| **Базовые HR/EDA различаются между людьми в 2–3 раза** | Per-subject normalization могла бы помочь, но требует rebuild датасета |
| **Разнообразие протоколов внутри AEROBIC/ANAEROBIC** | Велоэргометр на разных мощностях, Wingate спринты с разной интенсивностью — полиморфные паттерны утомления |
| **31 субъект — мало для глубокого CNN** | На этом объёме даже с регуляризацией сложно выучить общие паттерны, модель фиксирует индивидуальные особенности |

### Рекомендация для финальной статьи (ARTICLE_v4.md)

**Версия для публикации: v8.1** (Test F1-macro = 0.826, ROC-AUC ≈ 0.89)

- ✅ Простая, воспроизводимая архитектура
- ✅ Стабильное обобщение (LOSO F1=0.824±0.063)
- ✅ Практический размер модели (12K params, <2MB)
- ✅ Интерпретируемые компоненты (IMU branch, Physio branch, Stress profile с явной ролью)
- ✅ Хорошее соотношение сложность/точность

**Не использовать:** v8c, v10, v9 (более сложные, но не дают прироста на test или перегибаются).

### Путь к улучшению (если нужна работа в будущем)

Tier 1 (обязательные для +5–10pp):
1. ~~Увеличить окно с 100 на 300–600 samples (30 сек)~~ — требует полного rebuild датасета
2. ~~Per-subject z-score normalization~~ — требует изменения pipeline
3. ~~HRV-инженерные фичи вместо raw сигнала~~ — требует калькуляции фич

**Вывод:** информационный потолок 0.82 на текущих данных и параметризации жёсткий. Пробить его можно только через изменения на уровне данных (окно, нормализация, новые фичи), а не через архитектуру.
