# Открытые датасеты для перехода на гибридную модель CNN+LSTM

Дата актуализации: 2026-02-25.

## Что важно для CNN+LSTM в задаче переутомления
- `CNN` извлекает локальные паттерны из многоканальных рядов (IMU/PPG/EDA/HR).
- `LSTM` захватывает более длинную динамику (нарастание утомления, переходы фаз нагрузки).
- Для этого нужны датасеты с последовательностями (raw/длинные окна), а не только с отдельно вырезанными шагами.

## Таблица 1. Сравнение открытых датасетов

| Датасет | Домены/спорт | Сенсоры и модальности | Разметка, полезная для fatigue | Масштаб | Лицензия | Пригодность для CNN+LSTM |
|---|---|---|---|---|---|---|
| [Zenodo 7997851](https://zenodo.org/records/7997851) | Бег | 1 поясной IMU (Shimmer3), acc+gyro (и derived magnitude) | Прямая: `F/NF` (после/до fatiguing beep test), сегментировано по strides | 19 бегунов, 256 Hz, 8 CSV | CC BY 4.0 | Очень высокая (базовый supervised fatigue-корпус) |
| [Zenodo 11114096](https://zenodo.org/records/11114096) | Бег | 1 поясной IMU (Shimmer3), raw time series | Прямая: отдельные фазы `NF`, `BeepTest`, `F` | 19 бегунов, 256 Hz, raw CSV | CC BY 4.0 | Очень высокая (лучший источник длинной временной динамики для LSTM) |
| [4TU 14307743](https://doi.org/10.4121/14307743) | Бег | Мульти-IMU, MAT (`*_strides`, pre/post fatigue runs) | Прямая: протокол утомления до `RPE>16`, post-fatigue run | v1 содержит файлы `p001`-`p008`, ~6 GB unzipped | CC BY-NC-ND 4.0 | Высокая (хорошо для cross-dataset проверки обобщения) |
| [SWIFTIES 15172815](https://zenodo.org/records/15172815) | Fatiguing задачи верхней конечности | IMU + EMG + motion capture + pressure insole | Прямая fatigue-нагрузка по протоколу (25%/45% MVC, static/dynamic) | 32 участника, 23.6 GB | CC BY 4.0 | Средняя/высокая для предобучения multimodal-энкодера fatigue |
| [Zenodo 4266157](https://zenodo.org/records/4266157) | Longitudinal wearable monitoring | Мультимодальные wearable сигналы + PRO анкеты | Ежедневные self-reported fatigue PRO | 28 участников, 973 дней; matched 27/405, 1-minute resolution | CC BY 4.0 | Средняя (персонализация и регрессия fatigue score, не спорт-специфично) |
| [PhysioNet wearable-device-dataset 1.0.1](https://www.physionet.org/content/wearable-device-dataset/1.0.1/Wearable_Dataset/) | Stress + aerobic/anaerobic exercise | Empatica E4: BVP(64 Hz), ACC(32 Hz), TEMP(4 Hz), EDA(4 Hz) | Разметка по протоколам stress/aerobic/anaerobic + self-report stress | 36 stress, 30 aerobic, 31 anaerobic | Open Data Commons Attribution v1.0 | Средняя (полезно для предобучения физиологической ветки LSTM) |
| [GIT 10841412](https://zenodo.org/records/10841412) | Cycling/Running/Kayaking/Rowing | HR, VO2, lactate и производные индексы | Интенсивность и пороговые физиологические состояния (proxy fatigue) | 835 graded incremental tests (v2) | CC BY 4.0 | Средняя (сильный источник физиологических таргетов, но не IMU raw) |

## Таблица 2. Какие датасеты можно использовать вместе

| Комбинация | Можно объединять напрямую? | Зачем объединять | Что нужно сделать перед обучением |
|---|---|---|---|
| `Zenodo 7997851 + Zenodo 11114096` | Да, высокая совместимость | Единый беговой домен: labels из stride-версии + длинная динамика из raw | Общий маппинг каналов (`ax, ay, az, gx, gy, gz`), единая длина окна (например 5-20 c), subject-wise split |
| `Zenodo 7997851 + 4TU 14307743` | Да, но через гармонизацию | Повышение обобщающей способности между датчиками/протоколами | Ресемплинг, нормализация амплитуд, выравнивание label-схемы (`pre-fatigue`/`post-fatigue`) |
| `Zenodo 11114096 + 4TU 14307743` | Да, средняя | Тренировать LSTM на raw (Zenodo) и проверять перенос на другой датасет | В 11114096 сделать stride/window extraction, в 4TU унифицировать сегменты в те же окна |
| `Базовый беговой стек + SWIFTIES` | Частично (лучше через transfer learning) | Предобучение multimodal fatigue-энкодера и перенос в running задачу | Использовать SWIFTIES как pretraining (self-supervised/multitask), затем fine-tune на running fatigue |
| `Базовый беговой стек + PhysioNet wearable` | Частично | Улучшить физиологическую чувствительность модели (PPG/EDA/TEMP) | Двухветочная архитектура: IMU-ветка и physio-ветка, затем late fusion |
| `Базовый беговой стек + Zenodo 4266157` | Частично | Персонализация под индивидуальные fatigue-профили | Добавить personalization head (subject embedding), обучать на PRO как auxiliary target |
| `Базовый беговой стек + GIT 10841412` | Частично | Добавить физиологические прокси усталости (лактат/VO2/HR) | Multi-task: основная классификация fatigue + auxiliary regression физиологических индексов |

## Рекомендованный стартовый стек под ваш текущий проект

1. `Zenodo 11114096` + `Zenodo 7997851` + `4TU 14307743`.
2. Базовая цель: `binary fatigue (pre/post)` + вспомогательная цель `fatigue stage` (для участков BeepTest из 11114096).
3. Дальше расширение:
   - добавить `PhysioNet wearable-device-dataset` для предобучения физиологической LSTM-ветки;
   - добавить `SWIFTIES` для transfer learning по fatigue-динамике.

## Источники
- https://zenodo.org/records/7997851
- https://zenodo.org/records/11114096
- https://doi.org/10.4121/14307743
- https://zenodo.org/records/15172815
- https://zenodo.org/records/4266157
- https://www.physionet.org/content/wearable-device-dataset/1.0.1/Wearable_Dataset/
- https://zenodo.org/records/10841412
