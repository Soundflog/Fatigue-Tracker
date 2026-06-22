# ML Service

Отдельный проект ML-сервиса для AthleteFatigueTracker.

Сейчас сервис поднимает FastAPI API по контракту из ТЗ:

- `POST /ml/v1/predict`
- `POST /ml/v1/predict-batch`
- `GET /healthz`
- `GET /readyz`

gRPC-схема сохранена в `proto/fatigue_inference.proto` и готова для последующего подключения gRPC-сервера без изменения REST-контракта.

## Быстрый старт

```powershell
cd ml_service
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Настройки

Основные параметры лежат в `settings.yaml`.

Самое важное поле:

```yaml
model:
  path: ../res/results_v8_stress/results_test/best_model_v8_best.pth
```

При несовместимом checkpoint или отсутствии реального класса модели сервис остаётся работоспособным и отвечает через заглушку инференса. Это позволяет сначала стабилизировать API и wiring, а потом заменить `StubInferenceEngine` на реальный загрузчик.

## Ограничение текущей версии

В основном пакете репозитория нет явного production-класса для v8 dual-branch модели со stress-profile/FiLM. Поэтому runtime сейчас устроен как адаптер:

- читает и валидирует входной контракт;
- хранит путь к checkpoint в настройках;
- пытается проверить доступность файла модели;
- выдаёт предсказание через детерминированную заглушку, пока не будет подключён правильный runtime класса модели.

## Что менять дальше

Когда будет готов точный класс v8-модели, достаточно заменить реализацию в `app/model_runtime.py`:

1. загрузить checkpoint из `settings.yaml`;
2. применить нужную нормализацию;
3. вернуть реальный `fatigueDegree` вместо stub-значения.