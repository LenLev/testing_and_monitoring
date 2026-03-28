# MLflow + FastAPI service

Сервис на FastAPI, который:
- при старте приложения загружает ML‑модель из MLflow
- имеет хэндлер `POST /predict` &mdash; принимает на вход признаки
- имеет хэндлер `POST /updateModel`, который принимает `run_id` и подменяет текущую модель на модель из этого run
- отправляет метрики в Graphite через StatsD
- имеет хэндлер `GET /metrics` с информацией о текущем бэкенде метрик
- поддерживает фоновую отправку drift-отчетов в Evidently (при наличии конфигурации)

## Переменные окружения

- **`MLFLOW_TRACKING_URI`**: URI вашего MLflow Tracking Server (например, `http://158.160.2.37:5000/`)
- **`DEFAULT_RUN_ID`**: то модель загрузится из запуска с этим ID
- **`STATSD_HOST`**: хост StatsD (по умолчанию `graphite`)
- **`STATSD_PORT`**: порт StatsD (по умолчанию `8125`)
- **`STATSD_PREFIX`**: префикс метрик в Graphite (по умолчанию `ml_service`)
- **`EVIDENTLY_URL`**: URL Evidently UI/Workspace (опционально)
- **`EVIDENTLY_PROJECT_ID`**: ID проекта в Evidently (опционально)
- **`DRIFT_INTERVAL_SECONDS`**: период отправки drift-репортов, сек (опционально, по умолчанию `300`)
- **`DRIFT_MIN_SAMPLES`**: минимальный размер накопленного чанка для drift-репорта (опционально, по умолчанию `100`)

## Запуск

Через docker compose:

```bash
export MLFLOW_TRACKING_URI=http://158.160.2.37:5000/
export DEFAULT_RUN_ID=<your_run_id>
docker compose up --build
```

Сервис будет доступен на `http://<ip>:1488`.

Graphite UI будет доступен на `http://<ip>:8080`.

### Эндпоинты

- `GET /health` - состояние сервиса и статус загрузки модели
- `POST /predict` - предсказание
- `POST /updateModel` - обновление модели по `run_id`
- `GET /metrics` - информация о конфигурации Graphite/StatsD

