# Отчет по доработке сервиса

## 1. Что доработано по надежности и отказоустойчивости

Добавлена обработка следующих ситуаций:

1. `MLFLOW_TRACKING_URI` не задан или недоступен на старте:
- сервис не падает;
- пишет ошибку в лог;
- продолжает работу в деградированном режиме.

2. `DEFAULT_RUN_ID` не задан на старте:
- сервис не падает;
- модель не загружается;
- `/health` показывает `model_loaded=false`.

3. Модель не смогла загрузиться на старте (битый/несуществующий run):
- сервис не падает;
- ошибка логируется;
- фиксируется неуспешное обновление модели в метриках.

4. Запрос на `/predict`, когда модель не загружена:
- возвращается `503` и понятный `detail`.

5. Невалидный вход (типовые ошибки схемы Pydantic):
- возвращается `422` от FastAPI/Pydantic.

6. Переданы не все фичи, требуемые моделью:
- возвращается `422` с сообщением `Missing required features`.

7. Модель требует неизвестные сервису фичи:
- возвращается `422` с сообщением `Unsupported features required by model`.

8. Ошибка во время предобработки:
- возвращается `500` с `Preprocessing failed`;
- ошибка логируется.

9. Ошибка инференса (`predict_proba` упал):
- возвращается `500` с `Inference failed`;
- ошибка логируется.

10. Невалидный `run_id` в `/updateModel` (пустой):
- отклоняется валидацией запроса (422).

11. Некорректный/несуществующий `run_id` в `/updateModel`:
- возвращается `400` с `Unable to load model for run_id=...`;
- ошибка логируется;
- событие неуспешного обновления отражается в метриках.

## 2. Добавленные тесты

Добавлены следующие тесты:

1. Предобработка (`tests/test_features.py`):
- успешная сборка `DataFrame` с нужным набором колонок;
- ошибка при пропущенной обязательной фиче;
- ошибка при неизвестной фиче.

2. Загрузка модели и инференс (`tests/test_model.py`):
- успешная загрузка модели через `Model.set`;
- проверка `run_id` и списка фич;
- поведение `features`, когда модель еще не загружена.

3. Хэндлеры и валидация (`tests/test_app_handlers.py`):
- успешный `/predict`;
- `503`, если модель не загружена;
- `422`, если отсутствует обязательная фича;
- `500`, если инференс падает;
- `400` для невалидного `run_id` в обновлении модели;
- проверка, что `/metrics` возвращает конфигурацию Graphite backend.

4. E2E (`tests/test_service_e2e.py`):
- сервис поднимается через `TestClient`;
- `/health` доступен;
- `/predict` возвращает предсказание.

## 3. Логирование метрик (Graphite)

Метрики отправляются в Graphite через StatsD.

Хэндлер `GET /metrics` возвращает служебную информацию о backend и параметрах подключения StatsD.

Логируются:

1. Технические метрики сервиса:
- `ml_service.http.requests_total.<method>.<path>.<status_code>` (число запросов);
- `ml_service.http.response_time_ms.<method>.<path>` (время ответа);
- `ml_service.http.errors_total.<path>.<status_code>` (статистика ошибок);
- `ml_service.system.process.memory_bytes` (RSS память);
- `ml_service.system.process.cpu_percent` (CPU процесса).

2. Метрики входных данных:
- `ml_service.data.preprocessing_ms` (время предобработки);
- `ml_service.features.numeric.<feature>.value` (значения числовых фич);
- `ml_service.features.categorical.<feature>.<value>.total` (распределение категориальных фич).

3. Метрики модели:
- `ml_service.model.inference_ms` (время инференса);
- `ml_service.model.probability` (распределение вероятностей);
- `ml_service.model.predictions.<class>.total` (распределение классов предсказаний).

4. События и состояние прод-модели:
- `ml_service.model.update.<status>.total` (успешные/неуспешные обновления);
- `ml_service.model.active.type` (тип активной модели, StatsD set);
- `ml_service.model.active.run_id` (run_id активной модели, StatsD set);
- `ml_service.model.active.required_feature` и `ml_service.model.active.required_features.count` (фичи активной модели).

Перцентили 75%, 90%, 95%, 99%, 99.9% считаются StatsD/Graphite для таймеров (`response_time_ms`, `preprocessing_ms`, `inference_ms`). Для этого в `statsd/config.js` задан `percentThreshold: [75, 90, 95, 99, 99.9]`.

## 4. Мониторинг дрифта (Evidently)

Добавлен буфер и фоновая корутина:

1. На каждом `predict` сохраняются входные фичи + `prediction` + `probability`.
2. Корутинa (`asyncio.ensure_future`) периодически проверяет накопленные данные.
3. При достижении порога строится `DataDriftPreset` отчет и отправляется в Evidently Workspace.
4. Опорный (reference) набор инициализируется первым накопленным чанком.

Параметры через переменные окружения:
- `EVIDENTLY_URL`
- `EVIDENTLY_PROJECT_ID`
- `DRIFT_INTERVAL_SECONDS` (по умолчанию 300)
- `DRIFT_MIN_SAMPLES` (по умолчанию 100)

## 5. Что нужно доделать вам вручную

1. Получить доступ к Grafana:
- в Stepik указать ваш логин ВМ и получить пароль.

2. В Grafana создать дашборд для всех метрик:
- запросы/ошибки/latency;
- ресурсы CPU/RAM;
- preprocessing/inference latency;
- распределения фичей, вероятностей и предсказаний;
- события обновления модели и активная модель.

3. Настроить алерты (минимум 5):
- на latency сервиса;
- на latency инференса;
- на 5xx;
- на потребление ресурсов;
- на drift фичей/предсказаний/вероятностей.

4. Настроить contact point и канал доставки алертов:
- например, Telegram contact point в Grafana;
- дать проверяющему доступ к месту получения алертов.

5. Для сдачи:
- приложить ссылку на PR;
- приложить ссылку на Grafana-дашборд(ы);
- если используете Telegram, приложить invite-ссылку на чат.

