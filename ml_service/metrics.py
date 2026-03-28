import time
from typing import Mapping

from fastapi import Request
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency at runtime
    psutil = None


REQUESTS_TOTAL = Counter(
    'service_requests_total',
    'Total number of incoming HTTP requests',
    ['method', 'path', 'status_code'],
)
REQUEST_LATENCY_SECONDS = Histogram(
    'service_request_latency_seconds',
    'End-to-end request latency in seconds',
    ['method', 'path'],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0),
)
ERRORS_TOTAL = Counter(
    'service_errors_total',
    'Total number of service errors by status code',
    ['path', 'status_code'],
)

PREPROCESSING_SECONDS = Histogram(
    'service_preprocessing_seconds',
    'Data preprocessing time in seconds',
    buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1),
)
INFERENCE_SECONDS = Histogram(
    'service_inference_seconds',
    'Model inference time in seconds',
    buckets=(0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2),
)
MODEL_PROBABILITY = Histogram(
    'model_probability',
    'Predicted probability for positive class',
    buckets=(0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0),
)
MODEL_PREDICTIONS_TOTAL = Counter(
    'model_predictions_total',
    'Predicted classes distribution',
    ['prediction'],
)

FEATURE_NUMERIC = Histogram(
    'feature_numeric_value',
    'Distribution of numeric feature values',
    ['feature'],
    buckets=(
        -1_000_000,
        -100_000,
        -10_000,
        -1000,
        -100,
        -10,
        0,
        1,
        10,
        100,
        1000,
        10_000,
        100_000,
        1_000_000,
    ),
)
FEATURE_CATEGORICAL_TOTAL = Counter(
    'feature_categorical_total',
    'Distribution of categorical feature values',
    ['feature', 'value'],
)

PROCESS_MEMORY_BYTES = Gauge(
    'service_process_memory_bytes',
    'Resident memory size in bytes',
)
PROCESS_CPU_PERCENT = Gauge(
    'service_process_cpu_percent',
    'CPU usage percent of the process',
)

MODEL_UPDATES_TOTAL = Counter(
    'model_updates_total',
    'Number of model update attempts',
    ['status'],
)
MODEL_INFO = Info(
    'model_info',
    'Active model information',
)
MODEL_REQUIRED_FEATURES = Info(
    'model_required_features',
    'Required features for currently active model',
)


def metrics_response() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST


def observe_feature_values(feature_values: Mapping[str, object]) -> None:
    for feature, value in feature_values.items():
        if value is None:
            continue

        if isinstance(value, (int, float)):
            FEATURE_NUMERIC.labels(feature=feature).observe(float(value))
        else:
            FEATURE_CATEGORICAL_TOTAL.labels(feature=feature, value=str(value)).inc()


def observe_prediction(probability: float, prediction: int) -> None:
    MODEL_PROBABILITY.observe(probability)
    MODEL_PREDICTIONS_TOTAL.labels(prediction=str(prediction)).inc()


def observe_model_update(
    run_id: str,
    status: str,
    features: list[str] | None = None,
    model_type: str | None = None,
) -> None:
    MODEL_UPDATES_TOTAL.labels(status=status).inc()

    if status == 'success':
        MODEL_INFO.info({'run_id': run_id, 'model_type': model_type or 'unknown'})
        MODEL_REQUIRED_FEATURES.info({'features': ','.join(features or [])})


def refresh_resource_metrics() -> None:
    if psutil is None:
        return

    process = psutil.Process()
    PROCESS_MEMORY_BYTES.set(process.memory_info().rss)
    PROCESS_CPU_PERCENT.set(process.cpu_percent(interval=None))


async def track_http_metrics(request: Request, call_next):
    started = time.perf_counter()
    path = request.url.path
    method = request.method

    try:
        response = await call_next(request)
    except Exception:
        elapsed = time.perf_counter() - started
        REQUEST_LATENCY_SECONDS.labels(method=method, path=path).observe(elapsed)
        REQUESTS_TOTAL.labels(method=method, path=path, status_code='500').inc()
        ERRORS_TOTAL.labels(path=path, status_code='500').inc()
        refresh_resource_metrics()
        raise

    elapsed = time.perf_counter() - started
    status_code = str(response.status_code)
    REQUEST_LATENCY_SECONDS.labels(method=method, path=path).observe(elapsed)
    REQUESTS_TOTAL.labels(method=method, path=path, status_code=status_code).inc()

    if response.status_code >= 400:
        ERRORS_TOTAL.labels(path=path, status_code=status_code).inc()

    refresh_resource_metrics()
    return response
