import importlib
import logging
import os
import re
import time
from typing import Any, Mapping

from fastapi import Request

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency at runtime
    psutil = None

LOGGER = logging.getLogger(__name__)


def _sanitize(part: str) -> str:
    normalized = re.sub(r'[^a-zA-Z0-9_]+', '_', str(part))
    return normalized.strip('_').lower() or 'unknown'


_STATSD_HOST = os.getenv('STATSD_HOST', 'graphite')
_STATSD_PORT = int(os.getenv('STATSD_PORT', '8125'))
_STATSD_PREFIX = os.getenv('STATSD_PREFIX', 'ml_service')

try:
    statsd_module = importlib.import_module('statsd')
    StatsClient = getattr(statsd_module, 'StatsClient', None)
except ImportError:  # pragma: no cover - optional dependency at runtime
    StatsClient = None


class _NoopStatsClient:
    def incr(self, _metric: str, count: int = 1) -> None:
        _ = count

    def gauge(self, _metric: str, value: float) -> None:
        _ = value

    def timing(self, _metric: str, value: int) -> None:
        _ = value

    def set(self, _metric: str, value: str) -> None:
        _ = value


if StatsClient is None:
    LOGGER.warning('statsd package is not installed, metric emission is disabled')
    _CLIENT: Any = _NoopStatsClient()
else:
    _CLIENT = StatsClient(host=_STATSD_HOST, port=_STATSD_PORT, prefix=_STATSD_PREFIX)


def metrics_backend_info() -> dict[str, str | int]:
    return {
        'backend': 'graphite',
        'statsd_host': _STATSD_HOST,
        'statsd_port': _STATSD_PORT,
        'statsd_prefix': _STATSD_PREFIX,
    }


def _incr(metric: str, count: int = 1) -> None:
    try:
        _CLIENT.incr(metric, count=count)
    except Exception:
        LOGGER.exception('Failed to send counter metric: %s', metric)


def _gauge(metric: str, value: float) -> None:
    try:
        _CLIENT.gauge(metric, value)
    except Exception:
        LOGGER.exception('Failed to send gauge metric: %s', metric)


def _timing_ms(metric: str, seconds: float) -> None:
    milliseconds = max(int(seconds * 1000), 0)
    try:
        _CLIENT.timing(metric, milliseconds)
    except Exception:
        LOGGER.exception('Failed to send timing metric: %s', metric)


def _set(metric: str, value: str) -> None:
    try:
        _CLIENT.set(metric, value)
    except Exception:
        LOGGER.exception('Failed to send set metric: %s', metric)


def observe_feature_values(feature_values: Mapping[str, object]) -> None:
    for feature, value in feature_values.items():
        if value is None:
            continue

        feature_name = _sanitize(feature)

        if isinstance(value, (int, float)):
            _gauge(f'features.numeric.{feature_name}.value', float(value))
        else:
            category = _sanitize(str(value))
            _incr(f'features.categorical.{feature_name}.{category}.total')


def observe_preprocessing_duration(seconds: float) -> None:
    _timing_ms('data.preprocessing_ms', seconds)


def observe_inference_duration(seconds: float) -> None:
    _timing_ms('model.inference_ms', seconds)


def observe_prediction(probability: float, prediction: int) -> None:
    _gauge('model.probability', probability)
    _incr(f'model.predictions.{_sanitize(str(prediction))}.total')


def observe_model_update(
    run_id: str,
    status: str,
    features: list[str] | None = None,
    model_type: str | None = None,
) -> None:
    _incr(f'model.update.{_sanitize(status)}.total')

    if status == 'success':
        _set('model.active.run_id', run_id)
        _set('model.active.type', model_type or 'unknown')
        _gauge('model.active.required_features.count', float(len(features or [])))
        for feature in features or []:
            _set('model.active.required_feature', feature)


def refresh_resource_metrics() -> None:
    if psutil is None:
        return

    try:
        process = psutil.Process()
        _gauge('system.process.memory_bytes', float(process.memory_info().rss))
        _gauge('system.process.cpu_percent', float(process.cpu_percent(interval=None)))
    except Exception:
        LOGGER.exception('Failed to collect process resource metrics')


async def track_http_metrics(request: Request, call_next):
    started = time.perf_counter()
    path = _sanitize(request.url.path)
    method = _sanitize(request.method)

    try:
        response = await call_next(request)
    except Exception:
        elapsed = time.perf_counter() - started
        _timing_ms(f'http.response_time_ms.{method}.{path}', elapsed)
        _incr(f'http.requests_total.{method}.{path}.500')
        _incr(f'http.errors_total.{path}.500')
        refresh_resource_metrics()
        raise

    elapsed = time.perf_counter() - started
    status_code = _sanitize(str(response.status_code))
    _timing_ms(f'http.response_time_ms.{method}.{path}', elapsed)
    _incr(f'http.requests_total.{method}.{path}.{status_code}')

    if response.status_code >= 400:
        _incr(f'http.errors_total.{path}.{status_code}')

    refresh_resource_metrics()
    return response
