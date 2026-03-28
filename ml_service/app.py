import asyncio
import logging
import time
from typing import Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

from ml_service import config
from ml_service.drift import run_drift_reporter, track_for_drift
from ml_service.features import request_to_feature_dict, to_dataframe
from ml_service.metrics import (
    metrics_backend_info,
    observe_inference_duration,
    observe_feature_values,
    observe_model_update,
    observe_preprocessing_duration,
    observe_prediction,
    refresh_resource_metrics,
    track_http_metrics,
)
from ml_service.mlflow_utils import configure_mlflow
from ml_service.model import Model
from ml_service.schemas import (
    PredictRequest,
    PredictResponse,
    UpdateModelRequest,
    UpdateModelResponse,
)


MODEL = Model()
LOGGER = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.

    Loads the initial model from MLflow on startup.
    """
    try:
        configure_mlflow()
    except Exception:
        LOGGER.exception('Failed to configure MLflow tracking URI, service keeps running')

    run_id: str | None = None
    try:
        run_id = config.default_run_id()
        MODEL.set(run_id=run_id)
        observe_model_update(
            run_id=run_id,
            status='success',
            features=MODEL.features,
            model_type=type(MODEL.get().model).__name__,
        )
        LOGGER.info('Model loaded at startup, run_id=%s', run_id)
    except Exception:
        LOGGER.exception('Failed to load startup model, service will stay alive in degraded mode')
        observe_model_update(run_id=run_id or 'undefined', status='failed')

    drift_task = asyncio.ensure_future(run_drift_reporter())
    yield
    drift_task.cancel()
    # add any teardown logic here if needed


def create_app() -> FastAPI:
    app = FastAPI(title='MLflow FastAPI service', version='1.0.0', lifespan=lifespan)
    app.middleware('http')(track_http_metrics)

    @app.get('/health')
    def health() -> dict[str, Any]:
        model_state = MODEL.get()
        run_id = model_state.run_id
        model_loaded = model_state.model is not None
        refresh_resource_metrics()
        return {
            'status': 'ok',
            'run_id': run_id,
            'model_loaded': model_loaded,
            'metrics_backend': 'graphite',
        }

    @app.get('/metrics')
    def metrics() -> dict[str, Any]:
        return {
            'status': 'ok',
            'message': 'Graphite backend is enabled. Metrics are emitted via StatsD.',
            **metrics_backend_info(),
        }

    @app.post('/predict', response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        model = MODEL.get().model
        if model is None:
            raise HTTPException(status_code=503, detail='Model is not loaded yet')

        feature_values = request_to_feature_dict(request)
        observe_feature_values(feature_values)

        preprocess_started = time.perf_counter()
        try:
            df = to_dataframe(request, needed_columns=MODEL.features)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            LOGGER.exception('Unexpected preprocessing error')
            raise HTTPException(status_code=500, detail='Preprocessing failed') from exc
        finally:
            observe_preprocessing_duration(time.perf_counter() - preprocess_started)

        infer_started = time.perf_counter()
        try:
            probability = float(model.predict_proba(df)[0][1])
        except Exception as exc:
            LOGGER.exception('Inference failed')
            raise HTTPException(status_code=500, detail='Inference failed') from exc
        finally:
            observe_inference_duration(time.perf_counter() - infer_started)

        prediction = int(probability >= 0.5)
        observe_prediction(probability=probability, prediction=prediction)
        track_for_drift(feature_values=feature_values, prediction=prediction, probability=probability)

        return PredictResponse(prediction=prediction, probability=probability)

    @app.post('/updateModel', response_model=UpdateModelResponse)
    def update_model(req: UpdateModelRequest) -> UpdateModelResponse:
        run_id = req.run_id
        try:
            MODEL.set(run_id=run_id)
            observe_model_update(
                run_id=run_id,
                status='success',
                features=MODEL.features,
                model_type=type(MODEL.get().model).__name__,
            )
        except Exception as exc:
            LOGGER.exception('Model update failed for run_id=%s', run_id)
            observe_model_update(run_id=run_id, status='failed')
            raise HTTPException(status_code=400, detail=f'Unable to load model for run_id={run_id}') from exc
        return UpdateModelResponse(run_id=run_id)

    return app


app = create_app()
