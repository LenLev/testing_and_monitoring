from fastapi.testclient import TestClient

import ml_service.app as app_module


class DummyPipeline:
    feature_names_in_ = [
        'age',
        'workclass',
        'fnlwgt',
        'education',
        'education.num',
        'marital.status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'capital.gain',
        'capital.loss',
        'hours.per.week',
        'native.country',
    ]

    def predict_proba(self, _):
        return [[0.15, 0.85]]


class BrokenPipeline(DummyPipeline):
    def predict_proba(self, _):
        raise RuntimeError('boom')


def _set_model(model):
    with app_module.MODEL.lock:
        app_module.MODEL.data = app_module.MODEL.data._replace(model=model, run_id='test_run')


def test_predict_success(valid_payload):
    _set_model(DummyPipeline())
    client = TestClient(app_module.create_app())

    resp = client.post('/predict', json=valid_payload)

    assert resp.status_code == 200
    body = resp.json()
    assert body['prediction'] == 1
    assert 0.0 <= body['probability'] <= 1.0


def test_predict_returns_422_if_required_feature_missing(valid_payload):
    _set_model(DummyPipeline())
    client = TestClient(app_module.create_app())

    payload = {**valid_payload}
    del payload['age']

    resp = client.post('/predict', json=payload)

    assert resp.status_code == 422
    assert 'Missing required features' in resp.json()['detail']


def test_predict_returns_503_when_model_not_loaded(valid_payload):
    _set_model(None)
    client = TestClient(app_module.create_app())

    resp = client.post('/predict', json=valid_payload)

    assert resp.status_code == 503


def test_predict_returns_500_if_inference_fails(valid_payload):
    _set_model(BrokenPipeline())
    client = TestClient(app_module.create_app())

    resp = client.post('/predict', json=valid_payload)

    assert resp.status_code == 500
    assert resp.json()['detail'] == 'Inference failed'


def test_update_model_returns_400_for_invalid_run_id(monkeypatch):
    client = TestClient(app_module.create_app())

    def fake_set(*, run_id=None):
        raise RuntimeError('invalid run id')

    monkeypatch.setattr(app_module.MODEL, 'set', lambda run_id: (_ for _ in ()).throw(RuntimeError('invalid run id')))

    resp = client.post('/updateModel', json={'run_id': 'bad'})

    assert resp.status_code == 400
    assert 'Unable to load model' in resp.json()['detail']


def test_metrics_endpoint_exposes_graphite_backend_info():
    client = TestClient(app_module.create_app())

    resp = client.get('/metrics')

    assert resp.status_code == 200
    body = resp.json()
    assert body['backend'] == 'graphite'
