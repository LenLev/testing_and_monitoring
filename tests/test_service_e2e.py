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
        return [[0.4, 0.6]]


def test_service_start_and_predict(valid_payload):
    with app_module.MODEL.lock:
        app_module.MODEL.data = app_module.MODEL.data._replace(model=DummyPipeline(), run_id='e2e_run')

    client = TestClient(app_module.create_app())

    health = client.get('/health')
    prediction = client.post('/predict', json=valid_payload)

    assert health.status_code == 200
    assert health.json()['model_loaded'] is True

    assert prediction.status_code == 200
    assert prediction.json()['prediction'] in [0, 1]
