from ml_service.model import Model


class DummyPipeline:
    def __init__(self):
        self.feature_names_in_ = ['age', 'hours.per.week']

    def predict_proba(self, _):
        return [[0.1, 0.9]]


def test_model_set_loads_model(monkeypatch):
    model = Model()

    def fake_load_model(*, run_id=None, model_uri=None):
        assert run_id == 'run_1'
        assert model_uri is None
        return DummyPipeline()

    monkeypatch.setattr('ml_service.model.load_model', fake_load_model)

    model.set('run_1')

    current = model.get()
    assert current.run_id == 'run_1'
    assert current.model is not None
    assert model.features == ['age', 'hours.per.week']


def test_model_features_empty_when_not_loaded():
    model = Model()
    assert model.features == []
