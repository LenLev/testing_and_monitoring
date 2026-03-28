import pytest

from ml_service.features import to_dataframe
from ml_service.schemas import PredictRequest


def test_to_dataframe_success_with_needed_columns(valid_payload):
    req = PredictRequest(**valid_payload)
    needed_columns = ['age', 'workclass', 'hours.per.week']

    df = to_dataframe(req, needed_columns=needed_columns)

    assert list(df.columns) == needed_columns
    assert int(df.iloc[0]['age']) == valid_payload['age']


def test_to_dataframe_raises_for_missing_required_feature(valid_payload):
    payload = {**valid_payload, 'hours.per.week': None}
    req = PredictRequest(**payload)

    with pytest.raises(ValueError, match='Missing required features'):
        to_dataframe(req, needed_columns=['hours.per.week'])


def test_to_dataframe_raises_for_unknown_feature(valid_payload):
    req = PredictRequest(**valid_payload)

    with pytest.raises(ValueError, match='Unsupported features required by model'):
        to_dataframe(req, needed_columns=['unknown_feature'])
