import pandas as pd

from ml_service.schemas import PredictRequest


FEATURE_COLUMNS = [
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


def _to_attr_name(column: str) -> str:
    return column.replace('.', '_')


def request_to_feature_dict(req: PredictRequest) -> dict[str, object]:
    values: dict[str, object] = {}
    for column in FEATURE_COLUMNS:
        values[column] = getattr(req, _to_attr_name(column))
    return values


def to_dataframe(req: PredictRequest, needed_columns: list[str] = None) -> pd.DataFrame:
    if needed_columns is None:
        columns = FEATURE_COLUMNS
    else:
        unknown = [column for column in needed_columns if column not in FEATURE_COLUMNS]
        if unknown:
            raise ValueError(f'Unsupported features required by model: {unknown}')
        columns = needed_columns

    feature_values = request_to_feature_dict(req)
    missing = [column for column in columns if feature_values.get(column) is None]
    if missing:
        raise ValueError(f'Missing required features: {missing}')

    row = [feature_values[column] for column in columns]
    return pd.DataFrame([row], columns=columns)
