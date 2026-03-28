import pytest


@pytest.fixture
def valid_payload() -> dict:
    return {
        'age': 39,
        'workclass': 'Private',
        'fnlwgt': 77516,
        'education': 'Bachelors',
        'education.num': 13,
        'marital.status': 'Never-married',
        'occupation': 'Adm-clerical',
        'relationship': 'Not-in-family',
        'race': 'White',
        'sex': 'Male',
        'capital.gain': 2174,
        'capital.loss': 0,
        'hours.per.week': 40,
        'native.country': 'United-States',
    }
