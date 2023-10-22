from src.data import fetch_diabetes_data, save_data_to_csv
import os
import pandas as pd

def test_fetch_diabetes_data():
    X, y = fetch_diabetes_data()
    assert X is not None
    assert y is not None
    assert len(X) == len(y)

def test_save_data_to_csv():
    X, y = fetch_diabetes_data()
    save_data_to_csv(X, y)
    assert os.path.exists(os.path.join('data', 'diabetes.csv'))
    df = pd.read_csv(os.path.join('data', 'diabetes.csv'))
    assert 'target' in df.columns
