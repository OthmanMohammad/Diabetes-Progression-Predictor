from sklearn import datasets
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def fetch_diabetes_data():
    return datasets.load_diabetes(return_X_y=True)

def save_data_to_csv(X, y):
    import pandas as pd

    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['target'] = y

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(os.path.join(DATA_DIR, 'diabetes.csv'), index=False)

if __name__ == "__main__":
    X, y = fetch_diabetes_data()
    save_data_to_csv(X, y)
