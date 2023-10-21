import mlflow
import mlflow.tensorflow
from model import DiabetesPredictor
from data import fetch_diabetes_data
from sklearn.model_selection import train_test_split

def train_model():
    mlflow.tensorflow.autolog()

    X, y = fetch_diabetes_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_instance = DiabetesPredictor()

    with mlflow.start_run():
        history = model_instance.model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))
        
        # Optionally: Log parameters, metrics, or artifacts.
        # mlflow.log_param("param_name", "param_value")
        # mlflow.log_metric("metric_name", metric_value)
        # mlflow.log_artifact("path_to_artifact")

if __name__ == "__main__":
    train_model()
