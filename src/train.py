import json
import mlflow
import mlflow.keras
import mlflow.tensorflow
from model import DiabetesPredictor
from data import fetch_diabetes_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load configuration
with open("config.json", "r") as file:
    config = json.load(file)

test_size = config["train_test_split"]["test_size"]
random_state = config["train_test_split"]["random_state"]
epochs = config["training"]["epochs"]

# Set experiment name
experiment_name = "Diabetes_Prediction_v1"
mlflow.set_experiment(experiment_name)


def train_and_evaluate_model():
    mlflow.tensorflow.autolog()

    X, y = fetch_diabetes_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model_instance = DiabetesPredictor()

    with mlflow.start_run():
        history = model_instance.model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))
        
        # Log parameters and metrics
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("optimizer", "adam")
        mlflow.log_metric("loss", history.history["loss"][-1])
        mlflow.log_metric("mae", history.history["mae"][-1])
        
        evaluate_model(model_instance.model, X_test, y_test)
        
        # Log model
        mlflow.keras.log_model(model_instance.model, "model")


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)

    print(f"MSE: {mse}, R2: {r2}")


if __name__ == "__main__":
    train_and_evaluate_model()
