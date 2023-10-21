import matplotlib.pyplot as plt

def plot_loss(history, save_path="loss_plot.png"):
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_predictions_vs_actual(y_true, y_pred, save_path="predictions_vs_actual.png"):
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.title('Predictions vs Actual')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red')
    plt.savefig(save_path)
    plt.close()

def plot_residuals(y_true, y_pred, save_path="residual_plot.png"):
    plt.figure()
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.title('Residuals vs Predicted')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.axhline(y=0, color='red', linestyle='-')
    plt.savefig(save_path)
    plt.close()
