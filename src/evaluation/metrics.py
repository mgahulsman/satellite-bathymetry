import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Computes standard evaluation metrics for depth predictions."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        "RMSE": round(float(rmse), 4),
        "MAE": round(float(mae), 4),
        "R2": round(float(r2), 4)
    }