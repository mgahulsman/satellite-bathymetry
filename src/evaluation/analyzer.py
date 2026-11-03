from typing import Dict
from .metrics import compute_metrics

class OverfittingAnalyzer:
    """Evaluates train vs. validation gaps to detect overfitting without physically moving code."""

    def __init__(self, rmse_gap_threshold: float = 0.15):
        self.threshold = rmse_gap_threshold

    def evaluate_run(
        self,
        model_name: str,
        y_train_true,
        y_train_pred,
        y_val_true,
        y_val_pred
    ) -> Dict:
        train_metrics = compute_metrics(y_train_true, y_train_pred)
        val_metrics = compute_metrics(y_val_true, y_val_pred)

        rmse_gap = val_metrics["RMSE"] - train_metrics["RMSE"]
        is_overfitting = rmse_gap > self.threshold

        return {
            "model_name": model_name,
            "train_rmse": train_metrics["RMSE"],
            "val_rmse": val_metrics["RMSE"],
            "train_r2": train_metrics["R2"],
            "val_r2": val_metrics["R2"],
            "rmse_gap": round(float(rmse_gap), 4),
            "status": "OVERFITTING" if is_overfitting else "NORMAL"
        }