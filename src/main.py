import numpy as np
import pandas as pd
from models.random_forest import RandomForestBathymetry
from evaluation.analyzer import OverfittingAnalyzer

def generate_synthetic_bathymetry_data(n_samples=500, random_state=42):
    """Generates dummy satellite band reflectances and depth ground truth."""
    np.random.seed(random_state)
    # 3 band reflectances (e.g., Blue, Green, Red)
    X = np.random.uniform(0.1, 0.8, size=(n_samples, 3))
    # Synthetic depth target with non-linear relationship and noise
    y = (X[:, 0] * 12.0) - (X[:, 1] * 8.0) + (X[:, 2] ** 2 * 5.0) + np.random.normal(0, 0.5, size=n_samples)
    
    # Train / Val Split
    split_idx = int(n_samples * 0.8)
    return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]

def main():
    print("--- Satellite Bathymetry Experiment Runner ---\n")

    # 1. Load Data
    X_train, y_train, X_val, y_val = generate_synthetic_bathymetry_data()
    print(f"Dataset Loaded: Train shape={X_train.shape}, Val shape={X_val.shape}\n")

    # 2. Setup Evaluation Analyzer
    analyzer = OverfittingAnalyzer(rmse_gap_threshold=0.10)
    results = []

    # 3. Parameter Sweep over max_depth
    depth_configurations = [2, 5, 10, None]

    for depth in depth_configurations:
        # Initialize and Train Model
        model = RandomForestBathymetry(max_depth=depth, n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        # Generate Predictions
        train_preds = model.predict(X_train)
        val_preds = model.predict(X_val)

        # Analyze Performance & Fit
        run_summary = analyzer.evaluate_run(
            model_name=model.name,
            y_train_true=y_train,
            y_train_pred=train_preds,
            y_val_true=y_val,
            y_val_pred=val_preds
        )
        results.append(run_summary)

    # 4. Display Results
    results_df = pd.DataFrame(results)
    print("=== Experiment Summary ===")
    print(results_df.to_string(index=False))

if __name__ == "__main__":
    main()