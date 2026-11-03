from sklearn.ensemble import RandomForestRegressor
import numpy as np
from .base_model import BaseBathymetryModel

class RandomForestBathymetry(BaseBathymetryModel):
    """Random Forest Regressor supporting max_depth control for depth sweeps."""

    def __init__(self, max_depth: int = None, n_estimators: int = 100, random_state: int = 42):
        super().__init__(name=f"RandomForest_depth_{max_depth}")
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)