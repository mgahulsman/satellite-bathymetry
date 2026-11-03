from abc import ABC, abstractmethod
import numpy as np

class BaseBathymetryModel(ABC):
    """Abstract base class for bathymetry prediction models."""

    def __init__(self, name: str):
        self.name = name
        self.model = None

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model on input features and bathymetry targets."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict water depths/bathymetry for given features."""
        pass