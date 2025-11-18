import pandas as pd
import numpy as np
from .base_model import BasePredictor


class SimplePredictor(BasePredictor):
    """Simple baseline predictor. Always predicts the home team will win."""

    def __init__(self):
        super().__init__(name="Simple")

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        print(X)
        return np.full(len(X), 1)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), 1)