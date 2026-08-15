"""Scalers used for reconstruction."""

from __future__ import annotations

from typing import Any

import numpy as np
import warnings
from sklearn.base import TransformerMixin, RegressorMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, QuantileTransformer, PowerTransformer, FunctionTransformer


class LogScaler(TransformerMixin):
    """Apply a logarithmic transform with a specified base."""

    def __init__(self, base: float = np.e) -> None:
        """Initialise the scaler.

        Args:
            base: Logarithm base used for the transform.
        """
        self.base = base

    def fit(self, X: np.ndarray) -> LogScaler:
        """Fit the scaler.

        Args:
            X: Input data.

        Returns:
            The scaler instance.
        """
        print("No fitting required for LogScaler")
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data into log space.

        Args:
            X: Input data.

        Returns:
            Log-transformed data.
        """
        return np.emath.logn(self.base, X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data back from log space.

        Args:
            X: Log-space data.

        Returns:
            Data in the original space.
        """
        return np.emath.power(self.base, X)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit the scaler and transform the data.

        Args:
            X: Input data.

        Returns:
            Log-transformed data.
        """
        return self.transform(X) 


class MultiScaler():
    """Apply a sequence of scalers in order."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialise the multi-scaler.

        Args:
            **kwargs: Must contain a ``scalers`` iterable.
        """
        self.scalers = kwargs["scalers"]

    def fit(self, X: np.ndarray) -> MultiScaler:
        """Fit each scaler in sequence.

        Args:
            X: Input data.

        Returns:
            The scaler instance.
        """
        X = X.copy()
        for scaler in self.scalers:
            X = scaler.fit(X)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data through the scaler chain.

        Args:
            X: Input data.

        Returns:
            Transformed data.
        """
        X = X.copy()
        for scaler in self.scalers:
            X = scaler.transform(X)
        return X
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Invert the scaler chain.

        Args:
            X: Transformed data.

        Returns:
            Data mapped back to the original space.
        """
        X = X.copy()
        for scaler in reversed(self.scalers):
            X = scaler.inverse_transform(X)
        return X
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit each scaler and transform the data.

        Args:
            X: Input data.

        Returns:
            Transformed data.
        """
        X = X.copy()
        for scaler in self.scalers:
            X = scaler.fit_transform(X)
        return X
    

class Scaler(TransformerMixin):
    """Abstract base class for scalers."""

    available = {
        'standard': StandardScaler,
        'minmax': MinMaxScaler,
        'maxabs': MaxAbsScaler,
        'robust': RobustScaler,
        'quantile': QuantileTransformer,
        'power': PowerTransformer,
        'log': LogScaler,
        'function': FunctionTransformer,
        'multi': MultiScaler
    }
    def __init__(self) -> None:
        """Warn on direct instantiation."""
        warnings.warn("This is an abstract class and should not be instantiated directly.", UserWarning)
        
    def fit(self, X: np.ndarray) -> Scaler:
        """Fit the scaler.

        Args:
            X: Input data.

        Returns:
            The scaler instance.

        Raises:
            NotImplementedError: Always raised by the abstract base class.
        """
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the data.

        Args:
            X: Input data.

        Raises:
            NotImplementedError: Always raised by the abstract base class.
        """
        raise NotImplementedError
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Inverse transform the data.

        Args:
            X: Transformed data.

        Raises:
            NotImplementedError: Always raised by the abstract base class.
        """
        raise NotImplementedError


def get_scaler(scaler_name: str, **kwargs: Any) -> Any:
    """Construct a supported scaler instance.

    Args:
        scaler_name: Name of the scaler to construct.
        **kwargs: Keyword arguments forwarded to the scaler constructor.

    Returns:
        A scaler instance.

    Raises:
        ValueError: If the scaler name is not supported.
    """
    if scaler_name not in Scaler.available:
        raise ValueError(f"Scaler {scaler_name} not supported")
    return Scaler.available[scaler_name](**kwargs)
