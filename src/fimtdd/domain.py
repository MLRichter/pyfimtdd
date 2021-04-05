from abc import ABC, abstractmethod
from typing import Protocol, Optional, Callable
from attr import attrs, attrib
import numpy as np


class Predictor(ABC):

    def fit(self, X: np.ndarray, y: np.ndarray, is_alt: bool) -> "Predictor":
        self(X, y, is_alt)
        return self

    @abstractmethod
    def predict(self, X: np.ndarray, is_alt: bool) -> np.ndarray:
        ...

    def fit_predict(self, X: np.ndarray, y: np.ndarray, is_alt: bool) -> np.ndarray:
        return self(X, y, is_alt)

    @abstractmethod
    def __call__(self, X: np.ndarray, y: np.ndarray, is_alt: bool) -> np.ndarray:
        ...

    def offspring(self) -> "Predictor":
        ...


class DriftDetector(Protocol):

    def update(self, y_true, y_pred) -> None:
        ...

    def detect_drift(self) -> bool:
        ...

    def copy(self) -> "DriftDetector":
        ...


class NodeSplitter(Protocol):

    def update(self, X: np.ndarray, y: np.ndarray, is_alt: bool) -> bool:
        ...

    def split(self, node: Predictor, predictor: Predictor, is_alt: bool) -> Predictor:
        ...

    def copy(self) -> "NodeSplitter":
        ...


class SwappingCriterion(Protocol):
    def update(self, y: np.ndarray, y_pred: np.ndarray, alt_y_pred: np.ndarray) -> bool:
        ...

    def copy(self) -> "SwappingCriterion":
        ...
