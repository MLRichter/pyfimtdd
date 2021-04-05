from copy import deepcopy
from typing import Dict, Callable, Optional

import numpy as np
from sklearn.linear_model import SGDRegressor

from .domain import Predictor
from attr import attrs, attrib

from .nodes import Node
import padasip as pa


@attrs(auto_attribs=True, slots=True, frozen=True)
class BinaryIntermediatePredictor(Predictor):

    left: Node
    right: Node
    binary_condition: Callable[[np.ndarray, Dict[bool, Node]], Node]
    _mapping: Dict[bool, Node]

    def predict(self, X: np.ndarray, is_alt: bool) -> np.ndarray:
        return self.binary_condition(X, self._mapping).predict(X, is_alt)

    def __call__(self, X: np.ndarray, y: np.ndarray, is_alt: bool) -> np.ndarray:
        return self.binary_condition(X, self._mapping)(X, y, is_alt)

    def offspring(self) -> "BinaryIntermediatePredictor":
        return deepcopy(self)


@attrs(auto_attribs=True, slots=True, frozen=True)
class LeastSquarePredictor(Predictor):

    n_dim: int
    init_weights: Optional[np.ndarray] = None
    _adaptive_filter: pa.filters.FilterRLS = attrib(init=False)

    def __attrs_post_init__(self):
        object.__setattr__(self, "_adaptive_filter", pa.filters.FilterRLS(self.n_dim))
        if self.init_weights is not None:
            self._adaptive_filter.w = self.init_weights

    def predict(self, X: np.ndarray, is_alt: bool) -> np.ndarray:
        x = np.hstack((1.0, X))
        return self._adaptive_filter.predict(x)

    def __call__(self, X: np.ndarray, y: np.ndarray, is_alt: bool) -> np.ndarray:
        x = np.hstack((1.0, X))
        self._adaptive_filter.adapt(y, x)
        return self._adaptive_filter.predict(x)

    def offspring(self) -> "LeastSquarePredictor":
        return LeastSquarePredictor(n_dim=self.n_dim, init_weights=self._adaptive_filter.w)


@attrs(auto_attribs=True, slots=True, frozen=True)
class GradientDecentPredictor(Predictor):

    sgd_factory: Callable[[], SGDRegressor] = SGDRegressor
    init_sgd: Optional[SGDRegressor] = None
    _sgd: SGDRegressor = attrib(init=False)

    def __attrs_post_init__(self):
        object.__setattr__(self, "_sgd", self.init_sgd if self.init_sgd is None else self.sgd_factory())

    def predict(self, X: np.ndarray, is_alt: bool) -> np.ndarray:
        return self._sgd.predict(X)

    def __call__(self, X: np.ndarray, y: np.ndarray, is_alt: bool) -> np.ndarray:
        y_pred = self.predict(X, is_alt=is_alt)
        self._sgd.partial_fit(X, y)
        return y_pred

    def offspring(self) -> "GradientDecentPredictor":
        return GradientDecentPredictor(self.sgd_factory, deepcopy(self._sgd))
