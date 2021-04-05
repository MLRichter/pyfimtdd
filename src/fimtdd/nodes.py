from typing import Optional, Callable

from attr import attrs

from fimtdd.domain import Predictor, NodeSplitter, DriftDetector, SwappingCriterion
import numpy as np


@attrs(auto_attribs=True, slots=True)
class Node(Predictor):

    predictor: Predictor
    node_splitter: NodeSplitter
    drift_detector: DriftDetector
    swapping_criterion: SwappingCriterion
    alt_predictor: Optional[Predictor]
    predictor_factory: Callable[[Predictor], Predictor]

    def predict(self, X: np.ndarray, is_alt: bool = False) -> np.ndarray:
        return self.predictor.predict(X, is_alt)

    def __call__(self, X: np.ndarray, y: np.ndarray, is_alt: bool = False) -> np.ndarray:
        y_pred = self.predictor.fit_predict(X, y, is_alt)
        if self.drift_detector.update(y, y_pred):
            self.alt_predictor = self.predictor_factory(self.predictor)
        if self.alt_predictor is not None:
            alt_y_pred = self.alt_predictor.fit_predict(X, y, True)
            if self.swapping_criterion.update(y, y_pred, alt_y_pred):
                self.predictor = self.alt_predictor
                self.alt_predictor = None
        if self.node_splitter.update(X, y, is_alt):
            self.predictor = self.node_splitter.split(self, self.predictor, is_alt)
        return y_pred

    def offspring(self) -> "Predictor":
        return self.predictor.offspring()
