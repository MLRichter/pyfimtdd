from .domain import DriftDetector
from attr import attrs, attrib
import numpy as np


@attrs(auto_attribs=True, slots=True)
class PageHinckleyDetector(DriftDetector):

    threshold: int = 50
    alpha: float = 0.005
    min_samples: int = 50
    _seen_samples_: int = 0
    _cumulative_error_: float = 0.0
    _current_score_: float = 0.0
    _min_score_: float = np.nan

    def update(self, y_true: float, y_pred: float) -> None:
        self._seen_samples_ += 1
        error = np.fabs(y_true - y_pred)
        self._cumulative_error_ += error
        self._current_score_ += error - (self._cumulative_error_ / self._seen_samples_) - self.alpha
        if np.isnan(self._min_score_) or self._current_score_ < self._min_score_:
            self._min_score_ = self._current_score_

    def detect_drift(self) -> bool:
        return (self._current_score_ - self._min_score_) > self.threshold
