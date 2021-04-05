from .domain import SwappingCriterion
from attr import attrs, attrib
import numpy as np
from .domain import SwappingCriterion


@attrs(auto_attribs=True, slots=True)
class DecayingSquaredErrorSwapper(SwappingCriterion):

    decay: float = 0.995
    check_intervall: int = 50
    _samples: int = 0
    _cumulative_squared_error: float = 0.0
    _cumulative_squared_error_alt: float = 0.0

    def update(self, y: float, y_pred: float, alt_y_pred: float) -> bool:
        self._samples += 1
        self._cumulative_squared_error = (self._cumulative_squared_error*self.decay) + ((y-y_pred)**2)
        self._cumulative_squared_error_alt = (self._cumulative_squared_error_alt*self.decay) + ((y-alt_y_pred)**2)
        if self._samples == 0 or self._samples%self.check_intervall != 0:
            return False
        else:
            return np.log(self._cumulative_squared_error / self._cumulative_squared_error_alt) > 0

    def copy(self) -> "SwappingCriterion":
        return DecayingSquaredErrorSwapper(self.decay, self.check_intervall)
