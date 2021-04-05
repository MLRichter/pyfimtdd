import numpy as np
from typing import Dict
from attr import attrs

from fimtdd.nodes import Node


@attrs(auto_attribs=True, slots=True, frozen=True)
class BinaryThresholdDecisionStrategy:

    key: float
    key_dim: int

    def __call__(self, X: np.ndarray, mapping: Dict[bool, Node]) -> Node:
        return mapping[X[self.key_dim] <= self.key]
