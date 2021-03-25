from typing import Callable, Iterable, Optional

import numpy as np
from attr import attrs, attrib
from tqdm import tqdm
from typing import Sequence

from fimtdd.nodes import LeafNode, Node


def _expand_data_matrix(X: np.ndarray) -> np.ndarray:
    if len(X.shape) == 1:
        X_matrix = np.expand_dims(X, 0)
    else:
        X_matrix = X
    return X_matrix


def _count(node: Node, leafs_only: bool = False) -> int:
    if type(node) == LeafNode:
        return 1
    else:
        sum = _count(node.left) + _count(node.right)
        sum += 0 if leafs_only else 1


@attrs(auto_attribs=True, slots=True)
class FIMTDD_Base:
    """
    The Learning Algorithm as Object
    """
    leaf_node_factory: Callable[[float, float, float, int, float, bool], LeafNode]
    alpha: float = 0.05
    gamma: float = 0.01
    threshold: float = 50
    n_min: int = 96
    learn_rate: float = 0.01
    verbose: bool = False
    _root_node: Node = attrib(init=False)

    def __attrs_post_init__(self):
        self._root_node = self.leaf_node_factory(self.alpha, self.gamma, self.threshold, self.n_min, self.learn_rate, self.verbose)
        self._root_node.set_root()

    def _obtain_progressbar(self, iterable: Iterable, desc: str = "", total: Optional[int] = None) -> Iterable:
        if self.verbose:
            return tqdm(iterable, desc=desc, total=total if total is not None else getattr(iterable, "__len__", 0))
        else:
            return iterable

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FIMTDD_Base":
        X_matrix = _expand_data_matrix(X)
        for X_i, y_i in self._obtain_progressbar(zip(X_matrix, y), desc="fitting on datapoints", total=len(y)):
            self._root_node.fit(X_i, y_i)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_matrix = _expand_data_matrix(X)
        y_pred = np.zeros(len(X_matrix))
        for i, X_i in self._obtain_progressbar(enumerate(X_matrix), desc="fitting on datapoints", total=len(X_matrix)):
            y_pred[i] = self._root_node.predict(X_i)
        return y_pred

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.predict(X)

    def __call__(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit_predict(X, y)

    def count_nodes(self) -> int:
        num_nodes = _count(self._root_node)
        return num_nodes

    def count_leafs(self) -> int:
        num_nodes = _count(self._root_node, leafs_only=True)
        return num_nodes
