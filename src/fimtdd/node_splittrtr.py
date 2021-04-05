from typing import Callable, Dict, Optional, Union, List

from attr import attrs, attrib

from fimtdd.binary_search.tree import EBSTree
from fimtdd.domain import Predictor, NodeSplitter
import numpy as np

from fimtdd.nodes import Node
from fimtdd.predictor import BinaryIntermediatePredictor
from collections import defaultdict

from fimtdd.utils import compute_standard_deviation_reduction


def ebstcollection():
    return defaultdict(EBSTree)


@attrs(auto_attribs=True, slots=True)
class MinSampleBinaryEBSTHoefdingerBoundNodeSplitter(NodeSplitter):

    n_min: int = 50
    _sample_counter_: int = 0
    _value_collections_: Dict[int, EBSTree] = attrib(factory=ebstcollection)

    def hoefdinger_bound(self) -> bool:
        ...

    def _find_best_split_by_depth_first_search(self, tree: EBSTree, sdr: Optional[Dict[str, float]] = None) -> Dict[str, Union[float, int]]:
        """
        Recursively calculate the best split of an attribute
        :param tree:    A EBST-tree
        :param sdr:     dictionary with global information for the tree search
        :return:        dictionary with best split and some additional information
        """
        if sdr is None:
            # this marks the start of the search
            sdr: Optional[Dict[str, float]] = dict()
            sdr['sumtotalLeft'] = 0.0
            sdr['sumtotalRight'] = tree.root.l_y + tree.root.r_y
            sdr['sumsqtotalLeft'] = 0.0
            sdr['sumsqtotalRight'] = tree.root.l_y_sq + tree.root.r_y_sq
            sdr['righttotal'] = tree.root.l_count + tree.root.r_count
            sdr['total'] = sdr['righttotal']
            sdr['n'] = sdr['total']
            sdr['max'] = None

        if tree.root.left is not None:
            # go to the leftmost path first
            self._find_best_split_by_depth_first_search(EBSTree(tree.root.left), sdr)

        sdr['sumtotalLeft'] = sdr['sumtotalLeft'] + tree.root.l_y
        sdr['sumtotalRight'] = sdr['sumtotalRight'] - tree.root.l_y
        sdr['sumsqtotalLeft'] = sdr['sumsqtotalLeft'] + tree.root.l_y_sq
        sdr['sumsqtotalRight'] = sdr['sumsqtotalRight'] - tree.root.l_y_sq
        sdr['righttotal'] = sdr['righttotal'] - tree.root.l_count

        new_sdr = compute_standard_deviation_reduction(sdr)
        if (sdr['max'] is None or new_sdr > sdr['max']):
            # if the standard deviation reduction is the largest, replace, else reduce
            sdr['2nd'] = sdr['max']
            sdr['max'] = new_sdr
            try:
                if not new_sdr == 0.0:
                    sdr['score'] = new_sdr  # sdr['2nd'] / new_sdr
                else:
                    sdr['score'] = 1.0
            except:
                sdr['score'] = 1.0
            sdr['bestsplit'] = tree.root.key

        if tree.root.right != None:
            self._find_best_split_by_depth_first_search(EBSTree(tree.root.right), sdr)

        sdr['sumtotalLeft'] = sdr['sumtotalLeft'] - tree.root.l_y
        sdr['sumtotalRight'] = sdr['sumtotalRight'] + tree.root.l_y
        sdr['sumsqtotalLeft'] = sdr['sumsqtotalLeft'] - tree.root.l_y_sq
        sdr['sumsqtotalRight'] = sdr['sumsqtotalRight'] + tree.root.l_y_sq
        sdr['righttotal'] = sdr['righttotal'] + tree.root.l_count
        return sdr

    def _obtain_max_index_from_splits(self, splits: List[Dict[str, Union[float, int]]]):
        """
        :param splits:  list of dictionaries containing the best split per attribute
        :return:        index of the dictionary with the best split over all attributes
        """
        max_index: Optional[int] = None
        second_place: Optional[int] = None
        for i in range(len(splits)):
            m = splits[i]['max']
            if max_index is None or m > max_index:
                second_place = max_index
                max_index = i

        if second_place is not None:
            splits[max_index]['score'] = splits[second_place]['max'] / splits[max_index]['max']
        return max_index

    def _obtain_condiction(self) -> Callable[[np.ndarray, Dict[bool, Node]], Node]:
        ...

    def update(self, X: np.ndarray, y: float, is_alt: bool) -> bool:
        for i, xi in enumerate(X):
            self._value_collections_[i].add(xi, y)
        self._sample_counter_ += 1
        return self._sample_counter_ != 0 and ((self._sample_counter_ % self.n_min) == 0) and not is_alt

    def copy(self) -> "MinSampleBinaryEBSTHoefdingerBoundNodeSplitter":
        return MinSampleBinaryEBSTHoefdingerBoundNodeSplitter()

    def split(self, node: Node, predictor: Predictor, is_alt: bool) -> Predictor:
        new_predictor_left, new_predictor_right = predictor.offspring(), predictor.offspring()
        left = Node(new_predictor_left,
                    node_splitter=node.node_splitter,
                    drift_detector=node.drift_detector,
                    swapping_criterion=node.swapping_criterion,
                    predictor_factory=predictor.offspring,
                    alt_predictor=None
                    )

        right = Node(new_predictor_left,
                     node_splitter=node.node_splitter,
                     drift_detector=node.drift_detector,
                     swapping_criterion=node.swapping_criterion,
                     predictor_factory=predictor.offspring,
                     alt_predictor=None
                     )
        return BinaryIntermediatePredictor(left, right, self._obtain_condiction())
