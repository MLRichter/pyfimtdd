from attr import attrs


@attrs(auto_attribs=True, slots=True)
class Node:

    is_root: bool = False
    alpha: float = 0.05
    gamma: float = 0.01
    threshold: float = 50
    n_min: int = 96
    learn_rate: float = 0.01
    verbose: bool = False

    def set_root(self):
        self.is_root = True


@attrs(auto_attribs=True, slots=True)
class IntermediateNode(Node):
    ...


@attrs(auto_attribs=True, slots=True)
class LeafNode(Node):
    ...
