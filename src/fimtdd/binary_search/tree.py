from typing import Optional, Any, Protocol

from attr import attrs, attrib


class TreeNode(Protocol):

    key: float
    y: float
    parent: Optional["TreeNode"]
    left: Optional["TreeNode"]
    right: Optional["TreeNode"]
    l_count: int
    l_y: float
    l_y_sq: float
    r_count: int
    r_y: float
    r_y_sq: float

    def add(self, val: float, y: float) -> None:
        ...


@attrs(auto_attribs=True, slots=True)
class NodeEBST(TreeNode):

    key: float
    y: float
    parent: Optional[TreeNode] = None
    left: Optional[TreeNode] = attrib(init=False, default=None)
    right: Optional[TreeNode] = attrib(init=False, default=None)
    l_count: int = attrib(init=False, default=1)
    l_y: float = attrib(init=False)
    l_y_sq: float = attrib(init=False)
    r_count: int = attrib(init=False, default=0)
    r_y: float = attrib(init=False, default=0)
    r_y_sq: float = attrib(init=False, default=0)

    def __attrs_post_init__(self):
        self.l_y = self.y
        self.l_y_sq = self.y**2

    def add(self, val: float, y: float) -> None:
        if val <= self.key:
            self.l_count += 1
            self.l_y += y
            self.l_y_sq += y**2
            if self.left is None and val != self.key:
                self.left = NodeEBST(val, y, self)
            elif val == self.key:
                pass
            else:
                self.left.add(val,y)
        else:
            self.r_count += 1
            self.r_y += y
            self.r_y_sq += y**2
            if self.right is None:
                self.right = NodeEBST(val, y, self)
            else:
                self.right.add(val,y)
        return


@attrs(auto_attribs=True, slots=True)
class EBSTree:

    root: Optional[TreeNode] = None

    def add(self, key: float, y: float) -> None:
        if self.root is None:
            self.root = NodeEBST(key, y)
        else:
            self.root.add(key, y)
