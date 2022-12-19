"""Custom collection types to facilitate inductive graph algorithms."""
import heapq
import typing

from graph.tools import concat_map

A = typing.TypeVar("A", bound=typing.Hashable)
B = typing.TypeVar("B")
C = typing.TypeVar("C")


class Tree(typing.Generic[A]):
    """A labeled tree structure."""

    branch: A
    trees: tuple["Tree[A]", ...]

    def __init__(self, branch: A, trees: tuple["Tree[A]", ...]):
        self.branch = branch
        self.trees = trees

    def __repr__(self):
        return f"Tree({self.branch!r}, {self.trees!r})"

    def __eq__(self, other) -> bool:
        match other:
            case Tree(branch=branch, trees=trees):
                return self.branch == branch and self.trees == trees
        return NotImplemented

    def __hash__(self):
        return hash((self.branch, hash(self.trees)))

    def post_order(self) -> tuple[A, ...]:
        """Nodes of the tree in reverse (post-) order, depth-first."""
        return concat_map(Tree.post_order, self.trees) + (self.branch,)

    def tmap(self, fn: typing.Callable[[A], B]) -> "Tree[B]":
        """Converts the tree's nodes recursively via the function `fn`."""
        return Tree(fn(self.branch), tuple(tree.tmap(fn) for tree in self.trees))


class ImmutableHeap(typing.Generic[A]):
    """An immutable heap.

    Wraps the relevant `heapq` methods so that the heap is copied before modification.
    """

    def __init__(self, heap: list[A]):
        self.heap = heap

    def __bool__(self):
        return bool(self.heap)

    @classmethod
    def empty(cls) -> "ImmutableHeap[A]":
        return cls([])

    @classmethod
    def unit(cls, item: A) -> "ImmutableHeap[A]":
        return cls([item])

    @classmethod
    def from_iterable(cls, iterable: typing.Sequence[A]) -> "ImmutableHeap[A]":
        heap = list(iterable)
        heapq.heapify(heap)
        return cls(heap)

    @classmethod
    def merge(cls, *heaps: "ImmutableHeap[A]") -> "ImmutableHeap[A]":
        heap = list(heapq.merge(*(heap.heap for heap in heaps)))
        return cls(heap)

    def _clone(self):
        return ImmutableHeap(list(self.heap))

    def push(self, item: A) -> "ImmutableHeap[A]":
        heap = self._clone().heap
        heapq.heappush(heap, item)
        return ImmutableHeap(heap)

    def pop(self) -> tuple[A, "ImmutableHeap[A]"]:
        heap = self._clone().heap
        item = heapq.heappop(heap)
        return item, ImmutableHeap(heap)
