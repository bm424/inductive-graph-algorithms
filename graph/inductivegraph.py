"""Definition and implementation of an inductive graph and its algorithms."""
import abc
import collections.abc
import functools
import typing

from graph.collections import Tree, ImmutableHeap
from graph.tools import first, nub, concat_map

A = typing.TypeVar("A")
B = typing.TypeVar("B")
C = typing.TypeVar("C")
D = typing.TypeVar("D")


class Node(int):
    """A node.

    For convenience, nodes are represented by integers.
    """


class Adj(collections.abc.Sequence[tuple[B, Node]]):
    """Adjacency relationships."""

    edges: tuple[tuple[B, Node], ...]

    def __init__(self, edges: tuple[tuple[B, Node], ...] = ()):
        self.edges = edges

    def __repr__(self):
        return f"Adj({self.edges!r})"

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, item):
        return self.edges[item]

    def __eq__(self, other):
        return self.edges == other.edges

    def __add__(self, other):
        return Adj(self.edges + other.edges)


class Context(typing.Generic[A, B]):
    """A context.

    A node's context describes (some of) its surroundings, including its label, its
    adjacent predecessors and its adjacent successors.
    """

    predecessors: Adj[B]
    node: Node
    label: A
    successors: Adj[B]

    def __init__(
        self,
        predecessors: Adj[B],
        node: Node,
        label: A,
        successors: Adj[B],
    ):
        self.predecessors = predecessors
        self.node = node
        self.label = label
        self.successors = successors

    def __repr__(self):
        return f"Context({self.predecessors!r}, {self.node!r}, {self.label!r}, {self.successors!r})"

    def __eq__(self, other):
        return (
            self.predecessors == other.predecessors
            and self.node == other.node
            and self.label == other.label
            and self.successors == other.successors
        )

    def pop(self, node: Node) -> tuple["Context[None, B]", "Context[A, B]"]:
        node_predecessors = Adj(
            tuple((l, self.node) for l, n in self.successors if n == node)
        )
        node_successors = Adj(
            tuple((l, self.node) for l, n in self.predecessors if n == node)
        )
        node_context = Context(node_predecessors, node, None, node_successors)

        predecessors = Adj(tuple((l, n) for l, n in self.predecessors if n != node))
        successors = Adj(tuple((l, n) for l, n in self.successors if n != node))
        context = Context(predecessors, self.node, self.label, successors)
        return node_context, context

    def __or__(self, other):
        predecessors = Adj(nub((*self.predecessors, *other.predecessors)))
        successors = Adj(nub((*self.successors, *other.successors)))
        return Context(predecessors, self.node, self.label, successors)

    def __and__(self, other):
        match other:
            case Context():
                return _ContextPartial((other, self))
            case InductiveGraph() if self.node in other.nodes():
                raise NodeExistsError(
                    f"context {self} refers to existing node {self.node} in graph {other}"
                )
            case InductiveGraph() if disjoint_nodes := tuple(
                node
                for node in self.pre + self.suc
                if node not in other.nodes() and node != self.node
            ):
                raise NodeDoesNotExistError(
                    f"context {self} "
                    f"refers to adjacent nodes {disjoint_nodes} "
                    f"which are not in graph {other}"
                )
            case InductiveGraph() | EmptyGraph():
                return InductiveGraph(self, other)

    def map_label(self, fn: typing.Callable[[A], C]) -> "Context[C, B]":
        return Context(self.predecessors, self.node, fn(self.label), self.successors)

    @classmethod
    def label_mapper(
        cls, fn: typing.Callable[[A], C]
    ) -> typing.Callable[["Context[A, B]"], "Context[C, B]"]:
        return functools.partial(cls.map_label, fn=fn)

    @property
    def pre(self) -> tuple[Node, ...]:
        return tuple(node for _, node in self.predecessors)

    @property
    def suc(self) -> tuple[Node, ...]:
        return tuple(node for _, node in self.successors)


class Graph(abc.ABC, typing.Generic[A, B]):
    """An abstract graph.

    Implementations should define `is_empty` and `pop`.
    """

    @property
    @abc.abstractmethod
    def is_empty(self) -> bool:
        """True if the graph is empty (contains no nodes), False otherwise."""

    @abc.abstractmethod
    def pop(
        self, node: Node | None = None
    ) -> tuple[Context[A, B], "Graph[A, B]"] | None:
        """Extracts a given `node` from the graph.

        Returns the node's context, and the remaining graph. If `node` is None, any
        node may be extracted. If `node` is not in the graph, returns None.
        """

    @property
    def vertices(self) -> frozenset["LNode[A]"]:
        """The set of (labeled) nodes in the graph."""

        def add_vertex(
            context: Context[A, B], result: frozenset[LNode[A]]
        ) -> frozenset[LNode[A]]:
            return result | {LNode(context.node, context.label)}

        return self.ufold(add_vertex, frozenset())

    @property
    def edges(self) -> frozenset["_Edge[B]"]:
        """The set of (labeled) edges in the graph."""
        vertex_dict = {vertex.node: vertex for vertex in self.vertices}

        def add_edges(
            context: Context[A, B], result: frozenset["_Edge[B]"]
        ) -> frozenset["_Edge[B]"]:
            return (
                result
                | {
                    _Edge(vertex_dict[node], label, vertex_dict[context.node])
                    for label, node in context.predecessors
                }
                | {
                    _Edge(vertex_dict[context.node], label, vertex_dict[node])
                    for label, node in context.successors
                }
            )

        return self.ufold(add_edges, frozenset())

    def __eq__(self, other):
        return self.vertices == other.vertices and self.edges == other.edges

    @property
    def node_labels(self) -> dict[Node, A]:
        """Maps nodes to their labels."""

        def add(context: Context[A, B], result: dict[Node, A]) -> dict[Node, A]:
            return {**result, context.node: context.label}

        return self.ufold(add, {})

    @property
    def label_nodes(self) -> dict[A, Node]:
        """Maps labels to their nodes."""

        def add(context: Context[A, B], result: dict[A, Node]) -> dict[A, Node]:
            return {**result, context.label: context.node}

        return self.ufold(add, {})

    def ufold(self, fn: typing.Callable[[Context[A, B], C], C], u: C) -> C:
        """Un-ordered fold."""
        if self.is_empty:
            return u
        head, graph = self.pop()
        return fn(head, graph.ufold(fn, u))

    def gmap(
        self, fn: typing.Callable[[Context[A, B]], Context[C, D]]
    ) -> "Graph[C, D]":
        """Convert the graph into another graph via `fn` over its contexts."""

        return self.ufold(
            lambda context, result: fn(context) & result, EmptyGraph[C, D]()
        )

    def grev(self) -> "Graph[A, B]":
        """Inverts the graph (reverses its directed edges)."""

        def swap(context: Context[A, B]) -> Context[A, B]:
            return Context(
                context.successors,
                context.node,
                context.label,
                context.predecessors,
            )

        return self.gmap(swap)

    def nodes(self) -> tuple[Node, ...]:
        """The nodes of the graph."""

        return self.ufold(lambda context, result: (context.node, *result), ())

    def undir(self) -> "Graph[A, B]":
        """Converts a directed graph into an undirected graph."""

        def undir_context(context: Context[A, B]) -> Context[A, B]:
            adjacents = Adj(nub(context.predecessors + context.successors))
            return Context(adjacents, context.node, context.label, adjacents)

        return self.gmap(undir_context)

    def gsuc(self, node: Node) -> tuple[Node, ...]:
        """The successor nodes of `node` in the graph."""
        popped = self.pop(node)
        if not popped:
            return ()
        context, _ = popped
        return context.suc

    def deg(self, node: Node) -> int | None:
        """Degree of `node` (the number of connected nodes)."""
        popped = self.pop(node)
        if not popped:
            return None
        context, _ = popped
        return len(context.predecessors) + len(context.successors)

    def rm(self, node: Node) -> "Graph[A, B]":
        """Graph with `node` removed."""
        popped = self.pop(node)
        if not popped:
            return self
        _, graph = popped
        return graph

    def _dfs(self, nodes: tuple[Node, ...]) -> tuple[Node, ...]:
        if not nodes:
            return ()
        if self.is_empty:
            return ()
        head, *tail = nodes
        match self.pop(head):
            case (context, graph):
                return head, *graph._dfs(context.suc + tuple(tail))
            case None:
                return self._dfs(tuple(tail))

    def dfs(self, node: Node) -> tuple[Node, ...]:
        """Depth-first search from `node`."""
        return self._dfs((node,))

    def _bfs(self, nodes: tuple[Node, ...]) -> tuple[Node, ...]:
        if not nodes:
            return ()
        if self.is_empty:
            return ()
        head, *tail = nodes
        match self.pop(head):
            case (context, graph):
                return head, *graph._bfs(tuple(tail) + context.suc)
            case None:
                return self._bfs(tuple(tail))

    def bfs(self, node: Node) -> tuple[Node, ...]:
        """Breadth-first search from `node`."""
        return self._bfs((node,))

    def _dff(
        self, nodes: tuple[Node, ...]
    ) -> tuple[tuple[Tree[Node], ...], "Graph[A, B]"]:
        if not nodes:
            return (), self
        head, *tail = nodes
        match self.pop(head):
            case c, g:
                f, g1 = g._dff(c.suc)
                f_, g2 = g1._dff(tuple(tail))
                return (Tree(head, f), *f_), g2
            case None:
                return self._dff(tuple(tail))

    def dff(self) -> tuple[Tree[Node], ...]:
        """Depth-first spanning forest."""
        return self._dff(self.nodes())[0]

    def topsort(self) -> tuple[Node, ...]:
        """Topological sort."""
        return tuple(reversed(concat_map(Tree[Node].post_order, self.dff())))

    def scc(self) -> tuple[Tree[Node], ...]:
        """Strongly-connected components."""
        return self.grev().dff()

    def _bf(self, paths: tuple[tuple[Node, ...], ...]) -> tuple[tuple[Node, ...], ...]:
        if self.is_empty:
            return ()
        if not paths:
            return ()
        p, *ps = paths
        v, *_ = p
        match self.pop(v):
            case (c, g):
                return p, *g._bf(tuple(ps) + tuple((s, *p) for s in c.suc))
            case None:
                return self._bf(tuple(ps))

    def bft(self, node: Node) -> tuple[tuple[Node, ...], ...]:
        """Breadth-first trees.

        Paths starting from all nodes connected to `node` leading to `node`.
        """
        return self._bf(((node,),))

    def esp(self, s: Node, t: Node) -> tuple[Node, ...] | None:
        """Edge-shortest path between nodes `s` and `t`.

        Does not take into account edge labels.
        """
        result = first(lambda path: path[0] == t, self.bft(s))
        return result if result is None else tuple(reversed(result))

    @staticmethod
    def _expand(
        item: float, lpath: "LPath[float]", context: Context[A, float]
    ) -> tuple[ImmutableHeap["LPath[float]"], ...]:
        return tuple(
            ImmutableHeap.unit(LPath((LNode(node, item + label), *lpath)))
            for label, node in context.successors
        )

    def _dijkstra(self, heap: ImmutableHeap["LPath[float]"]) -> "LRTree[float]":
        if not heap or self.is_empty:
            return LRTree()
        p, h = heap.pop()
        (v, d), *_ = p
        match self.pop(v):
            case (c, g):
                return LRTree(
                    (p, *g._dijkstra(ImmutableHeap.merge(heap, *self._expand(d, p, c))))
                )
            case None:
                return self._dijkstra(h)

    def _spt(self, node: Node) -> "LRTree[float]":
        heap = ImmutableHeap.unit(LPath((LNode(node, 0.0),)))
        return self._dijkstra(heap)

    def sp(self, s: Node, t: Node) -> tuple[Node, ...] | None:
        """Shortest path between nodes `s` and `t`.

        Uses Dijkstra's algorithm with edge labels representing distance.
        """
        return self._spt(s).get_path(t)

    @staticmethod
    def _add_edges(
        lpath: "LPath[B]", context: Context[A, B]
    ) -> tuple[ImmutableHeap["LPath[B]"], ...]:
        return tuple(
            ImmutableHeap.unit(LPath((LNode(node, label), *lpath)))
            for label, node in context.successors
        )

    def _prim(self, heap: ImmutableHeap["LPath[float]"]) -> "LRTree[float]":
        if not heap or self.is_empty:
            return LRTree()
        p, h = heap.pop()
        (v, _), *_ = p
        match self.pop(v):
            case (c, g):
                return LRTree(
                    (p, *g._prim(ImmutableHeap.merge(h, *self._add_edges(p, c))))
                )
            case None:
                return self._prim(h)

    def mst(self, node: Node | None) -> "LRTree[float]":
        if node is None:
            node = self.pop()[0].node
        return self._prim(ImmutableHeap.unit(LPath((LNode(node, 0.0),))))

    def indep(self) -> tuple[Node, ...]:
        """Maximum independent node set."""
        if self.is_empty:
            return ()
        vs = self.nodes()
        m = max(map(self.deg, vs))
        v = first(lambda v: self.deg(v) == m, vs)
        c, g_ = self.pop(v)
        i1 = g_.indep()
        i2 = (v, *functools.reduce(Graph.rm, c.pre + c.suc, g_).indep())
        return i1 if len(i1) > len(i2) else i2


class EmptyGraph(Graph[A, B]):
    """An empty graph, containing no nodes or edges."""

    def __repr__(self):
        return "EmptyGraph"

    @property
    def is_empty(self) -> bool:
        return True

    def pop(
        self, node: typing.Optional[Node] = None
    ) -> tuple[Context[A, B], "Graph[A, B]"] | None:
        return None


class InductiveGraph(Graph[A, B]):
    """An inductive graph.

    The `head` of the graph is a context that can only refer to nodes in the `tail`,
    which is also a graph.
    This makes arbitrary context extraction efficient.
    """

    head: Context[A, B]
    tail: Graph[A, B]

    def __init__(self, head: Context[A, B], tail: Graph[A, B]):
        self.head = head
        self.tail = tail

    def __repr__(self):
        return f"{self.head!r} & {self.tail!r}"

    @property
    def is_empty(self) -> bool:
        return False

    def pop(
        self, node: typing.Optional[Node] = None
    ) -> tuple[Context[A, B], "Graph[A, B]"] | None:
        if node is None or node == self.head.node:
            return self.head, self.tail
        node_context, graph_context = self.head.pop(node)
        if match := self.tail.pop(node):
            sub_node_context, subgraph = match
            return node_context | sub_node_context, graph_context & subgraph
        return None


class LNode(typing.Generic[A]):
    """Labeled node."""

    node: Node
    label: A

    def __init__(self, node: Node, label: A):
        self.node = node
        self.label = label

    def __repr__(self):
        return f"LNode({self.node!r}, {self.label!r})"

    def __eq__(self, other):
        return self.label == other.label

    def __hash__(self):
        return hash(self.label)

    def __iter__(self):
        yield self.node
        yield self.label


class LPath(typing.Sequence[LNode[A]]):
    """Labeled path.

    A path is a queue of nodes, so a labeled path is a queue of labeled nodes.

    A labeled path is "equal" to another if their first labels are equal.
    A labeled path is "less than" another if its first label is less than the other's
    first label.
    """

    lnodes: tuple[LNode[A], ...]

    def __init__(self, lnodes: typing.Sequence[LNode[A]] = ()):
        self.lnodes = tuple(lnodes)

    def __repr__(self):
        return f"LPath({self.lnodes!r})"

    def __iter__(self):
        return iter(self.lnodes)

    def __getitem__(self, item):
        return self.lnodes[item]

    def __len__(self):
        return len(self.lnodes)

    def __eq__(self, other):
        (_, x), *_ = self.lnodes
        (_, y), *_ = other.lnodes
        return x == y

    def __lt__(self, other):
        (_, x), *_ = self.lnodes
        (_, y), *_ = other.lnodes
        return x < y


class LRTree(typing.Sequence[LPath[A]]):
    """Labeled R-tree.

    An R-tree is a queue of paths originating from the same node, ending at that node,
    so a labeled R-tree is a queue of labeled paths originating from the same labeled
    node, ending at that labeled node.
    """

    lpaths: tuple[LPath[A], ...]

    def __init__(self, lpaths: typing.Sequence[LPath] = ()):
        self.lpaths = tuple(lpaths)

    def __getitem__(self, item):
        return self.lpaths[item]

    def __len__(self):
        return len(self.lpaths)

    def __eq__(self, other):
        return self.lpaths == other.lpaths

    def __repr__(self):
        return f"LRTree({self.lpaths!r})"

    def get_path(self, node: Node) -> tuple[Node, ...] | None:
        def path_node_is_node(lpath: LPath[A]) -> bool:
            return lpath[0].node == node

        path = first(path_node_is_node, self.lpaths)
        return tuple(reversed([lnode.node for lnode in path])) if path else None


class NodeExistsError(ValueError):
    """Raised when context is added to a graph already containing its node."""


class NodeDoesNotExistError(ValueError):
    """Raised when context is added to a graph not containing its adjacent nodes."""


class _ContextPartial(typing.Generic[A, B]):
    """A limited collection of contexts.

    We want to construct graphs like `context_0 & context_1 & ... & empty_graph`, but
    this requires right-to-left resolution and Python only provides left-to-right
    resolution. We can emulate this by allowing `context_0 & context_1` to construct a
    "context_partial" which implements `__and__` such that `context_partial & graph`
    returns a graph resolved in the correct order.
    """

    contexts: tuple["Context[A, B]", ...]

    def __init__(self, contexts: tuple["Context[A, B]", ...]):
        self.contexts = contexts

    def __and__(self, other):
        match other:
            case Context():
                return _ContextPartial((other, *self.contexts))
            case Graph():
                return functools.reduce(lambda a, b: b & a, self.contexts, other)


class _Edge(typing.Generic[B]):
    """Convenience class for establishing graph equivalence."""

    source: LNode
    label: B
    target: LNode

    def __init__(self, source: LNode, label: B, target: LNode):
        self.source = source
        self.label = label
        self.target = target

    def __eq__(self, other):
        return (
            self.source == other.source
            and self.label == other.label
            and self.target == other.target
        )

    def __hash__(self):
        return hash((self.source, self.label, self.target))
