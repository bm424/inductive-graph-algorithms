"""Tests for functional graph algorithms."""
import pytest

from graph.collections import Tree
from graph.inductivegraph import (
    Graph,
    Context,
    Node,
    EmptyGraph,
    Adj,
    LRTree,
    LPath,
    LNode,
)


@pytest.mark.parametrize(
    "graph, expected",
    [
        pytest.param({}, {}, id="empty"),
        pytest.param({"a": "a"}, {"a": "a"}, id="loop"),
        pytest.param(
            {"a": "b"},
            {"b": "a"},
            id="single-edge",
        ),
        pytest.param(
            {"a": "b", "b": "c", "c": "a"},
            {"b": "a", "c": "b", "a": "c"},
            id="cycle",
        ),
        pytest.param(
            {"a": "b", "b": "a"},
            {"a": "b", "b": "a"},
            id="invariant",
        ),
        pytest.param(
            {"a": "bcd"},
            {"b": "a", "c": "a", "d": "a"},
            id="hub",
        ),
    ],
    indirect=True,
)
def test_grev(graph: Graph, expected: Graph):
    assert graph.grev() == expected
    assert graph.grev().grev() == graph


@pytest.mark.parametrize(
    "graph, expected_nodes",
    [
        pytest.param({"a": "b"}, ((Node(1), Node(2))), id="single-edge"),
        pytest.param({"a": "b", "b": "a"}, ((Node(1), Node(2))), id="loop"),
        pytest.param(
            {"a": "b", "b": "ac"},
            ((Node(1), Node(2), Node(3))),
            id="three-nodes",
        ),
    ],
    indirect=["graph"],
)
def test_nodes(graph: Graph, expected_nodes: frozenset[Node]):
    assert graph.nodes() == expected_nodes


@pytest.mark.parametrize(
    "graph, expected",
    [
        pytest.param({}, {}, id="empty"),
        pytest.param(
            Context((), Node(0), "a", ()) & EmptyGraph(),
            Context((), Node(0), "a", ()) & EmptyGraph(),
            id="single-node",
        ),
        pytest.param({"a": "a"}, {"a": "a"}, id="loop"),
        pytest.param(
            {"a": "b"},
            {"a": "b", "b": "a"},
            id="one-direction",
        ),
        pytest.param(
            {"a": "b", "b": "a"},
            {"a": "b", "b": "a"},
            id="already-undir",
        ),
    ],
    indirect=True,
)
def test_undir(graph: Graph, expected: Graph):
    assert graph.undir() == expected


@pytest.mark.parametrize(
    "graph, node, expected_node_labels",
    [
        pytest.param({"a": "bc"}, "a", ("b", "c"), id="two-suc"),
        pytest.param({"a": "bc"}, "b", (), id="no-suc"),
        pytest.param({"a": "bc", "b": "a"}, "b", ("a",), id="loop"),
        pytest.param(EmptyGraph(), Node(1), (), id="empty-graph"),
    ],
    indirect=["graph", "node"],
)
def test_gsuc(graph: Graph, node: Node, expected_node_labels: tuple[str, ...]):
    result = graph.gsuc(node)
    expected_nodes = tuple(
        graph.label_nodes[node_label] for node_label in expected_node_labels
    )
    assert result == expected_nodes


@pytest.mark.parametrize(
    "graph, node, expected",
    [
        pytest.param({"a": "b"}, "a", 1, id="pre-edge"),
        pytest.param({"a": "b"}, "b", 1, id="end-edge"),
        pytest.param({"a": "b", "b": "a"}, "a", 2, id="loop"),
        pytest.param({"a": "b", "b": "a"}, "b", 2, id="loop-reverse"),
        pytest.param({"a": "b", "b": "a"}, Node(3), None, id="node-not-in-graph"),
    ],
    indirect=["graph", "node"],
)
def test_deg(graph: Graph, node: Node, expected: int | None):
    assert graph.deg(node) == expected


@pytest.mark.parametrize(
    "graph, node, expected",
    [
        pytest.param(
            {"a": "b"}, "a", Context((), Node(2), "b", ()) & EmptyGraph(), id="pre-edge"
        ),
        pytest.param(
            {"a": "b"}, "b", Context((), Node(1), "a", ()) & EmptyGraph(), id="end-edge"
        ),
        pytest.param(
            {"a": "b", "b": "a"},
            "b",
            Context((), Node(1), "a", ()) & EmptyGraph(),
            id="loop",
        ),
        pytest.param({"a": "b", "b": "c"}, "a", {"b": "c"}, id="pre-chain"),
        pytest.param(
            {"a": "b", "b": "c"},
            "b",
            Context((), Node(1), "a", ())
            & Context((), Node(3), "c", ())
            & EmptyGraph(),
            id="mid-chain",
        ),
        pytest.param(
            {"a": "b", "b": "c"}, Node(4), {"a": "b", "b": "c"}, id="node-not-in-graph"
        ),
    ],
    indirect=True,
)
def test_rm(graph: Graph, node: Node, expected: Graph):
    assert graph.rm(node) == expected


@pytest.mark.parametrize(
    "graph, node, expected_labels",
    [
        pytest.param({"a": "b"}, "a", ("a", "b"), id="pre-edge"),
        pytest.param({"a": "b"}, "b", ("b",), id="end-edge"),
        pytest.param({"a": "b", "b": "c"}, "a", ("a", "b", "c"), id="pre-chain"),
        pytest.param({"a": "b", "b": "c"}, "b", ("b", "c"), id="mid-chain"),
        pytest.param({"a": "b", "b": "c"}, "c", ("c",), id="end-chain"),
        pytest.param(
            {"a": "bc", "b": "d", "c": "e"},
            "a",
            ("a", "b", "d", "c", "e"),
            id="pre-tree",
        ),
        pytest.param({"a": "b", "b": "c", "c": "a"}, "c", ("c", "a", "b"), id="cycle"),
        pytest.param(EmptyGraph(), Node(1), (), id="empty-graph"),
        pytest.param({"a": "b", "b": "c"}, Node(4), (), id="node-not-in-graph"),
    ],
    indirect=["graph", "node"],
)
def test_dfs(graph: Graph, node: Node, expected_labels: tuple[str, ...]):
    result = graph.dfs(node)
    expected_nodes = tuple(graph.label_nodes[label] for label in expected_labels)
    assert result == expected_nodes


@pytest.mark.parametrize(
    "graph, node, expected_labels",
    [
        pytest.param({"a": "b"}, "a", ("a", "b"), id="pre-edge"),
        pytest.param({"a": "b"}, "b", ("b",), id="end-edge"),
        pytest.param({"a": "b", "b": "c"}, "a", ("a", "b", "c"), id="pre-chain"),
        pytest.param(
            {"a": "bc", "b": "d", "c": "e"},
            "a",
            ("a", "b", "c", "d", "e"),
            id="pre-tree",
        ),
        pytest.param({"a": "b", "b": "c", "c": "a"}, "c", ("c", "a", "b"), id="cycle"),
        pytest.param(EmptyGraph(), Node(1), (), id="empty-graph"),
        pytest.param({"a": "b", "b": "c"}, Node(4), (), id="node-not-in-graph"),
    ],
    indirect=["graph", "node"],
)
def test_bfs(graph: Graph, node: Node, expected_labels: tuple[str, ...]):
    result = graph.bfs(node)
    labels = tuple(graph.node_labels[node] for node in result)
    assert labels == expected_labels


@pytest.mark.parametrize(
    "graph, expected_trees",
    [
        pytest.param(
            {"a": "bc", "d": "e"},
            (Tree("a", (Tree("b", ()), Tree("c", ()))), Tree("d", (Tree("e", ()),))),
            id="disconnected",
        ),
        pytest.param(
            {"a": "bcd", "d": "e"},
            (Tree("a", (Tree("b", ()), Tree("c", ()), Tree("d", (Tree("e", ()),)))),),
            id="connected",
        ),
    ],
    indirect=["graph"],
)
def test_dff(
    graph: Graph,
    expected_trees: tuple[Tree[str], ...],
):
    label_nodes = graph.label_nodes
    result = graph.dff()
    expected = tuple(tree.tmap(label_nodes.get) for tree in expected_trees)
    assert result == expected


@pytest.mark.parametrize(
    "graph, expected_node_labels",
    [
        pytest.param({"a": "b", "b": "c"}, ("a", "b", "c"), id="chain"),
        pytest.param({"a": "b", "b": "c", "c": "a"}, ("a", "b", "c"), id="cycle"),
        pytest.param(
            {"a": "bc", "c": "d", "d": "efg"},
            "acdgfeb",
            id="tree",
        ),
        pytest.param(
            {"a": "bcd", "c": "ag", "d": "efg", "e": "a"},
            "adfecgb",
            id="mixed",
        ),
        pytest.param(
            {"a": "cd", "c": "f", "d": "f", "b": "de", "e": "g"},
            "begadcf",
            id="medium-example",
        ),
    ],
    indirect=["graph"],
)
def test_topsort(graph: Graph, expected_node_labels: tuple[str, ...]):
    label_nodes = graph.label_nodes
    result = graph.topsort()
    assert result == tuple(label_nodes[label] for label in expected_node_labels)


@pytest.mark.parametrize(
    "graph, expected_trees",
    [
        pytest.param(
            {"a": "b", "b": "c", "c": "a", "d": "e", "e": "d"},
            (
                Tree("a", (Tree("c", (Tree("b", ()),)),)),
                Tree("d", (Tree("e", ()),)),
            ),
            id="disconnected",
        ),
        pytest.param(
            {
                "a": "b",
                "b": "cef",
                "c": "dg",
                "d": "ch",
                "e": "af",
                "f": "g",
                "g": "f",
                "h": "dg",
            },
            (
                Tree("a", (Tree("e", (Tree("b", ()),)),)),
                Tree("c", (Tree("d", (Tree("h", ()),)),)),
                Tree("f", (Tree("g", ()),)),
            ),
            id="wikipedia-example",
            # https://commons.wikimedia.org/wiki/File:Scc-1.svg#/media/File:Scc-1.svg
        ),
    ],
    indirect=["graph"],
)
def test_scc(graph: Graph, expected_trees: tuple[Tree, ...]):
    label_nodes = graph.label_nodes
    result = graph.scc()
    expected = tuple(tree.tmap(label_nodes.get) for tree in expected_trees)
    assert result == expected


@pytest.mark.parametrize(
    "graph, node, expected_paths",
    [
        pytest.param({"a": "b"}, "a", (("a",), ("b", "a")), id="pre-edge"),
        pytest.param({"a": "b", "b": "c"}, "a", ("a", "ba", "cba"), id="pre-chain"),
        pytest.param({"a": "b", "b": "c"}, Node(4), (), id="node-not-in-graph"),
    ],
    indirect=["graph", "node"],
)
def test_bft(graph: Graph, node: Node, expected_paths: tuple[str, ...]):
    result = graph.bft(node)
    expected = tuple(
        tuple(graph.label_nodes[label] for label in path) for path in expected_paths
    )
    assert result == expected


@pytest.mark.parametrize(
    "graph",
    [
        {
            "a": "b",
            "b": "cef",
            "c": "dg",
            "d": "ch",
            "e": "af",
            "f": "g",
            "g": "f",
            "h": "dg",
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "start_label, end_label, expected_path_labels",
    [
        pytest.param("a", "b", ("a", "b"), id="edge"),
        pytest.param("a", "h", ("a", "b", "c", "d", "h"), id="indirect"),
        pytest.param("c", "a", None, id="no-path"),
    ],
)
def test_esp(
    graph: Graph,
    start_label: str,
    end_label: str,
    expected_path_labels: tuple[str, ...],
):
    label_nodes = graph.label_nodes
    s = label_nodes[start_label]
    t = label_nodes[end_label]
    result = graph.esp(s, t)
    if expected_path_labels is None:
        assert result is None
    else:
        assert result == tuple(label_nodes[label] for label in expected_path_labels)


def test_sp():
    # https://commons.wikimedia.org/wiki/File:Dijkstra_Animation.gif#/media/File:Dijkstra_Animation.gif
    graph = (
        Context(Adj(), Node(1), 1, Adj(((7, Node(2)), (9, Node(3)), (14, Node(6)))))
        & Context(Adj(), Node(2), 2, Adj(((10, Node(3)), (15, Node(4)))))
        & Context(Adj(), Node(3), 3, Adj(((11, Node(4)), (2, Node(6)))))
        & Context(Adj(), Node(4), 4, Adj(((6, Node(5)),)))
        & Context(Adj(), Node(5), 5, Adj(((9, Node(6)),)))
        & Context(Adj(), Node(6), 6, Adj())
        & EmptyGraph()
    ).undir()
    result = graph.sp(Node(1), Node(5))
    expected = (Node(1), Node(3), Node(6), Node(5))
    assert result == expected


def test_mst():
    # https://en.wikipedia.org/wiki/File:Minimum_spanning_tree.svg
    graph = (
        Context(
            Adj(), Node(1), 1, Adj(((6.0, Node(2)), (3.0, Node(3)), (9.0, Node(4))))
        )
        & Context(
            Adj(), Node(2), 2, Adj(((2.0, Node(5)), (4.0, Node(3)), (9.0, Node(6))))
        )
        & Context(
            Adj(), Node(3), 3, Adj(((2.0, Node(5)), (9.0, Node(7)), (9.0, Node(5))))
        )
        & Context(Adj(), Node(4), 4, Adj(((8.0, Node(7)), (18.0, Node(10)))))
        & Context(Adj(), Node(5), 5, Adj(((9.0, Node(6)), (8.0, Node(7)))))
        & Context(
            Adj(), Node(6), 6, Adj(((7.0, Node(7)), (4.0, Node(8)), (5.0, Node(9))))
        )
        & Context(Adj(), Node(7), 7, Adj(((9.0, Node(9)), (10.0, Node(10)))))
        & Context(Adj(), Node(8), 8, Adj(((1.0, Node(9)), (4.0, Node(10)))))
        & Context(Adj(), Node(9), 9, Adj(((3.0, Node(10)),)))
        & Context(Adj(), Node(10), 10, Adj())
        & EmptyGraph()
    ).undir()
    result = graph.mst(Node(1))
    expected = LRTree(
        (
            LPath((LNode(Node(1), 0.0),)),
            LPath((LNode(Node(3), 3.0), LNode(Node(1), 0.0))),
            LPath((LNode(Node(5), 2.0), LNode(Node(3), 3.0), LNode(Node(1), 0.0))),
            LPath(
                (
                    LNode(Node(2), 2.0),
                    LNode(Node(5), 2.0),
                    LNode(Node(3), 3.0),
                    LNode(Node(1), 0.0),
                )
            ),
            LPath(
                (
                    LNode(Node(7), 8.0),
                    LNode(Node(5), 2.0),
                    LNode(Node(3), 3.0),
                    LNode(Node(1), 0.0),
                )
            ),
            LPath(
                (
                    LNode(Node(6), 7.0),
                    LNode(Node(7), 8.0),
                    LNode(Node(5), 2.0),
                    LNode(Node(3), 3.0),
                    LNode(Node(1), 0.0),
                )
            ),
            LPath(
                (
                    LNode(Node(8), 4.0),
                    LNode(Node(6), 7.0),
                    LNode(Node(7), 8.0),
                    LNode(Node(5), 2.0),
                    LNode(Node(3), 3.0),
                    LNode(Node(1), 0.0),
                )
            ),
            LPath(
                (
                    LNode(Node(9), 1.0),
                    LNode(Node(8), 4.0),
                    LNode(Node(6), 7.0),
                    LNode(Node(7), 8.0),
                    LNode(Node(5), 2.0),
                    LNode(Node(3), 3.0),
                    LNode(Node(1), 0.0),
                )
            ),
            LPath(
                (
                    LNode(Node(10), 3.0),
                    LNode(Node(9), 1.0),
                    LNode(Node(8), 4.0),
                    LNode(Node(6), 7.0),
                    LNode(Node(7), 8.0),
                    LNode(Node(5), 2.0),
                    LNode(Node(3), 3.0),
                    LNode(Node(1), 0.0),
                )
            ),
            LPath(
                (
                    LNode(Node(4), 8.0),
                    LNode(Node(7), 8.0),
                    LNode(Node(5), 2.0),
                    LNode(Node(3), 3.0),
                    LNode(Node(1), 0.0),
                )
            ),
        )
    )
    assert result == expected


@pytest.mark.parametrize(
    "graph, expected_labels",
    [
        pytest.param(
            {"a": "b", "b": "c"},
            "ac",
            id="line-graph",
            # https://en.wikipedia.org/wiki/File:Mis_pathgraph_p3.png
        ),
        pytest.param(
            {"1": "237", "2": "48", "3": "45", "4": "6", "5": "67", "6": "8", "7": "8"},
            "1458",
            id="cube-graph",
            # https://en.wikipedia.org/wiki/File:Cube-maximal-independence.svg
        ),
        pytest.param(
            {"0": "12345678"},
            "12345678",
            id="star-graph",
            # https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/Mis_stargraph_s8.png/220px-Mis_stargraph_s8.png
        ),
    ],
    indirect=["graph"],
)
def test_undir_indep(graph, expected_labels):
    result = graph.undir().indep()
    expected = tuple(graph.label_nodes[label] for label in expected_labels)
    assert result == expected
