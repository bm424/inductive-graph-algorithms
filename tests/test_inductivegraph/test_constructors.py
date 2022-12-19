"""Tests for the construction and comparison of graphs."""
import pytest

from graph.inductivegraph import (
    EmptyGraph,
    Context,
    Node,
    Adj,
    NodeExistsError,
    NodeDoesNotExistError,
    Graph,
)


def test_invalid_construction_node_exists():
    with pytest.raises(NodeExistsError):
        (
            Context(Adj(), Node(1), "b", Adj())
            & Context(Adj(), Node(1), "a", Adj())
            & EmptyGraph()
        )


def test_invalid_construction_node_does_not_exist():
    with pytest.raises(NodeDoesNotExistError):
        (
            Context(Adj(((None, Node(3)),)), Node(2), "b", Adj())
            & Context(Adj(), Node(1), "a", Adj())
            & EmptyGraph()
        )


@pytest.mark.parametrize(
    "graph_a, graph_b",
    [
        pytest.param(EmptyGraph(), EmptyGraph(), id="empty"),
        pytest.param(
            Context(Adj(), Node(0), "a", Adj()) & EmptyGraph(),
            Context(Adj(), Node(0), "a", Adj()) & EmptyGraph(),
            id="singular",
        ),
        pytest.param(
            Context(Adj(((None, Node(0)),)), Node(0), "a", Adj()) & EmptyGraph(),
            Context(Adj(), Node(0), "a", Adj(((None, Node(0)),))) & EmptyGraph(),
            id="loop",
        ),
        pytest.param(
            Context(Adj(), Node(1), "a", Adj(((None, Node(0)),)))
            & Context(Adj(), Node(0), "b", Adj())
            & EmptyGraph(),
            Context(Adj(((None, Node(0)),)), Node(1), "b", Adj())
            & Context(Adj(), Node(0), "a", Adj())
            & EmptyGraph(),
            id="cyclic",
        ),
        pytest.param(
            Context(Adj(), Node(0), "a", Adj(((None, Node(1)),)))
            & Context(Adj(), Node(1), "b", Adj())
            & EmptyGraph(),
            Context(Adj(((None, Node(1)),)), Node(0), "b", Adj())
            & Context(Adj(), Node(1), "a", Adj())
            & EmptyGraph(),
            id="cyclic",
        ),
    ],
)
def test_eq(graph_a, graph_b):
    """Tests that equivalent graphs defined in multiple ways are equal."""
    assert graph_a == graph_b


def test_paper_example():
    graph: Graph[str, str] = (
        Context(
            Adj((("left", Node(2)), ("up", Node(3)))),
            Node(1),
            "a",
            Adj((("right", Node(2)),)),
        )
        & Context(Adj(), Node(2), "b", Adj((("down", Node(3)),)))
        & Context(Adj(), Node(3), "c", Adj())
        & EmptyGraph
    )
