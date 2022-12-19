import collections.abc
import typing

import pytest as pytest

from graph.inductivegraph import EmptyGraph, Context, Node, Adj, Graph
from graph.transforms import graph_from_mapping


@pytest.mark.parametrize(
    "mapping, graph",
    [
        pytest.param({}, EmptyGraph(), id="empty"),
        pytest.param(
            {"a": {"b", "c"}},
            Context(
                Adj(),
                Node(0),
                "a",
                Adj(((None, Node(1)), (None, Node(2)))),
            )
            & Context(Adj(), Node(1), "b", Adj())
            & Context(Adj(), Node(2), "c", Adj())
            & EmptyGraph(),
            id="branching",
        ),
        pytest.param(
            {"a": {"b"}, "b": {"c"}, "c": {"a"}},
            Context(
                Adj(((None, Node(0)),)),
                Node(1),
                "a",
                Adj(((None, Node(2)),)),
            )
            & Context(Adj(), Node(2), "b", Adj(((None, Node(0)),)))
            & Context(Adj(), Node(0), "c", Adj())
            & EmptyGraph(),
            id="cyclic",
        ),
        pytest.param(
            {"a": "a"},
            Context(Adj(((None, Node(0)),)), Node(0), "a", Adj()) & EmptyGraph(),
            id="loop",
        ),
        pytest.param(
            {"a": "b", "c": "d"},
            Context(Adj(((None, Node(0)),)), Node(1), "b", Adj())
            & Context(Adj(), Node(0), "a", Adj())
            & Context(Adj(), Node(2), "c", Adj(((None, Node(3)),)))
            & Context(Adj(), Node(3), "d", Adj())
            & EmptyGraph(),
            id="disconnected",
        ),
    ],
)
def test_graph_from_mapping(
    mapping: collections.abc.Mapping[str, typing.Sequence[str]], graph: Graph
):
    """Tests that graphs constructed from mappings are correct."""
    assert graph_from_mapping(mapping) == graph
