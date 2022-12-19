"""Tests for basic graph functions."""
import pytest

from graph.inductivegraph import Context, Adj, Node, EmptyGraph


@pytest.mark.parametrize(
    "graph, fn, initial, expected",
    [
        ({}, lambda _c, r: r + 1, 0, 0),
        ({"a": "a"}, lambda _c, r: r + 1, 0, 1),
        ({"a": "b"}, lambda _c, r: r + 1, 0, 2),
        ({"a": "b", "b": "a"}, lambda _c, r: r + 1, 0, 2),
        ({"a": "b", "b": "c"}, lambda _c, r: r + 1, 0, 3),
    ],
    indirect=["graph"],
)
def test_ufold(graph, fn, initial, expected):
    assert graph.ufold(fn, initial) == expected


@pytest.mark.parametrize(
    "graph, fn, expected",
    [
        ({}, Context.label_mapper("abdefg".index), {}),
        ({"a": "b"}, Context.label_mapper("abcdefg".index), {0: (1,)}),
    ],
    indirect=["graph", "expected"],
)
def test_gmap(graph, fn, expected):
    assert graph.gmap(fn) == expected


@pytest.mark.parametrize(
    "graph, node, expected_context, expected",
    [
        (
            {"a": "b"},
            "a",
            Context(Adj(), Node(1), "a", Adj(((None, Node(2)),))),
            Context(Adj(), Node(2), "b", Adj()) & EmptyGraph(),
        ),
        (
            {"a": "bc", "b": "a"},
            "a",
            Context(
                Adj(((None, Node(2)),)),
                Node(1),
                "a",
                Adj(((None, Node(2)), (None, Node(3)))),
            ),
            Context(Adj(), Node(2), "b", Adj())
            & Context(Adj(), Node(3), "c", Adj())
            & EmptyGraph(),
        ),
        (
            {"a": "bc", "b": "a"},
            "b",
            Context(
                Adj(((None, Node(1)),)),
                Node(2),
                None,
                Adj(((None, Node(1)),)),
            ),
            Context(Adj(), Node(1), "a", Adj(((None, Node(3)),)))
            & Context(Adj(), Node(3), "c", Adj())
            & EmptyGraph(),
        ),
    ],
    indirect=["graph", "node", "expected"],
)
def test_pop(graph, node, expected_context, expected):
    assert graph.pop(node) == (expected_context, expected)
