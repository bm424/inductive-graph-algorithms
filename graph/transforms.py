"""Transformations between graphs and other data structures."""

import itertools
import typing
from functools import reduce

from graph.inductivegraph import Context, Adj, Node, EmptyGraph, Graph

T = typing.TypeVar("T")


class SupportsLessThan(typing.Protocol):
    def __lt__(self: T, other: T):
        ...  # pragma: no cover


A = typing.TypeVar("A", bound=SupportsLessThan)
B = typing.TypeVar("B")


def graph_from_mapping(
    mapping: typing.Mapping[A, typing.Sequence[A]]
) -> Graph[A, None]:
    """Construct a graph with labeled nodes from `mapping`.

    The keys of the mapping should be the labels of nodes where edges start, and the
    values should be the labels of nodes where the edges end.

    Example:
        >>> mapping = {"a": "bc", "b": "ac", "c": "d"}
        >>> graph_from_mapping(mapping)
    """
    node_labels = tuple(
        reversed(
            sorted(
                set(itertools.chain.from_iterable((k, *v) for k, v in mapping.items()))
            )
        )
    )
    nodes = (Node(i) for i in range(len(node_labels), 0, -1))
    mapping_inverted = _invert_mapping(mapping)

    def reducer(result: Graph[A, None], lnode: tuple[A, Node]) -> Graph[A, None]:
        label, node = lnode
        extant_label_nodes = result.label_nodes | {label: node}

        predecessors = Adj(
            tuple(
                (None, extant_label_nodes[label])
                for label in mapping_inverted.get(label, ())
                if label in extant_label_nodes
            )
        )
        successors = Adj(
            tuple(
                (None, extant_label_nodes[label])
                for label in mapping.get(label, ())
                if label in extant_label_nodes
            )
        )

        context = Context(predecessors, node, label, successors)
        return context & result

    return reduce(reducer, zip(node_labels, nodes), EmptyGraph())


def _invert_mapping(
    mapping: typing.Mapping[A, typing.Iterable[A]]
) -> dict[A, tuple[A, ...]]:
    def outer_reducer(
        result_outer: dict[A, tuple[A, ...]], item_outer: tuple[A, typing.Iterable[A]]
    ) -> dict[A, tuple[A, ...]]:
        def inner_reducer(
            result_inner: dict[A, tuple[A, ...]], item_inner: A
        ) -> dict[A, tuple[A, ...]]:
            return {
                **result_inner,
                item_inner: (*(result_inner.get(item_inner, ())), item_outer[0]),
            }

        return reduce(inner_reducer, item_outer[1], result_outer)

    return reduce(outer_reducer, mapping.items(), {})
