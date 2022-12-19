import pytest

from graph.inductivegraph import Graph, Node
from graph.transforms import graph_from_mapping


def _make_graph(g) -> Graph:
    match g:
        case Graph():
            return g
        case dict():
            return graph_from_mapping(g)


@pytest.fixture
def graph(request):
    return _make_graph(request.param)


@pytest.fixture
def expected(request):
    return _make_graph(request.param)


@pytest.fixture
def node(request, graph):
    match request.param:
        case Node() as node:
            return node
        case label:
            return graph.label_nodes[label]
