# Inductive Graph Algorithms

A Python implementation of the inductive graph algorithms described in [Erwig 2001](https://www.cambridge.org/core/journals/journal-of-functional-programming/article/inductive-graphs-and-functional-graph-algorithms/2210F7C31A34EA4CF5008ED9E7B4EF62).

Inductive graphs allow functional implementations of graph algorithms like topological sort and minimum spanning 
trees, both of which are implemented in this library.

This implementation uses pure Python and no third-party libraries (except for testing and development).

# Usage

The code below constructs the graph and calculates the minimum spanning tree shown in the [Wikipedia article](https://en.wikipedia.org/wiki/Minimum_spanning_tree) on 
minimum spanning trees, returning an `LRTree` as described in the paper.

```python
from graph.inductivegraph import Context, Adj, Node, EmptyGraph

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

graph.mst(Node(1))
```

# References

1. https://www.cambridge.org/core/journals/journal-of-functional-programming/article/inductive-graphs-and-functional-graph-algorithms/2210F7C31A34EA4CF5008ED9E7B4EF62
2. https://en.wikipedia.org/wiki/Minimum_spanning_tree
