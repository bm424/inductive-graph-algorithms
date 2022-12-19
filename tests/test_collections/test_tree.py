import pytest

from graph.collections import Tree


@pytest.mark.parametrize(
    "tree_a, tree_b, expected",
    [
        (
            Tree(1, ()),
            Tree(1, ()),
            True,
        ),
        (
            Tree(1, (Tree(2, ()), Tree(3, ()))),
            Tree(1, (Tree(2, ()), Tree(3, ()))),
            True,
        ),
        (
            Tree(1, (Tree(2, (Tree(4, ()),)), Tree(3, ()))),
            Tree(1, (Tree(2, (Tree(4, ()),)), Tree(3, ()))),
            True,
        ),
        (
            Tree(1, (Tree(2, (Tree(4, ()),)), Tree(3, ()))),
            Tree(1, ()),
            False,
        ),
        (
            Tree(1, (Tree(2, (Tree(4, ()),)), Tree(3, ()))),
            Tree(1, (Tree(2, ()), Tree(3, ()))),
            False,
        ),
        (
            Tree(1, (Tree(2, ()), Tree(3, ()))),
            (1, (2, 3)),
            False,
        ),
    ],
)
def test_eq(tree_a: Tree, tree_b: Tree, expected: bool):
    assert (tree_a == tree_b) is expected


@pytest.mark.parametrize(
    "trees, expected_set",
    [
        ([Tree(1, ()), Tree(2, ())], {Tree(1, ()), Tree(2, ())}),
        ([Tree(1, ()), Tree(1, ())], {Tree(1, ())}),
        (
            [Tree(1, (Tree(2, ()),)), Tree(1, ())],
            {Tree(1, (Tree(2, ()),)), Tree(1, ())},
        ),
        (
            [Tree(1, (Tree(2, ()),)), Tree(1, (Tree(2, ()),))],
            {Tree(1, (Tree(2, ()),))},
        ),
    ],
)
def test_set(trees: list[Tree], expected_set: set[Tree]):
    assert set(trees) == expected_set


@pytest.mark.parametrize(
    "tree, expected_nodes",
    [
        (Tree(1, ()), (1,)),
        (Tree(1, (Tree(2, ()), Tree(3, ()))), (2, 3, 1)),
        (Tree(3, (Tree(2, ()), Tree(1, ()))), (2, 1, 3)),
        (Tree(1, (Tree(2, (Tree(3, ()),)), Tree(4, ()))), (3, 2, 4, 1)),
    ],
)
def test_postorder(tree: Tree[int], expected_nodes: tuple[int, ...]):
    assert tree.post_order() == expected_nodes


@pytest.mark.parametrize(
    "tree, fn, expected_tree",
    [
        (Tree("a", ()), "abcdefg".index, Tree(0, ())),
        (
            Tree("a", (Tree("b", ()), Tree("c", ()), Tree("d", ()))),
            "abcdefg".index,
            Tree(0, (Tree(1, ()), Tree(2, ()), Tree(3, ()))),
        ),
    ],
)
def test_tmap(tree, fn, expected_tree):
    assert tree.tmap(fn) == expected_tree
