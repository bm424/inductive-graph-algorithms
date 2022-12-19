import pytest

from graph.tools import concat_map, first, nub


@pytest.mark.parametrize(
    "fn, items, expected_items",
    [
        (range, [1, 2, 3], (0, 0, 1, 0, 1, 2)),
        (tuple, ["abc", "def"], ("a", "b", "c", "d", "e", "f")),
    ],
)
def test_concat_map(fn, items, expected_items):
    assert concat_map(fn, items) == expected_items


@pytest.mark.parametrize(
    "predicate, items, expected_item",
    [
        (bool, [0, 0, 0, 1, 2, 3], 1),
        (lambda t: t == 2, [0, 0, 0, 1, 2, 3], 2),
        (lambda t: t == 4, [0, 0, 0, 1, 2, 3], None),
    ],
)
def test_first(predicate, items, expected_item):
    assert first(predicate, items) == expected_item


@pytest.mark.parametrize(
    "items, expected_items",
    [
        ([1, 2, 3, 4], (1, 2, 3, 4)),
        ([4, 3, 2, 1], (4, 3, 2, 1)),
        ([1, 2, 1, 3], (1, 2, 3)),
        ([2, 1, 2, 3], (2, 1, 3)),
    ],
)
def test_nub(items, expected_items):
    assert nub(items) == expected_items
