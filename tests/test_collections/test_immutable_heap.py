import pytest

from graph.collections import ImmutableHeap


@pytest.fixture
def heap(request):
    return ImmutableHeap.from_iterable(request.param)


@pytest.mark.parametrize(
    "items, expected_heap",
    [
        ((1, 2, 3, 4), [1, 2, 3, 4]),
        ((4, 3, 2, 1), [1, 3, 2, 4]),
    ],
)
def test_from_iterable(items, expected_heap):
    heap = ImmutableHeap.from_iterable(items)
    assert heap.heap == expected_heap


def test_empty():
    assert ImmutableHeap.empty().heap == []


@pytest.mark.parametrize(
    "item",
    [
        1,
        (1, 2, 3),
        [4, 3, 2, 1],
    ],
)
def test_unit(item):
    assert ImmutableHeap.unit(item).heap == [item]


@pytest.mark.parametrize(
    "heap, expected",
    [
        ((1, 2, 3), True),
        ((), False),
    ],
    indirect=["heap"],
)
def test_bool(heap, expected):
    assert bool(heap) is expected


@pytest.mark.parametrize(
    "heap, item, expected_heap",
    [
        ([4, 3, 2], 1, [1, 2, 4, 3]),
        ([1, 2, 3], 4, [1, 2, 3, 4]),
    ],
    indirect=["heap"],
)
def test_push(heap, item, expected_heap):
    result = heap.push(item)
    assert result.heap == expected_heap
    assert heap is not result


@pytest.mark.parametrize(
    "heap, expected_item",
    [
        ([4, 3, 2, 1], 1),
        ([1, 2, 3, 4], 1),
    ],
    indirect=["heap"],
)
def test_pop(heap, expected_item):
    result, remainder = heap.pop()
    assert result == expected_item


@pytest.mark.parametrize(
    "heap_1, heap_2, expected_heap",
    [
        ([1, 3, 5], [2, 4], [1, 2, 3, 4, 5]),
    ],
)
def test_merge(heap_1, heap_2, expected_heap):
    heap_1 = ImmutableHeap.from_iterable(heap_1)
    heap_2 = ImmutableHeap.from_iterable(heap_2)
    result = ImmutableHeap.merge(heap_1, heap_2)
    assert result.heap == expected_heap
