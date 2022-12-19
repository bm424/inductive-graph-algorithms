"""Functional tools."""
import itertools
import typing

A = typing.TypeVar("A")
B = typing.TypeVar("B")
C = typing.TypeVar("C")


def concat_map(
    fn: typing.Callable[[A], typing.Sequence[B]], items: typing.Iterable[A]
) -> tuple[B, ...]:
    return tuple(itertools.chain.from_iterable(map(fn, items)))


def first(predicate: typing.Callable[[A], bool], items: typing.Iterable[A]) -> A | None:
    return next((item for item in items if predicate(item)), None)


def nub(items: typing.Sequence[A]) -> tuple[A, ...]:
    if not items:
        return ()
    item, *remainder = items
    return item, *nub(tuple(filter(lambda x: x != item, remainder)))
