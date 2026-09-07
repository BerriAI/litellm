from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from hypothesis import find, settings
from hypothesis.strategies import SearchStrategy

FixtureT = TypeVar("FixtureT")
FIND_SETTINGS = settings(max_examples=2_000, deadline=None, derandomize=True, database=None)


def find_fixture(
    strategy: SearchStrategy[FixtureT],
    predicate: Callable[[FixtureT], bool],
) -> FixtureT:
    return find(strategy, predicate, settings=FIND_SETTINGS)
