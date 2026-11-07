from collections.abc import Callable
from typing import Final, TypeVar

import pytest

LoopResult = TypeVar("LoopResult")


def runs_until_a_loop_address_repeats(
    run_on_new_loop: Callable[[], tuple[int, LoopResult]],
    earlier: tuple[tuple[int, LoopResult], ...] = (),
    limit: int = 50,
) -> tuple[tuple[int, LoopResult], ...]:
    if len(earlier) == limit:
        pytest.skip(f"none of {limit} collected loops had its address handed to a later loop")
    latest: Final = run_on_new_loop()
    runs: Final = (*earlier, latest)
    if latest[0] in {loop_id for loop_id, _ in earlier}:
        return runs
    return runs_until_a_loop_address_repeats(run_on_new_loop, runs, limit)
