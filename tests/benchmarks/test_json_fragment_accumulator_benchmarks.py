from collections.abc import Iterator
from typing import Final

import pytest

from litellm.litellm_core_utils.json_fragment_accumulator import JSONFragmentAccumulator

CONCATENATED_VALUES: Final = 20_000


def _popped_values(accumulator: JSONFragmentAccumulator) -> Iterator[object]:
    while True:
        found, value = accumulator.pop_next_value()
        if not found:
            return
        yield value


@pytest.mark.benchmark
def test_draining_many_concatenated_values() -> None:
    accumulator: Final = JSONFragmentAccumulator()
    accumulator.append('{"a": 1}' * CONCATENATED_VALUES)

    drained: Final = sum(1 for _ in _popped_values(accumulator))

    assert drained == CONCATENATED_VALUES
