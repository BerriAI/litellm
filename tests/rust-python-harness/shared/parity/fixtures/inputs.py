from __future__ import annotations

import queue
from typing import Final, TypeVar

from hypothesis import given, settings
from hypothesis.strategies import SearchStrategy

InputT = TypeVar("InputT")


def generate_case_inputs(strategy: SearchStrategy[InputT], examples: int) -> tuple[InputT, ...]:
    generated: Final[queue.SimpleQueue[InputT | None]] = queue.SimpleQueue()

    @settings(max_examples=examples, deadline=None, derandomize=True)
    @given(case_input=strategy)
    def generate_case(case_input: InputT) -> None:
        generated.put(case_input)

    generate_case()
    generated.put(None)
    return tuple(iter(generated.get, None))
