from __future__ import annotations

import sys
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from types import FrameType, FunctionType
from typing import Final


@dataclass(frozen=True, slots=True)
class FunctionTraceEvent:
    function: str
    depth: int


class PythonProfiler:
    def __init__(self, functions: Sequence[FunctionType]) -> None:
        self._names_by_code: Final = {function.__code__: function.__name__ for function in functions}
        self._seen_frames: Final[set[FrameType]] = set()
        self.events: Final[list[FunctionTraceEvent]] = []

    def __call__(self, frame: FrameType, event: str, _arg: object) -> None:
        if event != "call" or frame in self._seen_frames:
            return
        function_name: Final = self._names_by_code.get(frame.f_code)
        if function_name is None:
            return
        depth: Final = sum(ancestor.f_code in self._names_by_code for ancestor in _frame_ancestors(frame))
        self._seen_frames.add(frame)
        self.events.append(FunctionTraceEvent(function=function_name, depth=depth))


def _frame_ancestors(frame: FrameType) -> Generator[FrameType]:
    ancestor: Final = frame.f_back
    if ancestor is not None:
        yield ancestor
        yield from _frame_ancestors(ancestor)


@contextmanager
def profile_python(functions: Sequence[FunctionType]) -> Generator[PythonProfiler]:
    profiler: Final = PythonProfiler(functions)
    previous: Final = sys.getprofile()
    sys.setprofile(profiler)
    try:
        yield profiler
    finally:
        sys.setprofile(previous)
