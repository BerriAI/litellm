"""Gateway endpoint profiler - adapted from tests/sdk_function_trace/profiler.py"""
from __future__ import annotations

import sys
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import CodeType, FrameType, FunctionType
from typing import Final


@dataclass(frozen=True, slots=True)
class FunctionTraceEvent:
    function: str
    depth: int


class GatewayProfiler:
    """Profiles gateway endpoint execution using sys.setprofile."""

    def __init__(self, functions: Sequence[FunctionType], source_root: Path | None = None):
        self._source_root: Final = str(source_root.resolve()) + "/" if source_root is not None else None
        self._names_by_code: Final = {function.__code__: function.__name__ for function in functions}
        self._seen_frames: Final[set[FrameType]] = set()
        self.events: Final[list[FunctionTraceEvent]] = []

    def __call__(self, frame: FrameType, event: str, _arg: object) -> None:
        if event != "call" or frame in self._seen_frames:
            return
        function_name: Final = self.function_name(frame.f_code)
        if function_name is None:
            return
        ancestors: Final = tuple(
            name for ancestor in _frame_ancestors(frame) if (name := self.function_name(ancestor.f_code)) is not None
        )
        self._seen_frames.add(frame)
        self.events.append(FunctionTraceEvent(function=function_name, depth=len(ancestors)))

    def function_name(self, code: CodeType) -> str | None:
        if self._source_root is None:
            return self._names_by_code.get(code)
        if not code.co_filename.startswith(self._source_root):
            return None
        relative: Final = code.co_filename.removeprefix(self._source_root)
        return f"{relative}:{code.co_firstlineno} {getattr(code, 'co_qualname', code.co_name)}"


def _frame_ancestors(frame: FrameType) -> Generator[FrameType]:
    ancestor: Final = frame.f_back
    if ancestor is not None:
        yield ancestor
        yield from _frame_ancestors(ancestor)


@contextmanager
def profile_gateway(functions: Sequence[FunctionType] = (), *, source_root: Path | None = None) -> Generator[GatewayProfiler]:
    """Profile gateway endpoint execution."""
    profiler: Final = GatewayProfiler(functions, source_root)
    previous: Final = sys.getprofile()
    sys.setprofile(profiler)
    try:
        yield profiler
    finally:
        sys.setprofile(previous)
