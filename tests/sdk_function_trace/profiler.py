from __future__ import annotations

import sys
import threading
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


class PythonProfiler:
    def __init__(self, functions: Sequence[FunctionType], source_root: Path | None = None) -> None:
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
        depth: Final = sum(self.function_name(ancestor.f_code) is not None for ancestor in _frame_ancestors(frame))
        self._seen_frames.add(frame)
        self.events.append(FunctionTraceEvent(function=function_name, depth=depth))

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
def profile_python(
    functions: Sequence[FunctionType] = (), *, source_root: Path | None = None, threads: bool = False
) -> Generator[PythonProfiler]:
    profiler: Final = PythonProfiler(functions, source_root)
    previous_thread: Final = threading.getprofile()
    if threads:
        threading.setprofile(profiler)
    previous: Final = sys.getprofile()
    sys.setprofile(profiler)
    try:
        yield profiler
    finally:
        sys.setprofile(previous)
        if threads:
            threading.setprofile(previous_thread)
