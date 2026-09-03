from __future__ import annotations

import sys
import threading
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import CodeType, FrameType
from typing import Final


@dataclass(frozen=True, slots=True)
class FunctionTraceEvent:
    id: int
    parent_id: int | None
    function: str
    module_path: str | None = None
    file: str | None = None
    line: int | None = None

    @property
    def raw(self) -> str:
        location: Final = f"{self.file}:{self.line}" if self.file is not None and self.line is not None else ""
        qualified: Final = f"{self.module_path}::{self.function}" if self.module_path is not None else self.function
        return f"{location} {qualified}" if location else qualified


class PythonProfiler:
    def __init__(self, source_root: Path) -> None:
        self._source_root: Final = str(source_root.resolve()) + "/"
        self._seen_frames: Final[set[FrameType]] = set()
        self._event_ids: Final[dict[FrameType, int]] = {}
        self.events: Final[list[FunctionTraceEvent]] = []

    def __call__(self, frame: FrameType, event: str, _arg: object) -> None:
        if event != "call" or frame in self._seen_frames:
            return
        function_name: Final = self.function_name(frame.f_code)
        if function_name is None:
            return
        event_id: Final = len(self.events)
        parent_id: Final = next(
            (self._event_ids[ancestor] for ancestor in _frame_ancestors(frame) if ancestor in self._event_ids),
            None,
        )
        self._seen_frames.add(frame)
        self._event_ids[frame] = event_id
        self.events.append(FunctionTraceEvent(id=event_id, parent_id=parent_id, function=function_name))

    def function_name(self, code: CodeType) -> str | None:
        if not code.co_filename.startswith(self._source_root):
            return None
        relative: Final = code.co_filename.removeprefix(self._source_root)
        return f"{relative}:{code.co_firstlineno} {getattr(code, 'co_qualname', code.co_name)}"


class PythonFunctionUsageProfiler:
    def __init__(self, source_root: Path, functions: frozenset[str]) -> None:
        self._source_root: Final = str(source_root.resolve()) + "/"
        self._functions: Final = functions
        self.called: Final[set[str]] = set()

    def __call__(self, frame: FrameType, event: str, _arg: object) -> None:
        if event != "call":
            return
        code: Final = frame.f_code
        if not code.co_filename.startswith(self._source_root):
            return
        relative: Final = code.co_filename.removeprefix(self._source_root)
        function: Final = f"{relative}:{code.co_firstlineno} {getattr(code, 'co_qualname', code.co_name)}"
        if function in self._functions:
            self.called.add(function)


def _frame_ancestors(frame: FrameType) -> Generator[FrameType]:
    ancestor: Final = frame.f_back
    if ancestor is not None:
        yield ancestor
        yield from _frame_ancestors(ancestor)


@contextmanager
def profile_python(source_root: Path, *, threads: bool = False) -> Generator[PythonProfiler]:
    profiler: Final = PythonProfiler(source_root)
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


@contextmanager
def profile_python_function_usage(
    source_root: Path,
    functions: frozenset[str],
    *,
    threads: bool = False,
) -> Generator[PythonFunctionUsageProfiler]:
    profiler: Final = PythonFunctionUsageProfiler(source_root, functions)
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
