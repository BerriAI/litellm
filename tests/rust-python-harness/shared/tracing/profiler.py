from __future__ import annotations

import sys
import threading
from collections.abc import Generator, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import CodeType, FrameType, FunctionType, MappingProxyType
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
        function_name: Final = self.function_name(frame)
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

    def function_name(self, frame: FrameType) -> str | None:
        code: Final = frame.f_code
        if not code.co_filename.startswith(self._source_root):
            return None
        relative: Final = code.co_filename.removeprefix(self._source_root)
        return f"{relative}:{code.co_firstlineno} {_qualified_name(frame)}"


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
        function: Final = f"{relative}:{code.co_firstlineno} {_qualified_name(frame)}"
        if function in self._functions:
            self.called.add(function)


def _qualified_name(frame: FrameType) -> str:
    code: Final = frame.f_code
    native: Final = getattr(code, "co_qualname", None)
    if isinstance(native, str):
        return native
    enclosing: Final = next(
        (
            name
            for ancestor in _frame_ancestors(frame)
            for declared_code, name in _declared_functions(ancestor.f_locals, frozenset())
            if declared_code is code
        ),
        None,
    )
    if enclosing is not None:
        return enclosing
    module_name: Final = frame.f_globals.get("__name__")
    if not isinstance(module_name, str):
        return code.co_name
    return _module_qualnames(module_name).get(code, code.co_name)


@lru_cache(maxsize=None)
def _module_qualnames(module_name: str) -> Mapping[CodeType, str]:
    module: Final = sys.modules.get(module_name)
    if module is None:
        return MappingProxyType({})
    return MappingProxyType(dict(_declared_functions(vars(module), frozenset())))


def _declared_functions(namespace: Mapping[str, object], visited: frozenset[int]) -> Iterator[tuple[CodeType, str]]:
    for attribute in tuple(namespace.values()):
        for value in _accessors(attribute):
            if isinstance(value, FunctionType):
                yield from ((wrapped.__code__, wrapped.__qualname__) for wrapped in _unwrapped(value))
            elif isinstance(value, type) and id(value) not in visited:
                yield from _declared_functions(dict(vars(value)), visited | {id(value)})


def _unwrapped(function: FunctionType) -> Iterator[FunctionType]:
    yield function
    inner: Final = getattr(function, "__wrapped__", None)
    if isinstance(inner, FunctionType):
        yield from _unwrapped(inner)


def _accessors(value: object) -> tuple[object, ...]:
    if isinstance(value, (staticmethod, classmethod)):
        return (value.__func__,)
    if isinstance(value, property):
        return tuple(accessor for accessor in (value.fget, value.fset, value.fdel) if accessor is not None)
    return (value,)


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
