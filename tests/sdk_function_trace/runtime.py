from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from pydantic import BaseModel, ConfigDict

from tests.sdk_function_trace.fixtures import Invocation, sdk_invocation
from tests.sdk_function_trace.mock_provider import mock_provider
from tests.sdk_function_trace.profiler import FunctionTraceEvent, profile_python
from tests.sdk_function_trace.steps import Engine


class TraceEventPayload(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    function: str
    depth: int


class TraceResponsePayload(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    response: object
    trace: tuple[TraceEventPayload, ...] | list[TraceEventPayload]


@contextmanager
def _python_engine() -> Generator[None]:
    from litellm.rust_bridge import ocr as ocr_bridge

    previous_ocr: Final = ocr_bridge.rust_ocr_enabled()
    previous_rust: Final = os.environ.get("LITELLM_RUST")
    os.environ["LITELLM_RUST"] = "false"
    ocr_bridge.use_litellm_rust(False)
    try:
        yield
    finally:
        ocr_bridge.use_litellm_rust(previous_ocr)
        if previous_rust is None:
            os.environ.pop("LITELLM_RUST", None)
        else:
            os.environ["LITELLM_RUST"] = previous_rust


def _invoke(case: Invocation, api_base: str, *, asynchronous: bool) -> object:
    async def invoke_async() -> object:
        return await cast("Awaitable[object]", case.function(**case.kwargs, api_base=api_base))

    if asynchronous:
        return asyncio.run(invoke_async())
    return case.function(**case.kwargs, api_base=api_base)


def collect(case: Invocation, api_base: str, *, engine: Engine, asynchronous: bool) -> tuple[FunctionTraceEvent, ...]:
    import litellm

    if engine == "rust":
        payload: Final = TraceResponsePayload.model_validate(_invoke(case, api_base, asynchronous=asynchronous))
        return tuple(FunctionTraceEvent(event.function, event.depth) for event in payload.trace)
    with profile_python(source_root=Path(litellm.__file__).parent, threads=True) as profiler:
        _invoke(case, api_base, asynchronous=asynchronous)
    return tuple(profiler.events)


def run_trace(route: str, *, engine: Engine, asynchronous: bool = False) -> tuple[FunctionTraceEvent, ...]:
    case: Final = sdk_invocation(route, engine=engine, asynchronous=asynchronous)
    with _python_engine(), mock_provider(case.provider_response) as api_base:
        events: Final = collect(case, api_base, engine=engine, asynchronous=asynchronous)
    if not events:
        raise RuntimeError(f"No runtime events for {route}; rebuild the native extension with tracing support")
    return events


@dataclass(frozen=True, slots=True)
class TraceOk:
    events: tuple[FunctionTraceEvent, ...]


@dataclass(frozen=True, slots=True)
class TraceSkipped:
    reason: str


@dataclass(frozen=True, slots=True)
class TraceFailed:
    reason: str


TraceRun = TraceOk | TraceSkipped | TraceFailed


def attempt_trace(route: str, *, engine: Engine, asynchronous: bool) -> TraceRun:
    try:
        return TraceOk(run_trace(route, engine=engine, asynchronous=asynchronous))
    except Exception as error:
        reason: Final = f"{type(error).__name__}: {error}"
        if (
            route == "messages"
            and engine == "python"
            and not asynchronous
            and isinstance(error, ValueError)
            and str(error) == "anthropic_messages_handler is not implemented for sync calls"
        ):
            return TraceSkipped(reason)
        return TraceFailed(reason)


@dataclass(frozen=True, slots=True)
class TraceDiff:
    python_only: tuple[str, ...]
    rust_only: tuple[str, ...]
    shared_order_matches: bool

    @property
    def matches(self) -> bool:
        return not self.python_only and not self.rust_only and self.shared_order_matches


def trace_diff(python: tuple[FunctionTraceEvent, ...], rust: tuple[FunctionTraceEvent, ...]) -> TraceDiff:
    python_names: Final = {event.function for event in python}
    rust_names: Final = {event.function for event in rust}
    shared_python: Final = tuple(event.function for event in python if event.function in rust_names)
    shared_rust: Final = tuple(event.function for event in rust if event.function in python_names)
    return TraceDiff(
        python_only=tuple(event.function for event in python if event.function not in rust_names),
        rust_only=tuple(event.function for event in rust if event.function not in python_names),
        shared_order_matches=bool(shared_python) and shared_python == shared_rust,
    )
