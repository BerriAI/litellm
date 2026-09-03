from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Protocol, cast

from ....shared.parity.recorded_http import RecordedHttpResponse
from ....shared.parity.replay import replay_server
from ....shared.reporting.models import SdkFunction, Surface
from ....shared.tracing.native import native_trace_events
from ....shared.tracing.profiler import FunctionTraceEvent, profile_python
from ....shared.tracing.steps import Engine, Step, pipeline_issues, pipeline_steps
from ..reporting import TraceComparisonArtifact

TraceMode = Literal["sync", "async"]
TraceFailureSource = Literal["python", "rust", "harness"]


class SdkCall(Protocol):
    def __call__(self, **kwargs: object) -> object: ...


@dataclass(frozen=True, slots=True)
class RouteFixture:
    kwargs: dict[str, object]
    provider_response: RecordedHttpResponse


@dataclass(frozen=True, slots=True)
class RouteSpec:
    route: SdkFunction
    python_entrypoints: tuple[str, str]
    rust_entrypoints: tuple[str, str]
    fixture: Callable[[Engine], RouteFixture]


@dataclass(frozen=True, slots=True)
class TraceCase:
    route: RouteSpec
    steps: tuple[Step, ...]
    edges: tuple[tuple[str, str], ...]
    modes: tuple[TraceMode, ...] = ("sync", "async")
    matching_steps: bool = True
    exact: bool = False


@dataclass(frozen=True, slots=True)
class TraceExecutionFailure:
    engine: TraceFailureSource
    message: str


def _invoke(function: SdkCall, kwargs: dict[str, object], *, asynchronous: bool) -> object:
    async def invoke_async() -> object:
        return await cast(Awaitable[object], function(**kwargs))

    if asynchronous:
        return asyncio.run(invoke_async())
    return function(**kwargs)


def _entrypoint(spec: RouteSpec, engine: Engine, *, asynchronous: bool) -> SdkCall | TraceExecutionFailure:
    import litellm
    from litellm.anthropic_interface import messages as sdk_messages
    from litellm.rust_bridge import get_native_bridge

    if engine == "rust":
        bridge: Final = cast(object | None, get_native_bridge())
        if bridge is None:
            return TraceExecutionFailure("rust", "native Rust bridge is required for trace parity")
        trace_bridge: Final[object | None] = getattr(bridge, "_trace", None)
        if trace_bridge is None:
            return TraceExecutionFailure("rust", "native Rust bridge must include the trace-parity feature")
        entrypoint: Final = spec.rust_entrypoints[int(asynchronous)]
        function: Final[object | None] = getattr(trace_bridge, entrypoint, None)
        if function is None:
            return TraceExecutionFailure("rust", f"native Rust trace bridge does not expose {entrypoint}")
        return cast(SdkCall, function)
    owner: Final = sdk_messages if spec.route == "messages" else litellm
    return cast(SdkCall, getattr(owner, spec.python_entrypoints[int(asynchronous)]))


def _collect(
    function: SdkCall, kwargs: dict[str, object], engine: Engine, *, asynchronous: bool
) -> tuple[FunctionTraceEvent, ...]:
    if engine == "rust":
        return native_trace_events(_invoke(function, kwargs, asynchronous=asynchronous))
    import litellm

    with profile_python(Path(litellm.__file__).parent, threads=True) as profiler:
        _invoke(function, kwargs, asynchronous=asynchronous)
    return tuple(profiler.events)


def collect_trace(
    spec: RouteSpec, engine: Engine, *, asynchronous: bool
) -> tuple[FunctionTraceEvent, ...] | TraceExecutionFailure:
    function: Final = _entrypoint(spec, engine, asynchronous=asynchronous)
    if isinstance(function, TraceExecutionFailure):
        return function
    try:
        fixture: Final = spec.fixture(engine)
        with replay_server() as provider:
            provider.enqueue_response(fixture.provider_response)
            kwargs: Final = {
                **fixture.kwargs,
                "api_key": "test-key",
                "api_base": provider.url,
                **({"timeout_seconds": 5} if engine == "rust" else {"timeout": 5}),
            }
            events: Final = _collect(function, kwargs, engine, asynchronous=asynchronous)
            provider.take_requests(1)
    except Exception as error:
        return TraceExecutionFailure(engine, f"{type(error).__name__}: {error}")
    if not events:
        return TraceExecutionFailure(engine, "trace is empty")
    return events


def execute_trace(
    case: TraceCase, mode: TraceMode, surface: Surface
) -> TraceComparisonArtifact | TraceExecutionFailure:
    asynchronous: Final = mode == "async"
    python_trace: Final = collect_trace(case.route, "python", asynchronous=asynchronous)
    if isinstance(python_trace, TraceExecutionFailure):
        return python_trace
    rust_trace: Final = collect_trace(case.route, "rust", asynchronous=asynchronous)
    if isinstance(rust_trace, TraceExecutionFailure):
        return rust_trace
    python: Final = pipeline_steps("python", python_trace, case.steps)
    rust: Final = pipeline_steps("rust", rust_trace, case.steps)
    return TraceComparisonArtifact.from_traces(
        surface=surface,
        sdk_function=case.route.route,
        mode=mode,
        python=python,
        rust=rust,
        python_issues=pipeline_issues("python", python, case.steps, case.edges),
        rust_issues=pipeline_issues("rust", rust, case.steps, case.edges),
        requires_matching_steps=case.matching_steps,
        requires_exact_trace=case.exact,
    )
