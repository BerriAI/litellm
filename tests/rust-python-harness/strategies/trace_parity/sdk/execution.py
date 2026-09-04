from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from pathlib import Path
from typing import Final, Protocol, cast

from ....shared.parity.replay import replay_server
from ....shared.reporting.models import Surface
from ....shared.tracing.native import native_trace_events
from ....shared.tracing.profiler import FunctionTraceEvent, profile_python
from ....shared.tracing.steps import Engine, pipeline_projection
from ..models import RouteSpec, TraceExecutionFailure, TraceMode, TraceScenario
from ..reporting import TraceComparisonArtifact


class SdkCall(Protocol):
    def __call__(self, **kwargs: object) -> object: ...


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
        with replay_server() as provider:
            fixture: Final = spec.fixture(engine, provider.url)
            for response in fixture.provider_responses:
                provider.enqueue_response(response)
            kwargs: Final = {
                **fixture.kwargs,
                "api_key": "test-key",
                "api_base": provider.url,
                **({"timeout_seconds": 5} if engine == "rust" else {"timeout": 5}),
            }
            events: Final = _collect(function, kwargs, engine, asynchronous=asynchronous)
            provider.take_requests(len(fixture.provider_responses))
    except Exception as error:
        return TraceExecutionFailure(engine, f"{type(error).__name__}: {error}")
    if not events:
        return TraceExecutionFailure(engine, "trace is empty")
    return events


def _failure_message(result: tuple[FunctionTraceEvent, ...] | TraceExecutionFailure) -> str | None:
    if isinstance(result, tuple):
        return None
    return f"{result.engine}: {result.message}"


def execute_trace(
    route: RouteSpec, scenario: TraceScenario, mode: TraceMode, surface: Surface
) -> TraceComparisonArtifact:
    asynchronous: Final = mode == "async"
    mappings: Final = scenario.mappings_for(mode)
    scenario_route: Final = RouteSpec(
        route=route.route,
        python_entrypoints=route.python_entrypoints,
        rust_entrypoints=route.rust_entrypoints,
        fixture=scenario.fixture,
    )
    python_trace: Final = collect_trace(scenario_route, "python", asynchronous=asynchronous)
    rust_trace: Final = collect_trace(scenario_route, "rust", asynchronous=asynchronous)
    python_error: Final = _failure_message(python_trace)
    rust_error: Final = _failure_message(rust_trace)
    python_events: Final = python_trace if isinstance(python_trace, tuple) else ()
    rust_events: Final = rust_trace if isinstance(rust_trace, tuple) else ()
    try:
        python: Final = pipeline_projection("python", python_events, mappings)
        rust: Final = pipeline_projection("rust", rust_events, mappings)
    except ValueError as error:
        return TraceComparisonArtifact.from_traces(
            surface=surface,
            sdk_function=route.route,
            scenario=scenario.name,
            mode=mode,
            mappings=mappings,
            contract=scenario.contract,
            python=(),
            rust=(),
            python_unmatched=0,
            python_error=f"harness: {error}",
        )
    return TraceComparisonArtifact.from_traces(
        surface=surface,
        sdk_function=route.route,
        scenario=scenario.name,
        mode=mode,
        mappings=mappings,
        contract=scenario.contract,
        python=python.steps,
        rust=rust.steps,
        python_unmatched=python.unmatched,
        python_error=python_error,
        rust_error=rust_error,
    )
