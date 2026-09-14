from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterable, Awaitable, Iterable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, cast
from unittest.mock import patch

from ....shared.parity.replay import replay_server
from ....shared.reporting.models import Surface
from ....shared.tracing.native import TraceResponsePayload, native_trace_events
from ....shared.tracing.profiler import FunctionTraceEvent, profile_python
from ....shared.tracing.steps import Engine, pipeline_projection
from ..models import RouteFixture, RouteSpec, TraceEngine, TraceExecutionFailure, TraceScenario
from ..reporting import TraceArtifact


class SdkCall(Protocol):
    def __call__(self, **kwargs: object) -> object: ...


@dataclass(frozen=True, slots=True)
class _CollectedTrace:
    events: tuple[FunctionTraceEvent, ...]
    error: str | None = None


def _invoke(
    function: SdkCall,
    kwargs: dict[str, object],
    *,
    asynchronous: bool,
    consume_stream: bool = False,
) -> object:
    async def invoke_async() -> object:
        try:
            response: Final = await cast(Awaitable[object], function(**kwargs))
            if consume_stream and isinstance(response, AsyncIterable):
                stream = cast(AsyncIterable[object], response)
                return tuple([item async for item in stream])
            return response
        finally:
            await asyncio.sleep(0)
            from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

            await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10)
            await GLOBAL_LOGGING_WORKER.stop()

    if asynchronous:
        return asyncio.run(invoke_async())
    response: Final = function(**kwargs)
    if consume_stream and isinstance(response, Iterable):
        return tuple(cast(Iterable[object], response))
    return response


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
    function: SdkCall,
    fixture: RouteFixture,
    engine: Engine,
    *,
    asynchronous: bool,
) -> _CollectedTrace:
    kwargs: Final = fixture.kwargs
    if engine == "rust":
        payload: Final = TraceResponsePayload.model_validate(_invoke(function, kwargs, asynchronous=asynchronous))
        return _CollectedTrace(native_trace_events(payload), payload.error)
    import litellm

    previous_suppress_debug_info: Final = litellm.suppress_debug_info
    try:
        if fixture.expected_failure:
            litellm.suppress_debug_info = True
        with profile_python(Path(litellm.__file__).parent, threads=True) as profiler:
            error: str | None
            try:
                _invoke(function, kwargs, asynchronous=asynchronous, consume_stream=fixture.consume_stream)
                error = None
            except Exception as caught:
                error = f"{type(caught).__name__}: {caught}"
    finally:
        litellm.suppress_debug_info = previous_suppress_debug_info
    return _CollectedTrace(tuple(profiler.events), error)


def collect_trace(
    spec: RouteSpec, engine: Engine, *, asynchronous: bool, python_rust_enabled: bool = False
) -> tuple[FunctionTraceEvent, ...] | TraceExecutionFailure:
    function: Final = _entrypoint(spec, engine, asynchronous=asynchronous)
    if isinstance(function, TraceExecutionFailure):
        return function
    try:
        with replay_server() as provider:
            base_fixture: Final = spec.fixture(engine, provider.url)
            for response in base_fixture.provider_responses:
                provider.enqueue_response(response)
            fixture: Final = RouteFixture(
                kwargs={
                    **base_fixture.kwargs,
                    "api_key": "test-key",
                    "api_base": provider.url,
                    **({"timeout_seconds": 5} if engine == "rust" else {"timeout": 5}),
                },
                provider_responses=base_fixture.provider_responses,
                expected_failure=base_fixture.expected_failure,
                consume_stream=base_fixture.consume_stream,
            )
            environment: Final = (
                patch.dict(os.environ, {"LITELLM_RUST": "1" if python_rust_enabled else "0"})
                if engine == "python"
                else nullcontext()
            )
            with environment:
                collected: Final = _collect(function, fixture, engine, asynchronous=asynchronous)
            provider.take_requests(len(fixture.provider_responses))
    except Exception as error:
        return TraceExecutionFailure(engine, f"{type(error).__name__}: {error}")
    if fixture.expected_failure and collected.error is None:
        return TraceExecutionFailure(engine, "call succeeded but the scenario expects failure")
    if not fixture.expected_failure and collected.error is not None:
        return TraceExecutionFailure(engine, collected.error)
    if not collected.events:
        return TraceExecutionFailure(engine, "trace is empty")
    return collected.events


def _failure_message(result: tuple[FunctionTraceEvent, ...] | TraceExecutionFailure) -> str | None:
    if isinstance(result, tuple):
        return None
    return f"{result.engine}: {result.message}"


def execute_trace(
    route: RouteSpec,
    scenario: TraceScenario,
    surface: Surface,
    engine: TraceEngine = "both",
) -> TraceArtifact:
    mappings: Final = scenario.mappings
    scenario_route: Final = RouteSpec(
        route=route.route,
        python_entrypoints=route.python_entrypoints,
        rust_entrypoints=route.rust_entrypoints,
        fixture=scenario.fixture,
    )
    python_trace: Final = (
        collect_trace(
            scenario_route,
            "python",
            asynchronous=scenario.asynchronous,
            python_rust_enabled=scenario.python_rust_enabled,
        )
        if engine != "rust"
        else ()
    )
    rust_trace: Final = (
        collect_trace(scenario_route, "rust", asynchronous=scenario.asynchronous) if engine != "python" else ()
    )
    python_error: Final = _failure_message(python_trace)
    rust_error: Final = _failure_message(rust_trace)
    python_events: Final = python_trace if isinstance(python_trace, tuple) else ()
    rust_events: Final = rust_trace if isinstance(rust_trace, tuple) else ()
    try:
        python: Final = pipeline_projection("python", python_events, mappings)
        rust: Final = pipeline_projection("rust", rust_events, mappings)
    except ValueError as error:
        return TraceArtifact.from_traces(
            engine=engine,
            surface=surface,
            sdk_function=route.route,
            scenario=scenario.name,
            python=(),
            rust=(),
            python_error=f"harness: {error}",
        )
    return TraceArtifact.from_traces(
        engine=engine,
        surface=surface,
        sdk_function=route.route,
        scenario=scenario.name,
        python=python.steps,
        rust=rust.steps,
        python_error=python_error,
        rust_error=rust_error,
    )
