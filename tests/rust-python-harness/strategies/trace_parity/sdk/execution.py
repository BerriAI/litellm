from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterable, Awaitable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, cast
from unittest.mock import patch

from ....shared.parity.replay import replay_server
from ....shared.reporting.models import Surface
from ....shared.tracing.profiler import FunctionTraceEvent, profile_python
from ....shared.tracing.steps import pipeline_projection
from ..models import RouteFixture, RouteSpec, TraceExecutionFailure, TraceScenario
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


def _entrypoint(spec: RouteSpec, *, asynchronous: bool) -> SdkCall:
    import litellm
    from litellm.anthropic_interface import messages as sdk_messages

    owner: Final = sdk_messages if spec.route == "messages" else litellm
    return cast(SdkCall, getattr(owner, spec.python_entrypoints[int(asynchronous)]))


def _collect(
    function: SdkCall,
    fixture: RouteFixture,
    *,
    asynchronous: bool,
) -> _CollectedTrace:
    import litellm

    previous_suppress_debug_info: Final = litellm.suppress_debug_info
    try:
        if fixture.expected_failure:
            litellm.suppress_debug_info = True
        with profile_python(Path(litellm.__file__).parent, threads=True) as profiler:
            error: str | None
            try:
                _invoke(function, fixture.kwargs, asynchronous=asynchronous, consume_stream=fixture.consume_stream)
                error = None
            except Exception as caught:
                error = f"{type(caught).__name__}: {caught}"
    finally:
        litellm.suppress_debug_info = previous_suppress_debug_info
    return _CollectedTrace(tuple(profiler.events), error)


def collect_trace(spec: RouteSpec, *, asynchronous: bool) -> tuple[FunctionTraceEvent, ...] | TraceExecutionFailure:
    function: Final = _entrypoint(spec, asynchronous=asynchronous)
    try:
        with replay_server() as provider:
            base_fixture: Final = spec.fixture(provider.url)
            for response in base_fixture.provider_responses:
                provider.enqueue_response(response)
            fixture: Final = RouteFixture(
                kwargs={
                    "api_key": "test-key",
                    **base_fixture.kwargs,
                    "api_base": provider.url,
                    "timeout": 5,
                },
                provider_responses=base_fixture.provider_responses,
                expected_failure=base_fixture.expected_failure,
                consume_stream=base_fixture.consume_stream,
                environment=base_fixture.environment,
            )
            with patch.dict(os.environ, fixture.environment):
                collected: Final = _collect(function, fixture, asynchronous=asynchronous)
            provider.take_requests(len(fixture.provider_responses))
    except Exception as error:
        return TraceExecutionFailure("python", f"{type(error).__name__}: {error}")
    if fixture.expected_failure and collected.error is None:
        return TraceExecutionFailure("python", "call succeeded but the scenario expects failure")
    if not fixture.expected_failure and collected.error is not None:
        return TraceExecutionFailure("python", collected.error)
    if not collected.events:
        return TraceExecutionFailure("python", "trace is empty")
    return collected.events


def execute_trace(route: RouteSpec, scenario: TraceScenario, surface: Surface) -> TraceArtifact:
    scenario_route: Final = RouteSpec(
        route=route.route,
        python_entrypoints=route.python_entrypoints,
        fixture=scenario.fixture,
    )
    python_trace: Final = collect_trace(scenario_route, asynchronous=scenario.asynchronous)
    python_error: Final = None if isinstance(python_trace, tuple) else f"{python_trace.engine}: {python_trace.message}"
    python_events: Final = python_trace if isinstance(python_trace, tuple) else ()
    try:
        python: Final = pipeline_projection(python_events)
    except ValueError as error:
        return TraceArtifact.from_traces(
            surface=surface,
            sdk_function=route.route,
            scenario=scenario.name,
            python=(),
            python_error=f"harness: {error}",
        )
    return TraceArtifact.from_traces(
        surface=surface,
        sdk_function=route.route,
        scenario=scenario.name,
        python=python,
        python_error=python_error,
    )
