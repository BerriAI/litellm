from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, cast

import pytest

from ....shared.parity.recorded_http import RecordedHttpResponse
from ....shared.parity.replay import replay_server
from ....shared.tracing.native import native_trace_events
from ....shared.tracing.profiler import FunctionTraceEvent, profile_python
from ....shared.tracing.steps import Engine, Step, pipeline_issues, pipeline_steps, trace_diff


class SdkCall(Protocol):
    def __call__(self, **kwargs: object) -> object: ...


@dataclass(frozen=True, slots=True)
class RouteFixture:
    kwargs: dict[str, object]
    provider_response: RecordedHttpResponse


@dataclass(frozen=True, slots=True)
class RouteSpec:
    route: str
    python_entrypoints: tuple[str, str]
    rust_entrypoints: tuple[str, str]
    fixture: Callable[[Engine], RouteFixture]


def _invoke(function: SdkCall, kwargs: dict[str, object], *, asynchronous: bool) -> object:
    async def invoke_async() -> object:
        return await cast(Awaitable[object], function(**kwargs))

    if asynchronous:
        return asyncio.run(invoke_async())
    return function(**kwargs)


def _entrypoint(spec: RouteSpec, engine: Engine, *, asynchronous: bool) -> SdkCall:
    import litellm
    from litellm.anthropic_interface import messages as sdk_messages
    from litellm.rust_bridge import get_native_bridge

    if engine == "rust":
        bridge: Final = get_native_bridge()
        if bridge is None:
            pytest.fail("native Rust bridge is required for trace parity testing")
        return cast(SdkCall, getattr(bridge, spec.rust_entrypoints[int(asynchronous)]))
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


def collect_trace(spec: RouteSpec, engine: Engine, *, asynchronous: bool) -> tuple[FunctionTraceEvent, ...]:
    fixture: Final = spec.fixture(engine)
    function: Final = _entrypoint(spec, engine, asynchronous=asynchronous)
    with replay_server() as provider:
        provider.enqueue_response(fixture.provider_response)
        kwargs: Final = {
            **fixture.kwargs,
            "api_key": "test-key",
            "api_base": provider.url,
            **({"trace": True, "timeout_seconds": 5} if engine == "rust" else {"timeout": 5}),
        }
        events: Final = _collect(function, kwargs, engine, asynchronous=asynchronous)
        provider.take_requests(1)
    if not events:
        pytest.fail("native Rust bridge trace is empty; rebuild it with tracing support")
    return events


def assert_trace_parity(
    spec: RouteSpec,
    steps: Sequence[Step],
    edges: Sequence[tuple[str, str]],
    *,
    asynchronous: bool,
    matching_steps: bool = True,
    exact: bool = False,
) -> None:
    python: Final = pipeline_steps("python", collect_trace(spec, "python", asynchronous=asynchronous), steps)
    rust: Final = pipeline_steps("rust", collect_trace(spec, "rust", asynchronous=asynchronous), steps)

    assert pipeline_issues("python", python, steps, edges) == ()
    assert pipeline_issues("rust", rust, steps, edges) == ()
    if matching_steps:
        assert trace_diff(python, rust).matches
    if exact:
        assert python == rust
