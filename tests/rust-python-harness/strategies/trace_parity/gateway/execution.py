from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Final, Protocol, cast

import httpx
from pydantic import BaseModel, ConfigDict

from ....shared.parity.replay import replay_server
from ....shared.tracing.native import TraceResponsePayload, native_trace_events
from ....shared.tracing.profiler import FunctionTraceEvent, profile_python
from ....shared.tracing.steps import Engine, PipelineProjection, pipeline_projection
from ..models import GatewayRouteSpec, RouteFixture, TraceExecutionFailure, TraceMode, TraceScenario
from ..reporting import TraceComparisonArtifact


class _GatewayResponsePayload(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    status: int
    body: object


class _GatewayClient(Protocol):
    def post(self, url: str, *, json: object, headers: dict[str, str]) -> httpx.Response: ...


def _collect_python(fixture: RouteFixture) -> tuple[FunctionTraceEvent, ...]:
    import litellm
    from fastapi.testclient import TestClient

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.anthropic_endpoints.endpoints import user_api_key_auth
    from litellm.proxy import proxy_server

    provider_model: Final = cast(str, fixture.kwargs["provider_model"])
    model_alias: Final = cast(str, fixture.kwargs["model_alias"])
    old_router: Final = proxy_server.llm_router
    old_override: Final = proxy_server.app.dependency_overrides.get(user_api_key_auth)

    async def authorize() -> UserAPIKeyAuth:
        return UserAPIKeyAuth(api_key="trace-key")

    proxy_server.llm_router = litellm.Router(
        model_list=[
            {
                "model_name": model_alias,
                "litellm_params": {
                    "model": provider_model,
                    "api_key": "trace-provider-key",
                    "api_base": fixture.kwargs["api_base"],
                },
            }
        ]
    )
    proxy_server.app.dependency_overrides[user_api_key_auth] = authorize
    try:
        with profile_python(Path(litellm.__file__).parent, threads=True) as profiler:
            client: Final = cast(_GatewayClient, TestClient(proxy_server.app))
            response: Final = client.post(
                "/v1/messages",
                json=fixture.kwargs["body"],
                headers={"authorization": "Bearer trace-key"},
            )
        if response.status_code != 200:
            raise RuntimeError(f"Python gateway returned {response.status_code}: {response.text}")
        return tuple(profiler.events)
    finally:
        proxy_server.llm_router = old_router
        if old_override is None:
            proxy_server.app.dependency_overrides.pop(user_api_key_auth, None)
        else:
            proxy_server.app.dependency_overrides[user_api_key_auth] = old_override


def _collect_rust(fixture: RouteFixture) -> tuple[FunctionTraceEvent, ...]:
    from litellm.rust_bridge import get_native_bridge

    bridge: Final[object | None] = get_native_bridge()
    trace: Final[object | None] = getattr(bridge, "_trace", None) if bridge is not None else None
    gateway_messages: Final[object | None] = getattr(trace, "gateway_messages", None)
    if gateway_messages is None or not callable(gateway_messages):
        raise RuntimeError("native Rust trace bridge does not expose gateway_messages")
    invoke_gateway: Final = cast(Callable[[str, str, str, object], Awaitable[object]], gateway_messages)

    async def invoke() -> object:
        return await invoke_gateway(
            cast(str, fixture.kwargs["model_alias"]),
            cast(str, fixture.kwargs["provider_model"]),
            cast(str, fixture.kwargs["api_base"]),
            fixture.kwargs["body"],
        )

    result: Final = asyncio.run(invoke())
    payload: Final = TraceResponsePayload.model_validate(result)
    response: Final = _GatewayResponsePayload.model_validate(payload.response)
    if response.status != 200:
        raise RuntimeError(f"Rust gateway returned {response.status}: {response.body}")
    return native_trace_events(payload)


def _collect(scenario: TraceScenario, engine: Engine) -> tuple[FunctionTraceEvent, ...] | TraceExecutionFailure:
    try:
        with replay_server() as provider:
            base_fixture: Final = scenario.fixture(engine, provider.url)
            fixture: Final = RouteFixture(
                kwargs={**base_fixture.kwargs, "api_base": provider.url},
                provider_responses=base_fixture.provider_responses,
            )
            for response in fixture.provider_responses:
                provider.enqueue_response(response)
            events: Final = _collect_python(fixture) if engine == "python" else _collect_rust(fixture)
            provider.take_requests(len(fixture.provider_responses))
        return events
    except Exception as error:
        return TraceExecutionFailure(engine, f"{type(error).__name__}: {error}")


def _projections(
    python_events: tuple[FunctionTraceEvent, ...],
    rust_events: tuple[FunctionTraceEvent, ...],
    scenario: TraceScenario,
    mode: TraceMode,
) -> tuple[PipelineProjection, PipelineProjection, str | None]:
    mappings: Final = scenario.mappings_for(mode)
    try:
        return (
            pipeline_projection("python", python_events, mappings),
            pipeline_projection("rust", rust_events, mappings),
            None,
        )
    except ValueError as error:
        return PipelineProjection(), PipelineProjection(), f"harness: {error}"


def execute_gateway_trace(route: GatewayRouteSpec, scenario: TraceScenario, mode: TraceMode) -> TraceComparisonArtifact:
    mappings: Final = scenario.mappings_for(mode)
    python_trace: Final = _collect(scenario, "python")
    rust_trace: Final = _collect(scenario, "rust")
    collection_python_error: Final = None if isinstance(python_trace, tuple) else f"python: {python_trace.message}"
    rust_error: Final = None if isinstance(rust_trace, tuple) else f"rust: {rust_trace.message}"
    python_events: Final = python_trace if isinstance(python_trace, tuple) else ()
    rust_events: Final = rust_trace if isinstance(rust_trace, tuple) else ()
    python, rust, projection_error = _projections(python_events, rust_events, scenario, mode)
    python_error: Final = projection_error or collection_python_error
    return TraceComparisonArtifact.from_traces(
        surface="gateway",
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
