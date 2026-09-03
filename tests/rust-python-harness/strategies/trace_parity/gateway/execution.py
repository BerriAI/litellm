from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast

from ....shared.parity.replay import replay_server
from ....shared.reporting.models import SdkFunction
from ....shared.tracing.native import native_trace_events
from ....shared.tracing.profiler import FunctionTraceEvent, profile_python
from ....shared.tracing.steps import Engine, pipeline_projection
from ..reporting import TraceComparisonArtifact
from ..sdk.execution import RouteFixture, TraceExecutionFailure, TraceMode, TraceScenario


@dataclass(frozen=True, slots=True)
class GatewayRouteSpec:
    route: SdkFunction


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
            response = TestClient(proxy_server.app).post(
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

    bridge: Final = get_native_bridge()
    trace = getattr(bridge, "_trace", None) if bridge is not None else None
    if trace is None or not hasattr(trace, "gateway_messages"):
        raise RuntimeError("native Rust trace bridge does not expose gateway_messages")

    async def invoke() -> object:
        return await trace.gateway_messages(
            fixture.kwargs["model_alias"],
            fixture.kwargs["provider_model"],
            fixture.kwargs["api_base"],
            fixture.kwargs["body"],
        )

    result: Final = asyncio.run(invoke())
    response: Final = result.get("response", {})
    if response.get("status") != 200:
        raise RuntimeError(f"Rust gateway returned {response.get('status')}: {response.get('body')}")
    return native_trace_events(result)


def _collect(
    scenario: TraceScenario, engine: Engine
) -> tuple[FunctionTraceEvent, ...] | TraceExecutionFailure:
    try:
        with replay_server() as provider:
            fixture: Final = scenario.fixture(engine, provider.url)
            fixture = RouteFixture(
                kwargs={**fixture.kwargs, "api_base": provider.url},
                provider_responses=fixture.provider_responses,
            )
            for response in fixture.provider_responses:
                provider.enqueue_response(response)
            events: Final = _collect_python(fixture) if engine == "python" else _collect_rust(fixture)
            provider.take_requests(len(fixture.provider_responses))
        return events
    except Exception as error:
        return TraceExecutionFailure(engine, f"{type(error).__name__}: {error}")


def execute_gateway_trace(
    route: GatewayRouteSpec, scenario: TraceScenario, mode: TraceMode
) -> TraceComparisonArtifact:
    mappings: Final = scenario.mappings_for(mode)
    python_trace: Final = _collect(scenario, "python")
    rust_trace: Final = _collect(scenario, "rust")
    python_error: Final = None if isinstance(python_trace, tuple) else f"python: {python_trace.message}"
    rust_error: Final = None if isinstance(rust_trace, tuple) else f"rust: {rust_trace.message}"
    python_events: Final = python_trace if isinstance(python_trace, tuple) else ()
    rust_events: Final = rust_trace if isinstance(rust_trace, tuple) else ()
    try:
        python = pipeline_projection("python", python_events, mappings)
        rust = pipeline_projection("rust", rust_events, mappings)
    except ValueError as error:
        python_error = f"harness: {error}"
        python = pipeline_projection("python", (), mappings)
        rust = pipeline_projection("rust", (), mappings)
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
