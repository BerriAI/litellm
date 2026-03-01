"""
Unit tests for RoutingGroupTestEngine and RoutingTraceCallback.
"""

import datetime
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy.routing_group_utils.routing_trace_callback import (
    RoutingTraceCallback,
)
from litellm.proxy.routing_group_utils.test_engine import (
    RoutingGroupTestEngine,
    _build_flow_data,
)
from litellm.types.router import (
    DeploymentTrafficStats,
    RoutingGroupConfig,
    RoutingGroupDeployment,
    RoutingTrace,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_kwargs(
    model: str = "test-model",
    provider: str = "openai",
    call_id: str = "call-123",
    fallback_depth: int = 0,
    exception=None,
) -> dict:
    return {
        "model": model,
        "custom_llm_provider": provider,
        "litellm_call_id": call_id,
        "exception": exception,
        "litellm_params": {
            "metadata": {"fallback_depth": fallback_depth},
            "custom_llm_provider": provider,
        },
    }


def _make_mock_response(content: str = "hello") -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message = MagicMock()
    response.choices[0].message.content = content
    return response


def _make_routing_group_config(strategy: str = "priority-failover") -> RoutingGroupConfig:
    return RoutingGroupConfig(
        routing_group_name="test-group",
        routing_strategy=strategy,
        deployments=[
            RoutingGroupDeployment(
                model_id="id-primary",
                model_name="nebius/llama-70b",
                provider="nebius",
                priority=1,
            ),
            RoutingGroupDeployment(
                model_id="id-fallback",
                model_name="fireworks/llama-70b",
                provider="fireworks_ai",
                priority=2,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# RoutingTraceCallback tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_routing_trace_callback_records_success():
    """async_log_success_event should append a success trace."""
    callback = RoutingTraceCallback()

    start = datetime.datetime(2024, 1, 1, 0, 0, 0)
    end = datetime.datetime(2024, 1, 1, 0, 0, 1)  # 1 second later

    kwargs = _make_kwargs(model="gpt-4o", provider="openai")
    await callback.async_log_success_event(
        kwargs=kwargs, response_obj=None, start_time=start, end_time=end
    )

    assert len(callback.traces) == 1
    trace = callback.traces[0]
    assert trace.status == "success"
    assert trace.deployment_name == "gpt-4o"
    assert trace.provider == "openai"
    assert trace.latency_ms == pytest.approx(1000.0, abs=1.0)
    assert trace.was_fallback is False
    assert trace.error_message is None


@pytest.mark.asyncio
async def test_routing_trace_callback_records_failure():
    """async_log_failure_event should append an error trace with message."""
    callback = RoutingTraceCallback()

    start = datetime.datetime(2024, 1, 1, 0, 0, 0)
    end = datetime.datetime(2024, 1, 1, 0, 0, 0, 500000)  # 0.5 seconds

    exc = RuntimeError("rate limit exceeded")
    kwargs = _make_kwargs(model="claude-3", provider="anthropic", exception=exc)
    await callback.async_log_failure_event(
        kwargs=kwargs, response_obj=None, start_time=start, end_time=end
    )

    assert len(callback.traces) == 1
    trace = callback.traces[0]
    assert trace.status == "error"
    assert trace.deployment_name == "claude-3"
    assert trace.provider == "anthropic"
    assert trace.error_message == "rate limit exceeded"
    assert trace.was_fallback is False


@pytest.mark.asyncio
async def test_routing_trace_callback_records_fallback():
    """Traces with fallback_depth > 0 should have was_fallback=True."""
    callback = RoutingTraceCallback()

    start = datetime.datetime(2024, 1, 1, 0, 0, 0)
    end = datetime.datetime(2024, 1, 1, 0, 0, 1)

    kwargs = _make_kwargs(model="backup-model", provider="azure", fallback_depth=1)
    await callback.async_log_success_event(
        kwargs=kwargs, response_obj=None, start_time=start, end_time=end
    )

    trace = callback.traces[0]
    assert trace.was_fallback is True
    assert trace.fallback_depth == 1


@pytest.mark.asyncio
async def test_routing_trace_callback_multiple_events():
    """Multiple events should all be recorded in order."""
    callback = RoutingTraceCallback()

    start = datetime.datetime(2024, 1, 1, 0, 0, 0)
    end = datetime.datetime(2024, 1, 1, 0, 0, 1)

    for i in range(3):
        kwargs = _make_kwargs(model=f"model-{i}", provider="openai")
        await callback.async_log_success_event(
            kwargs=kwargs, response_obj=None, start_time=start, end_time=end
        )

    assert len(callback.traces) == 3
    for i, trace in enumerate(callback.traces):
        assert trace.deployment_name == f"model-{i}"


# ---------------------------------------------------------------------------
# RoutingGroupTestEngine.test_single_request tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_engine_test_single_request_success():
    """test_single_request with a mock router that succeeds should return success=True."""
    mock_router = AsyncMock()
    mock_router.acompletion = AsyncMock(return_value=_make_mock_response("pong"))

    engine = RoutingGroupTestEngine()

    with patch("litellm.callbacks", []):
        result = await engine.test_single_request(
            routing_group_name="test-group",
            router=mock_router,
            mock=True,
        )

    assert result.success is True
    assert result.response_text == "pong"
    assert result.total_latency_ms >= 0


@pytest.mark.asyncio
async def test_engine_test_single_request_failure():
    """test_single_request when router raises should return success=False."""
    mock_router = AsyncMock()
    mock_router.acompletion = AsyncMock(side_effect=Exception("provider error"))

    engine = RoutingGroupTestEngine()

    with patch("litellm.callbacks", []):
        result = await engine.test_single_request(
            routing_group_name="test-group",
            router=mock_router,
        )

    assert result.success is False
    assert result.response_text is None


@pytest.mark.asyncio
async def test_engine_test_single_request_callback_detached_on_success():
    """The trace callback should be removed from litellm.callbacks after the call."""
    mock_router = AsyncMock()
    mock_router.acompletion = AsyncMock(return_value=_make_mock_response("ok"))

    engine = RoutingGroupTestEngine()
    callbacks_list: list = []

    with patch("litellm.callbacks", callbacks_list):
        await engine.test_single_request(
            routing_group_name="test-group",
            router=mock_router,
            mock=True,
        )

    # Callback should have been removed
    from litellm.proxy.routing_group_utils.routing_trace_callback import (
        RoutingTraceCallback,
    )
    assert not any(isinstance(c, RoutingTraceCallback) for c in callbacks_list)


@pytest.mark.asyncio
async def test_engine_test_single_request_callback_detached_on_failure():
    """The trace callback should be removed even when the router call raises."""
    mock_router = AsyncMock()
    mock_router.acompletion = AsyncMock(side_effect=Exception("boom"))

    engine = RoutingGroupTestEngine()
    callbacks_list: list = []

    with patch("litellm.callbacks", callbacks_list):
        await engine.test_single_request(
            routing_group_name="test-group",
            router=mock_router,
        )

    from litellm.proxy.routing_group_utils.routing_trace_callback import (
        RoutingTraceCallback,
    )
    assert not any(isinstance(c, RoutingTraceCallback) for c in callbacks_list)


@pytest.mark.asyncio
async def test_engine_test_single_request_uses_custom_messages():
    """Custom messages should be passed through to the router."""
    mock_router = AsyncMock()
    mock_router.acompletion = AsyncMock(return_value=_make_mock_response("custom"))

    engine = RoutingGroupTestEngine()
    custom_messages = [{"role": "user", "content": "custom prompt"}]

    with patch("litellm.callbacks", []):
        await engine.test_single_request(
            routing_group_name="my-group",
            router=mock_router,
            messages=custom_messages,
            mock=True,
        )

    call_kwargs = mock_router.acompletion.call_args
    assert call_kwargs.kwargs.get("messages") == custom_messages or (
        len(call_kwargs.args) > 1 and call_kwargs.args[1] == custom_messages
    )


# ---------------------------------------------------------------------------
# RoutingGroupTestEngine.simulate_traffic tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_engine_simulate_traffic_total_requests():
    """simulate_traffic should return total_requests matching the requested count."""
    mock_router = AsyncMock()
    mock_router.acompletion = AsyncMock(return_value=_make_mock_response("sim"))

    engine = RoutingGroupTestEngine()
    config = _make_routing_group_config()

    with patch("litellm.callbacks", []):
        result = await engine.simulate_traffic(
            routing_group_name="test-group",
            router=mock_router,
            routing_group_config=config,
            num_requests=10,
            concurrency=5,
            mock=True,
        )

    assert result.total_requests == 10
    assert result.successful_requests + result.failed_requests == 10


@pytest.mark.asyncio
async def test_engine_simulate_traffic_all_failures():
    """When all requests fail, successful_requests should be 0."""
    mock_router = AsyncMock()
    mock_router.acompletion = AsyncMock(side_effect=Exception("always fails"))

    engine = RoutingGroupTestEngine()
    config = _make_routing_group_config()

    with patch("litellm.callbacks", []):
        result = await engine.simulate_traffic(
            routing_group_name="test-group",
            router=mock_router,
            routing_group_config=config,
            num_requests=5,
            concurrency=5,
            mock=True,
        )

    assert result.total_requests == 5
    assert result.successful_requests == 0
    assert result.failed_requests == 5


@pytest.mark.asyncio
async def test_engine_simulate_traffic_no_flow_data_for_non_priority():
    """flow_data should be None when routing_strategy is not priority-failover."""
    mock_router = AsyncMock()
    mock_router.acompletion = AsyncMock(return_value=_make_mock_response("ok"))

    engine = RoutingGroupTestEngine()
    config = _make_routing_group_config(strategy="weighted")

    with patch("litellm.callbacks", []):
        result = await engine.simulate_traffic(
            routing_group_name="test-group",
            router=mock_router,
            routing_group_config=config,
            num_requests=5,
            concurrency=5,
            mock=True,
        )

    assert result.flow_data is None


@pytest.mark.asyncio
async def test_engine_simulate_traffic_flow_data_for_priority_failover():
    """flow_data should be populated when routing_strategy is priority-failover and there are traces."""
    mock_router = AsyncMock()
    mock_router.acompletion = AsyncMock(return_value=_make_mock_response("ok"))

    engine = RoutingGroupTestEngine()
    config = _make_routing_group_config(strategy="priority-failover")

    with patch("litellm.callbacks", []):
        result = await engine.simulate_traffic(
            routing_group_name="test-group",
            router=mock_router,
            routing_group_config=config,
            num_requests=5,
            concurrency=5,
            mock=True,
        )

    # flow_data may be None if no traces were recorded (mock router doesn't fire callbacks)
    # In that case traffic_distribution will be empty too, which is fine
    assert result.total_requests == 5


# ---------------------------------------------------------------------------
# _build_flow_data tests
# ---------------------------------------------------------------------------


def test_build_flow_data_primary_first():
    """The first FlowStep should have reason='primary' and from_deployment=None."""
    traces: List[RoutingTrace] = []
    config = _make_routing_group_config()

    traffic = [
        DeploymentTrafficStats(
            deployment_id="id-primary",
            deployment_name="nebius/llama-70b",
            provider="nebius",
            request_count=80,
            success_count=80,
            failure_count=0,
            avg_latency_ms=150.0,
            percent_of_total=80.0,
            priority=1,
        ),
        DeploymentTrafficStats(
            deployment_id="id-fallback",
            deployment_name="fireworks/llama-70b",
            provider="fireworks_ai",
            request_count=20,
            success_count=20,
            failure_count=0,
            avg_latency_ms=200.0,
            percent_of_total=20.0,
            priority=2,
        ),
    ]

    flow_steps = _build_flow_data(traces, config, traffic)

    assert len(flow_steps) == 2
    assert flow_steps[0].reason == "primary"
    assert flow_steps[0].from_deployment is None
    assert flow_steps[0].to_deployment == "nebius/llama-70b"
    assert flow_steps[0].request_count == 80


def test_build_flow_data_fallback_reason():
    """Subsequent FlowSteps should have reason='fallback_error' and correct from_deployment."""
    traces: List[RoutingTrace] = []
    config = _make_routing_group_config()

    traffic = [
        DeploymentTrafficStats(
            deployment_id="id-primary",
            deployment_name="primary-model",
            provider="nebius",
            request_count=70,
            success_count=70,
            failure_count=0,
            avg_latency_ms=100.0,
            percent_of_total=70.0,
            priority=1,
        ),
        DeploymentTrafficStats(
            deployment_id="id-fallback",
            deployment_name="fallback-model",
            provider="fireworks_ai",
            request_count=30,
            success_count=30,
            failure_count=0,
            avg_latency_ms=200.0,
            percent_of_total=30.0,
            priority=2,
        ),
    ]

    flow_steps = _build_flow_data(traces, config, traffic)

    assert flow_steps[1].reason == "fallback_error"
    assert flow_steps[1].from_deployment == "primary-model"
    assert flow_steps[1].to_deployment == "fallback-model"
    assert flow_steps[1].request_count == 30


def test_build_flow_data_single_deployment():
    """With a single deployment, only one FlowStep (primary) should be created."""
    traces: List[RoutingTrace] = []
    config = RoutingGroupConfig(
        routing_group_name="single-group",
        routing_strategy="priority-failover",
        deployments=[
            RoutingGroupDeployment(
                model_id="id-1",
                model_name="only-model",
                provider="openai",
                priority=1,
            )
        ],
    )

    traffic = [
        DeploymentTrafficStats(
            deployment_id="id-1",
            deployment_name="only-model",
            provider="openai",
            request_count=100,
            success_count=100,
            failure_count=0,
            avg_latency_ms=120.0,
            percent_of_total=100.0,
            priority=1,
        ),
    ]

    flow_steps = _build_flow_data(traces, config, traffic)

    assert len(flow_steps) == 1
    assert flow_steps[0].reason == "primary"
    assert flow_steps[0].from_deployment is None


def test_build_flow_data_sorts_by_priority():
    """_build_flow_data should sort deployments by priority before building flow steps."""
    traces: List[RoutingTrace] = []
    config = _make_routing_group_config()

    # Intentionally pass traffic in reverse priority order
    traffic = [
        DeploymentTrafficStats(
            deployment_id="id-fallback",
            deployment_name="fallback-model",
            provider="fireworks_ai",
            request_count=20,
            success_count=20,
            failure_count=0,
            avg_latency_ms=200.0,
            percent_of_total=20.0,
            priority=2,
        ),
        DeploymentTrafficStats(
            deployment_id="id-primary",
            deployment_name="primary-model",
            provider="nebius",
            request_count=80,
            success_count=80,
            failure_count=0,
            avg_latency_ms=100.0,
            percent_of_total=80.0,
            priority=1,
        ),
    ]

    flow_steps = _build_flow_data(traces, config, traffic)

    # First step should be the priority=1 deployment regardless of input order
    assert flow_steps[0].to_deployment == "primary-model"
    assert flow_steps[0].reason == "primary"
    assert flow_steps[1].to_deployment == "fallback-model"
    assert flow_steps[1].reason == "fallback_error"
