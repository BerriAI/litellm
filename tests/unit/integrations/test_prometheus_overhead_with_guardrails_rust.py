from datetime import datetime

import pytest
from prometheus_client import REGISTRY

from litellm.integrations.prometheus import PrometheusLogger
from litellm.types.guardrails import GuardrailEventHooks
from tests.unit.integrations.prometheus_native_support import (
    enable_native_prometheus,
    native_success,
    series,
    standard_payload,
)


@pytest.fixture(autouse=True)
def native_prometheus(monkeypatch):
    enable_native_prometheus(monkeypatch)


@pytest.fixture(autouse=True)
def cleanup_prometheus_registry():
    for collector in list(REGISTRY._collector_to_names.keys()):
        REGISTRY.unregister(collector)
    yield
    for collector in list(REGISTRY._collector_to_names.keys()):
        REGISTRY.unregister(collector)


METRIC = "litellm_overhead_with_guardrails_latency_metric"


async def _observed(hidden_params: dict | None = None, **fields: object) -> list[float]:
    logger = PrometheusLogger()
    now = datetime.now()
    payload = standard_payload(
        hidden_params={"additional_headers": None} | (hidden_params or {}),
        **fields,
    )
    kwargs = {
        "model": "gpt-4o",
        "litellm_params": {"metadata": {}},
        "start_time": now,
        "end_time": now,
        "standard_logging_object": payload,
    }
    await native_success(logger, kwargs, None)
    return [value for _, value in series(METRIC)]


def _guardrail_seconds(observed: list[float]) -> float:
    assert len(observed) <= 1
    return observed[0] if observed else 0.0


@pytest.mark.asyncio
async def test_get_guardrail_overhead_seconds_sums_pre_post_excludes_during():
    observed = await _observed(
        guardrail_information=[
            {"guardrail_mode": GuardrailEventHooks.pre_call, "duration": 0.1},
            {"guardrail_mode": GuardrailEventHooks.during_call, "duration": 0.5},
            {"guardrail_mode": GuardrailEventHooks.post_call, "duration": 0.25},
            {"guardrail_mode": GuardrailEventHooks.post_call},
        ],
    )
    assert abs(_guardrail_seconds(observed) - 0.35) < 1e-6


@pytest.mark.asyncio
async def test_get_guardrail_overhead_seconds_no_guardrails_is_zero():
    assert _guardrail_seconds(await _observed(model="gpt-4o")) == 0.0


@pytest.mark.asyncio
async def test_get_guardrail_overhead_seconds_accepts_plain_string_mode():
    observed = await _observed(
        guardrail_information=[
            {"guardrail_mode": "pre_call", "duration": 0.2},
            {"guardrail_mode": "during_call", "duration": 0.9},
        ],
    )
    assert abs(_guardrail_seconds(observed) - 0.2) < 1e-6


@pytest.mark.asyncio
async def test_get_guardrail_overhead_seconds_excludes_list_mode_with_during_call():
    observed = await _observed(
        guardrail_information=[
            {
                "guardrail_mode": [
                    GuardrailEventHooks.pre_call,
                    GuardrailEventHooks.during_call,
                ],
                "duration": 0.3,
            },
            {"guardrail_mode": GuardrailEventHooks.post_call, "duration": 0.05},
        ],
    )
    assert abs(_guardrail_seconds(observed) - 0.05) < 1e-6


@pytest.mark.asyncio
async def test_get_guardrail_overhead_seconds_counts_pure_pre_post_list_mode():
    observed = await _observed(
        guardrail_information=[
            {"guardrail_mode": [GuardrailEventHooks.pre_call], "duration": 0.1},
            {"guardrail_mode": ["post_call"], "duration": 0.2},
        ],
    )
    assert abs(_guardrail_seconds(observed) - 0.3) < 1e-6


@pytest.mark.asyncio
async def test_get_guardrail_overhead_seconds_excludes_logging_only_and_mcp():
    observed = await _observed(
        guardrail_information=[
            {"guardrail_mode": GuardrailEventHooks.logging_only, "duration": 0.4},
            {"guardrail_mode": GuardrailEventHooks.pre_mcp_call, "duration": 0.3},
            {"guardrail_mode": GuardrailEventHooks.during_mcp_call, "duration": 0.2},
            {"guardrail_mode": GuardrailEventHooks.post_call, "duration": 0.05},
        ],
    )
    assert abs(_guardrail_seconds(observed) - 0.05) < 1e-6


@pytest.mark.asyncio
async def test_get_guardrail_overhead_seconds_ignores_dict_mode_without_error():
    observed = await _observed(
        guardrail_information=[
            {"guardrail_mode": {"tags": {"default": ["pre_call"]}}, "duration": 0.3},
            {"guardrail_mode": GuardrailEventHooks.post_call, "duration": 0.05},
        ],
    )
    assert abs(_guardrail_seconds(observed) - 0.05) < 1e-6


@pytest.mark.asyncio
async def test_get_guardrail_overhead_seconds_ignores_dict_inside_list_mode():
    observed = await _observed(
        guardrail_information=[
            {"guardrail_mode": [GuardrailEventHooks.pre_call, {"k": "v"}], "duration": 0.1},
        ],
    )
    assert abs(_guardrail_seconds(observed) - 0.1) < 1e-6


@pytest.mark.asyncio
async def test_get_guardrail_overhead_seconds_handles_single_dict_payload():
    observed = await _observed(
        guardrail_information={
            "guardrail_mode": "logging_only",
            "duration": 0.7,
            "guardrail_name": "xecguard",
        },
    )
    assert _guardrail_seconds(observed) == 0.0


@pytest.mark.asyncio
async def test_get_guardrail_overhead_seconds_counts_single_pre_call_dict():
    observed = await _observed(
        guardrail_information={"guardrail_mode": "pre_call", "duration": 0.3},
    )
    assert abs(_guardrail_seconds(observed) - 0.3) < 1e-6


@pytest.mark.asyncio
async def test_overhead_with_guardrails_recorded_when_only_guardrails_no_sdk_overhead():
    observed = await _observed(
        hidden_params={},
        guardrail_information=[{"guardrail_mode": GuardrailEventHooks.post_call, "duration": 0.2}],
    )

    assert len(observed) == 1
    assert abs(observed[0] - 0.2) < 1e-6


@pytest.mark.asyncio
async def test_overhead_with_guardrails_recorded_when_sdk_overhead_is_zero():
    observed = await _observed(
        hidden_params={"litellm_overhead_time_ms": 0.0},
        guardrail_information=[{"guardrail_mode": GuardrailEventHooks.pre_call, "duration": 0.1}],
    )

    assert abs(observed[0] - 0.1) < 1e-6


@pytest.mark.asyncio
async def test_overhead_with_guardrails_skipped_when_no_overhead_and_no_guardrails():
    observed = await _observed(hidden_params={})

    assert observed == []


def test_overhead_with_guardrails_metric_is_registered():
    PrometheusLogger()

    registered = [family.name for family in REGISTRY.collect() if family.name == METRIC]
    assert registered, "litellm_overhead_with_guardrails_latency_metric not registered"
