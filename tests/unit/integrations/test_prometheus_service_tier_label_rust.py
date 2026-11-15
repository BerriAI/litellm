import datetime

import pytest

from litellm.integrations.prometheus import PrometheusLogger
from tests.unit.integrations.prometheus_native_support import enable_native_prometheus, native_success


@pytest.fixture(autouse=True)
def native_prometheus(monkeypatch):
    enable_native_prometheus(monkeypatch)


def _clear_prometheus_registry() -> None:
    from prometheus_client import REGISTRY

    for collector in list(REGISTRY._collector_to_names.keys()):
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass


def _collected_samples(metric_name: str):
    from prometheus_client import REGISTRY

    return [sample for metric in REGISTRY.collect() for sample in metric.samples if sample.name == metric_name]


def _standard_logging_payload(
    response: object = None,
    usage_object: object = None,
    model_parameters: object = None,
) -> dict:
    return {
        "id": "t",
        "call_type": "completion",
        "response_cost": 0.001,
        "status": "success",
        "total_tokens": 30,
        "prompt_tokens": 20,
        "completion_tokens": 10,
        "startTime": 1.0,
        "endTime": 2.0,
        "completionStartTime": 1.5,
        "model": "gpt-4o-mini",
        "model_id": "model-123",
        "model_group": "gpt-4o-mini",
        "api_base": "https://api.openai.com",
        "custom_llm_provider": "openai",
        "request_tags": [],
        "end_user": None,
        "cache_hit": False,
        "stream": True,
        "response": response,
        "model_parameters": model_parameters,
        "metadata": {
            "user_api_key_hash": "h",
            "user_api_key_alias": "a",
            "user_api_key_team_id": "t",
            "user_api_key_team_alias": "ta",
            "user_api_key_user_id": "u",
            "user_api_key_user_email": "e@x.com",
            "user_api_key_org_id": None,
            "user_api_key_org_alias": None,
            "requester_metadata": None,
            "user_api_key_end_user_id": None,
            "usage_object": usage_object,
        },
        "hidden_params": {"litellm_overhead_time_ms": None, "additional_headers": None},
    }


@pytest.mark.asyncio
async def test_success_event_emits_service_tier_on_latency_and_spend_metrics():
    """
    End-to-end emit wiring.

    Drives the real logger with a payload whose response was served on the flex
    tier and asserts every latency histogram and the spend counter carries
    service_tier="flex". Fails if the label is dropped from a metric's label list
    or if the value is not populated on the success path.
    """
    payload = _standard_logging_payload(
        response={"service_tier": "flex"},
        model_parameters={"service_tier": "auto"},
    )
    now = datetime.datetime.now()
    kwargs = {
        "model": "gpt-4o-mini",
        "litellm_params": {"metadata": {}},
        "standard_logging_object": payload,
        "stream": True,
        "start_time": now - datetime.timedelta(seconds=3),
        "api_call_start_time": now - datetime.timedelta(seconds=2),
        "completion_start_time": now - datetime.timedelta(seconds=1),
        "end_time": now,
    }

    _clear_prometheus_registry()
    try:
        logger = PrometheusLogger()
        await native_success(logger, kwargs, None, now, now)

        for metric_name in (
            "litellm_request_total_latency_metric_bucket",
            "litellm_llm_api_latency_metric_bucket",
            "litellm_llm_api_time_to_first_token_metric_bucket",
            "litellm_spend_metric_total",
        ):
            samples = _collected_samples(metric_name)
            assert samples, f"expected {metric_name} to be emitted"
            assert all(sample.labels.get("service_tier") == "flex" for sample in samples), (
                f"{metric_name} must carry service_tier=flex, got "
                f"{sorted({sample.labels.get('service_tier') for sample in samples})}"
            )
    finally:
        _clear_prometheus_registry()
