"""
Unit tests for the service_tier Prometheus label on latency and spend metrics.

Covers the label being declared on the metrics that carry it, the precedence
between the tier a provider served and the tier a caller requested, and the
end-to-end emit wiring through async_log_success_event.

Run with:
    uv run pytest tests/test_litellm/integrations/test_prometheus_service_tier_label.py -v
"""

import datetime

import pytest

from litellm.integrations.prometheus import PrometheusLogger
from litellm.litellm_core_utils.service_tier_utils import (
    KNOWN_REQUEST_SERVICE_TIERS,
    get_service_tier_from_standard_logging_payload,
)
from litellm.types.integrations.prometheus import (
    PrometheusMetricLabels,
    UserAPIKeyLabelNames,
    UserAPIKeyLabelValues,
)

SERVICE_TIER_METRICS = [
    "litellm_llm_api_latency_metric",
    "litellm_llm_api_time_to_first_token_metric",
    "litellm_request_total_latency_metric",
    "litellm_spend_metric",
]


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


def test_service_tier_label_declared_on_latency_and_spend_metrics():
    for metric_name in SERVICE_TIER_METRICS:
        labels = PrometheusMetricLabels.get_labels(metric_name)
        assert UserAPIKeyLabelNames.SERVICE_TIER.value in labels, f"{metric_name} should carry the service_tier label"


def test_user_api_key_label_values_carries_service_tier():
    values = UserAPIKeyLabelValues(service_tier="flex")

    assert values.service_tier == "flex"
    assert values.model_dump()["service_tier"] == "flex"
    assert UserAPIKeyLabelValues().service_tier is None


def test_served_tier_wins_over_requested_tier():
    payload = _standard_logging_payload(
        response={"service_tier": "default"},
        model_parameters={"service_tier": "auto"},
    )

    assert get_service_tier_from_standard_logging_payload(payload) == "default"


def test_usage_object_tier_used_when_response_has_none():
    payload = _standard_logging_payload(
        response={"id": "chatcmpl-1"},
        usage_object={"prompt_tokens": 1, "service_tier": "standard"},
        model_parameters={"service_tier": "auto"},
    )

    assert get_service_tier_from_standard_logging_payload(payload) == "standard"


def test_requested_tier_used_when_no_served_tier():
    payload = _standard_logging_payload(
        response={"id": "chatcmpl-1"},
        usage_object={"prompt_tokens": 1},
        model_parameters={"service_tier": "flex"},
    )

    assert get_service_tier_from_standard_logging_payload(payload) == "flex"


def test_unrecognized_requested_tier_is_not_labelled():
    """
    A caller-supplied tier survives param mapping even where the provider then
    ignores it (Bedrock and Groq drop an unrecognized tier and still answer), so
    labelling it verbatim would let one caller mint a series per string.
    """
    payload = _standard_logging_payload(model_parameters={"service_tier": "attacker-controlled-a1b2c3"})

    assert get_service_tier_from_standard_logging_payload(payload) is None


def test_unrecognized_served_tier_is_labelled():
    """
    The tier a provider reports is not caller-controlled, so a tier added by a
    provider after this release still gets labelled instead of being dropped.
    """
    payload = _standard_logging_payload(
        response={"service_tier": "tier-added-by-provider-later"},
        model_parameters={"service_tier": "auto"},
    )

    assert get_service_tier_from_standard_logging_payload(payload) == "tier-added-by-provider-later"


@pytest.mark.parametrize("tier", sorted(KNOWN_REQUEST_SERVICE_TIERS))
def test_every_known_requested_tier_is_labelled(tier):
    payload = _standard_logging_payload(model_parameters={"service_tier": tier})

    assert get_service_tier_from_standard_logging_payload(payload) == tier


@pytest.mark.parametrize(
    "response, usage_object, model_parameters",
    [
        (None, None, None),
        ({"service_tier": None}, {"service_tier": ""}, {"service_tier": None}),
        ("redacted-by-litellm", None, {}),
        ({"service_tier": 1}, None, None),
    ],
)
def test_no_tier_resolves_to_none(response, usage_object, model_parameters):
    payload = _standard_logging_payload(
        response=response,
        usage_object=usage_object,
        model_parameters=model_parameters,
    )

    assert get_service_tier_from_standard_logging_payload(payload) is None


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
        await logger.async_log_success_event(kwargs, None, now, now)

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


def test_allowlist_covers_every_modeled_service_tier():
    """A tier modeled for cost calculation is real traffic, so it must resolve
    rather than being dropped as an unknown caller value."""
    from litellm.types.utils import ServiceTier

    missing = {tier.value for tier in ServiceTier} - KNOWN_REQUEST_SERVICE_TIERS
    assert not missing, f"ServiceTier values missing from the allowlist: {sorted(missing)}"
