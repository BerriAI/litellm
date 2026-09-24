"""
Tests for litellm/router_utils/cooldown_callbacks.py
"""

import datetime

import pytest

import litellm
from litellm import Router
from litellm.integrations.prometheus import PrometheusLogger
from litellm.router_utils.cooldown_callbacks import router_cooldown_event_callback


def _clear_prometheus_registry() -> None:
    from prometheus_client import REGISTRY

    for collector in list(REGISTRY._collector_to_names.keys()):
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass


def _collected_samples(metric_name: str):
    from prometheus_client import REGISTRY

    return [
        sample
        for metric in REGISTRY.collect()
        for sample in metric.samples
        if sample.name == metric_name
    ]


def _standard_logging_payload(litellm_model_name: str, model_id: str) -> dict:
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
        "model": litellm_model_name,
        "model_id": model_id,
        "model_group": "claude-opus-4-8",
        "api_base": "https://api.anthropic.com/v1/messages",
        "custom_llm_provider": "anthropic",
        "request_tags": [],
        "end_user": None,
        "cache_hit": False,
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
        },
        "hidden_params": {"litellm_overhead_time_ms": None, "additional_headers": None},
    }


@pytest.mark.asyncio
async def test_deployment_state_is_single_series_across_cooldown_failure_and_recovery(monkeypatch):
    """
    Regression test for litellm_deployment_state series fragmentation.

    The gauge's series identity is (litellm_model_name, model_id,
    api_provider). router_cooldown_event_callback used to label it with the
    deployment's public model_name group alias, and the gauge additionally
    carried an api_base label whose value differed between the config-driven
    cooldown path and the request-driven success/failure paths. Each writer
    therefore created its own series: the cooldown's state=2 was never reset by
    recovery, so dashboards reported the deployment as permanently unhealthy.

    Drives the real cooldown callback and the real success/failure logging
    paths for one deployment with no configured api_base (mirroring production
    anthropic/bedrock configs) and asserts every write lands on a single fully
    labeled time series that follows the deployment's actual state.
    """
    model_id = "deployment-state-regression-id"
    litellm_model_name = "anthropic/claude-opus-4-8"

    router = Router(
        model_list=[
            {
                "model_name": "claude-opus-4-8",
                "litellm_params": {
                    "model": litellm_model_name,
                    "api_key": "fake-key",
                },
                "model_info": {"id": model_id},
            }
        ]
    )

    _clear_prometheus_registry()
    try:
        logger = PrometheusLogger()
        monkeypatch.setattr(litellm, "callbacks", [logger])

        async def cooldown():
            await router_cooldown_event_callback(
                litellm_router_instance=router,
                deployment_id=model_id,
                exception_status="429",
                cooldown_time=60.0,
            )

        def single_deployment_state_sample():
            samples = _collected_samples("litellm_deployment_state")
            assert len(samples) == 1, (
                f"expected a single litellm_deployment_state series, got {[s.labels for s in samples]}"
            )
            assert samples[0].labels == {
                "litellm_model_name": litellm_model_name,
                "model_id": model_id,
                "api_provider": "anthropic",
            }
            return samples[0]

        await cooldown()
        assert single_deployment_state_sample().value == 2

        cooled_down_samples = _collected_samples("litellm_deployment_cooled_down_total")
        assert len(cooled_down_samples) == 1
        assert cooled_down_samples[0].labels["litellm_model_name"] == litellm_model_name

        now = datetime.datetime.now()
        await logger.async_log_success_event(
            {
                "model": litellm_model_name,
                "litellm_params": {
                    "custom_llm_provider": "anthropic",
                    "metadata": {"model_info": {"id": model_id}},
                },
                "standard_logging_object": _standard_logging_payload(litellm_model_name, model_id),
            },
            None,
            now,
            now,
        )
        assert single_deployment_state_sample().value == 0

        logger.set_llm_deployment_failure_metrics(
            {
                "model": litellm_model_name,
                "exception": litellm.exceptions.RateLimitError("rate limited", "anthropic", litellm_model_name),
                "litellm_params": {
                    "custom_llm_provider": "anthropic",
                    "metadata": {"model_info": {"id": model_id}},
                },
                "standard_logging_object": _standard_logging_payload(litellm_model_name, model_id),
            }
        )
        assert single_deployment_state_sample().value == 1

        await cooldown()
        assert single_deployment_state_sample().value == 2
    finally:
        _clear_prometheus_registry()
