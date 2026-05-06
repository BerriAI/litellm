from datetime import datetime, timedelta

import pytest
from prometheus_client import REGISTRY

import litellm
from litellm.integrations.prometheus import PrometheusLogger
from litellm.types.utils import (
    StandardLoggingHiddenParams,
    StandardLoggingMetadata,
    StandardLoggingModelInformation,
    StandardLoggingPayload,
)


@pytest.fixture
def prometheus_logger() -> PrometheusLogger:
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        REGISTRY.unregister(collector)
    return PrometheusLogger()


def _create_standard_logging_payload(usage_object: dict) -> StandardLoggingPayload:
    return StandardLoggingPayload(
        id="test_id",
        trace_id="test_trace_id",
        litellm_call_id="test_call_id",
        call_type="completion",
        stream=False,
        response_cost=0.1,
        cost_breakdown=None,
        response_cost_failure_debug_info=None,
        status="success",
        status_fields={},
        total_tokens=30,
        prompt_tokens=20,
        completion_tokens=10,
        startTime=1234567890.0,
        endTime=1234567891.0,
        completionStartTime=1234567890.5,
        response_time=1.0,
        model_map_information=StandardLoggingModelInformation(
            model_map_key="claude-sonnet-4", model_map_value=None
        ),
        model="claude-sonnet-4",
        model_id="model-123",
        model_group="anthropic-claude",
        custom_llm_provider="anthropic",
        api_base="https://api.anthropic.com",
        metadata=StandardLoggingMetadata(
            user_api_key_hash="test_hash",
            user_api_key_alias="test_alias",
            user_api_key_team_id="test_team",
            user_api_key_user_id="test_user",
            user_api_key_user_email="test@example.com",
            user_api_key_team_alias="test_team_alias",
            user_api_key_org_id=None,
            user_api_key_org_alias=None,
            spend_logs_metadata=None,
            requester_ip_address="127.0.0.1",
            requester_metadata=None,
            user_api_key_end_user_id="test_end_user",
            usage_object=usage_object,
        ),
        cache_hit=None,
        cache_key=None,
        saved_cache_cost=0.0,
        request_tags=[],
        end_user=None,
        requester_ip_address="127.0.0.1",
        user_agent=None,
        messages=[{"role": "user", "content": "What changed today?"}],
        response={"choices": [{"message": {"content": "Search summary"}}]},
        error_str=None,
        error_information=None,
        model_parameters={"stream": False},
        hidden_params=StandardLoggingHiddenParams(
            model_id="model-123",
            cache_key=None,
            api_base="https://api.anthropic.com",
            response_cost="0.1",
            litellm_overhead_time_ms=None,
            additional_headers=None,
            batch_models=None,
            litellm_model_name="claude-sonnet-4",
            usage_object=usage_object,
        ),
        guardrail_information=None,
        standard_built_in_tools_params=None,
    )


@pytest.mark.asyncio
async def test_prometheus_tracks_web_search_requests_from_server_tool_use(
    prometheus_logger,
):
    standard_logging_payload = _create_standard_logging_payload(
        usage_object={"server_tool_use": {"web_search_requests": 3}}
    )
    kwargs = {
        "model": "claude-sonnet-4",
        "litellm_params": {
            "metadata": {
                "user_api_key": "test_key",
                "user_api_key_user_id": "test_user",
                "user_api_key_team_id": "test_team",
                "user_api_key_end_user_id": "test_end_user",
            }
        },
        "start_time": datetime.now(),
        "completion_start_time": datetime.now(),
        "api_call_start_time": datetime.now(),
        "end_time": datetime.now() + timedelta(seconds=1),
        "standard_logging_object": standard_logging_payload,
    }

    await prometheus_logger.async_log_success_event(
        kwargs,
        response_obj={},
        start_time=kwargs["start_time"],
        end_time=kwargs["end_time"],
    )

    samples = {
        sample.name: sample.value
        for metric in REGISTRY.collect()
        for sample in metric.samples
    }

    assert samples["litellm_web_search_requests_metric_total"] == 3.0
