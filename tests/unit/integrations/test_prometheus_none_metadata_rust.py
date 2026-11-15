from datetime import datetime
from typing import Final

import pytest
from prometheus_client import REGISTRY

from litellm.integrations.prometheus import PrometheusLogger
from tests.unit.integrations.prometheus_native_support import enable_native_prometheus, native_success, series


@pytest.fixture(autouse=True)
def native_prometheus(monkeypatch):
    enable_native_prometheus(monkeypatch)


@pytest.fixture(scope="function")
def prometheus_logger():
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        REGISTRY.unregister(collector)
    return PrometheusLogger()


def _standard_logging_object() -> dict[str, object]:
    return {
        "id": "none-metadata",
        "call_type": "completion",
        "status": "success",
        "api_base": "https://api.openai.com",
        "hidden_params": {
            "additional_headers": None,
            "litellm_overhead_time_ms": None,
        },
        "metadata": {
            "user_api_key_hash": "test-key",
            "user_api_key_alias": None,
            "user_api_key_team_id": None,
            "user_api_key_team_alias": None,
            "user_api_key_user_id": None,
            "user_api_key_user_email": None,
            "user_api_key_org_id": None,
            "user_api_key_org_alias": None,
            "user_api_key_end_user_id": None,
            "requester_metadata": None,
            "usage_object": None,
        },
        "model": "gpt-4o",
        "model_group": "gpt-4o",
        "model_id": None,
        "custom_llm_provider": "openai",
        "response_cost": 0.001,
        "prompt_tokens": 5,
        "completion_tokens": 10,
        "total_tokens": 15,
        "request_tags": [],
        "end_user": None,
        "cache_hit": False,
        "stream": False,
        "model_parameters": {},
        "startTime": 1.0,
        "endTime": 2.0,
        "completionStartTime": 2.0,
        "response": {"id": "chatcmpl-1"},
    }


def _request_kwargs(litellm_params: dict[str, object] | None) -> dict[str, object]:
    now: Final = datetime.now()
    kwargs: Final[dict[str, object]] = {
        "model": "gpt-4o",
        "standard_logging_object": _standard_logging_object(),
        "start_time": now,
        "end_time": now,
    }
    if litellm_params is not None:
        kwargs["litellm_params"] = litellm_params
    return kwargs


def _deployment_success_labelsets() -> list[dict[str, str]]:
    return [labels for labels, value in series("litellm_deployment_success_responses_total") if value == 1.0]


class TestNoneMetadataHandling:
    @pytest.mark.asyncio
    async def test_set_llm_deployment_success_metrics_with_none_metadata(self, prometheus_logger):
        request_kwargs = _request_kwargs({"metadata": None, "custom_llm_provider": "openai"})

        await native_success(prometheus_logger, request_kwargs, None)

        assert len(_deployment_success_labelsets()) == 1

    @pytest.mark.asyncio
    async def test_set_llm_deployment_success_metrics_with_missing_litellm_params(self, prometheus_logger):
        request_kwargs = _request_kwargs(None)

        await native_success(prometheus_logger, request_kwargs, None)

        assert len(_deployment_success_labelsets()) == 1

    @pytest.mark.asyncio
    async def test_set_llm_deployment_success_metrics_with_litellm_metadata_key(self, prometheus_logger):
        request_kwargs = _request_kwargs(
            {
                "metadata": None,
                "litellm_metadata": {"model_info": {"id": "test-model-id"}},
                "custom_llm_provider": "openai",
            }
        )

        await native_success(prometheus_logger, request_kwargs, None)

        assert len(_deployment_success_labelsets()) == 1
        assert [labels["model_id"] for labels, _ in series("litellm_deployment_state")] == ["test-model-id"]
