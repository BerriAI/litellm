from datetime import datetime

import pytest
from prometheus_client import REGISTRY

from litellm.integrations.prometheus import PrometheusLogger
from litellm.proxy._types import UserAPIKeyAuth
from tests.unit.integrations.prometheus_native_support import (
    enable_native_prometheus,
    labelsets,
    native_failure,
    native_success,
    standard_payload,
)


@pytest.fixture(autouse=True)
def native_prometheus(monkeypatch):
    enable_native_prometheus(monkeypatch)


@pytest.fixture
def logger():
    for collector in list(REGISTRY._collector_to_names.keys()):
        REGISTRY.unregister(collector)
    return PrometheusLogger()


def _has_client(metric: str, client_ip: str, user_agent: str) -> bool:
    return any(
        labels.get("client_ip") == client_ip and labels.get("user_agent") == user_agent for labels in labelsets(metric)
    )


@pytest.mark.asyncio
async def test_async_post_call_failure_hook_includes_client_ip_user_agent(logger):
    request_data = {
        "model": "gpt-4",
        "metadata": {
            "requester_ip_address": "127.0.0.1",
            "user_agent": "test-agent",
        },
    }
    user_api_key_dict = UserAPIKeyAuth(token="test_token")
    original_exception = Exception("Test exception")

    await logger.async_post_call_failure_hook(
        request_data=request_data,
        original_exception=original_exception,
        user_api_key_dict=user_api_key_dict,
    )

    assert _has_client("litellm_proxy_failed_requests_metric_total", "127.0.0.1", "test-agent")
    assert _has_client("litellm_proxy_total_requests_metric_total", "127.0.0.1", "test-agent")


@pytest.mark.asyncio
async def test_async_post_call_success_hook_includes_client_ip_user_agent(logger):
    now = datetime.now()
    kwargs = {
        "model": "gpt-4",
        "litellm_params": {"metadata": {}},
        "start_time": now,
        "end_time": now,
        "standard_logging_object": standard_payload(
            metadata={
                "user_api_key_user_id": "user_1",
                "user_api_key_hash": "hash_1",
                "user_api_key_alias": "alias_1",
                "user_api_key_team_id": "team_1",
                "user_api_key_team_alias": "team_alias_1",
                "user_api_key_org_id": None,
                "user_api_key_org_alias": None,
                "user_api_key_user_email": "test@example.com",
                "user_api_key_request_route": "/chat/completions",
                "requester_ip_address": "192.168.1.1",
                "user_agent": "success-agent",
            },
            model_group="gpt-4",
            model_id="model_1",
            api_base="http://api.base",
            custom_llm_provider="openai",
            completion_tokens=10,
            total_tokens=20,
            response_cost=0.01,
            request_tags=[],
        ),
    }

    await native_success(logger, kwargs, None)

    assert _has_client("litellm_proxy_total_requests_metric_total", "192.168.1.1", "success-agent")


@pytest.mark.asyncio
async def test_set_llm_deployment_failure_metrics_includes_client_ip_user_agent(logger):
    now = datetime.now()
    request_kwargs = {
        "model": "gpt-4",
        "start_time": now,
        "end_time": now,
        "standard_logging_object": standard_payload(
            metadata={
                "requester_ip_address": "10.0.0.1",
                "user_agent": "failure-deployment",
                "user_api_key_team_id": "team_1",
                "user_api_key_team_alias": "team_alias_1",
                "user_api_key_alias": "key_alias_1",
            },
            status="failure",
            model_group="group_1",
            api_base="http://api.base",
            model_id="model_1",
        ),
        "litellm_params": {},
        "exception": Exception("Deployment failure"),
    }

    await native_failure(logger, request_kwargs, None)

    assert _has_client("litellm_deployment_failure_responses_total", "10.0.0.1", "failure-deployment")
    assert _has_client("litellm_deployment_total_requests_total", "10.0.0.1", "failure-deployment")
