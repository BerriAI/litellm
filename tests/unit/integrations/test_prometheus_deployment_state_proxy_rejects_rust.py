from typing import Final

import pytest
from fastapi import HTTPException
from prometheus_client import REGISTRY

from litellm.constants import PROXY_REJECTED_BEFORE_ROUTING_KEY
from litellm.exceptions import GuardrailRaisedException, MalformedRequestFieldError
from litellm.integrations.prometheus import PrometheusLogger
from litellm.proxy._types import ProxyException
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
from tests.unit.integrations.prometheus_native_support import enable_native_prometheus, native_failure, series


@pytest.fixture(autouse=True)
def native_prometheus(monkeypatch):
    enable_native_prometheus(monkeypatch)


@pytest.fixture(autouse=True)
def cleanup_prometheus_registry():
    for collector in list(REGISTRY._collector_to_names.keys()):
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass

    yield

    for collector in list(REGISTRY._collector_to_names.keys()):
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass


def _attributed_failure_kwargs(exception: Exception, rejected_before_routing: bool) -> dict:
    return {
        "model": "openai/gpt-4.1",
        "litellm_params": {
            "custom_llm_provider": "openai",
            "metadata": {"model_info": {"id": "dep-1"}, "model_group": "internal-model"},
            **({PROXY_REJECTED_BEFORE_ROUTING_KEY: True} if rejected_before_routing else {}),
        },
        "standard_logging_object": {
            "model_id": "dep-1",
            "model_group": "internal-model",
            "api_base": "https://api.openai.com",
            "metadata": {
                "user_api_key_hash": None,
                "user_api_key_alias": None,
                "user_api_key_team_id": None,
                "user_api_key_team_alias": None,
                "user_api_key_user_id": None,
                "user_api_key_user_email": None,
            },
        },
        "exception": exception,
        "start_time": None,
        "end_time": None,
    }


def _model_id_values(metric) -> set[str]:
    return {labels["model_id"] for labels, _ in series(metric.name)}


class _ProviderError(Exception):
    status_code = 500


@pytest.mark.parametrize(
    "rejection",
    [
        HTTPException(status_code=403, detail="guardrail blocked"),
        ProxyException(message="budget exceeded", type="budget_exceeded", param=None, code=400),
        ProxyRateLimitError(detail={"error": "key rpm limit"}),
        GuardrailRaisedException(guardrail_name="pii", message="blocked", status_code=403),
    ],
    ids=["http_exception", "proxy_exception", "proxy_rate_limit", "guardrail_raised"],
)
@pytest.mark.asyncio
async def test_attributed_proxy_reject_leaves_deployment_healthy(rejection: Exception):
    logger = PrometheusLogger()

    await native_failure(logger, _attributed_failure_kwargs(rejection, rejected_before_routing=True), None)

    assert series(logger.litellm_deployment_state.name) == []
    assert _model_id_values(logger.litellm_deployment_failure_responses) == {""}
    assert _model_id_values(logger.litellm_deployment_total_requests) == {""}


@pytest.mark.parametrize(
    "failure",
    [
        _ProviderError("upstream 500"),
        GuardrailRaisedException(guardrail_name="pii", message="response blocked", status_code=400),
    ],
    ids=["provider_error", "post_call_guardrail"],
)
@pytest.mark.asyncio
async def test_failure_after_routing_still_marks_deployment_partial_outage(failure: Exception):
    logger = PrometheusLogger()

    await native_failure(logger, _attributed_failure_kwargs(failure, rejected_before_routing=False), None)

    assert _model_id_values(logger.litellm_deployment_state) == {"dep-1"}
    assert _model_id_values(logger.litellm_deployment_failure_responses) == {"dep-1"}


@pytest.mark.asyncio
async def test_malformed_field_rejection_after_routing_leaves_deployment_healthy() -> None:
    logger: Final = PrometheusLogger()
    rejection: Final = MalformedRequestFieldError(
        message="tools: Input should be a valid list", model="internal-model", llm_provider="openai"
    )

    await native_failure(logger, _attributed_failure_kwargs(rejection, rejected_before_routing=False), None)

    assert series(logger.litellm_deployment_state.name) == []
    assert _model_id_values(logger.litellm_deployment_failure_responses) == {""}
