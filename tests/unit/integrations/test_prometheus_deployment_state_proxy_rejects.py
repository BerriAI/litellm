"""
LIT-7701 attributes a pre_call_hook rejection's failure log to the model
group's single deployment (``model_id`` and ``custom_llm_provider``) and flags
it with ``PROXY_REJECTED_BEFORE_ROUTING_KEY``. The deployment health metrics
must keep treating such rejects as "no deployment picked": a key rate limit or
guardrail block never reached the deployment, so it must not flip
``litellm_deployment_state`` to partial outage or count as a deployment failure
response. A failure raised after the router picked a deployment (a post-call
guardrail block, a provider error) carries no flag and keeps its deployment labels.
"""

import pytest
from fastapi import HTTPException
from prometheus_client import REGISTRY

from litellm.constants import PROXY_REJECTED_BEFORE_ROUTING_KEY
from litellm.exceptions import GuardrailRaisedException
from litellm.integrations.prometheus import PrometheusLogger
from litellm.proxy._types import ProxyException
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError


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
            "metadata": {},
        },
        "exception": exception,
    }


def _model_id_values(metric) -> set[str]:
    index = metric._labelnames.index("model_id")
    return {sample_key[index] for sample_key in metric._metrics}


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
def test_attributed_proxy_reject_leaves_deployment_healthy(rejection: Exception):
    logger = PrometheusLogger()

    logger.set_llm_deployment_failure_metrics(_attributed_failure_kwargs(rejection, rejected_before_routing=True))

    assert logger.litellm_deployment_state._metrics == {}
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
def test_failure_after_routing_still_marks_deployment_partial_outage(failure: Exception):
    logger = PrometheusLogger()

    logger.set_llm_deployment_failure_metrics(_attributed_failure_kwargs(failure, rejected_before_routing=False))

    assert _model_id_values(logger.litellm_deployment_state) == {"dep-1"}
    assert _model_id_values(logger.litellm_deployment_failure_responses) == {"dep-1"}
