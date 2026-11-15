from collections.abc import Iterator
from datetime import datetime
from typing import Final

import pytest
from fastapi import HTTPException
from prometheus_client import REGISTRY
from prometheus_client.samples import Sample

import litellm
from litellm.constants import LITELLM_LOGGING_NO_UPSTREAM_LLM_CALL
from litellm.integrations.prometheus import PrometheusLogger
from litellm.proxy._types import ProxyErrorTypes, ProxyException, UserAPIKeyAuth
from tests.unit.integrations.prometheus_native_support import enable_native_prometheus, native_failure, series


@pytest.fixture(autouse=True)
def native_prometheus(monkeypatch):
    enable_native_prometheus(monkeypatch)


CALLED_THE_PROVIDER: Final = {"api_call_start_time": datetime(2026, 10, 19, 12, 0, 0)}


@pytest.fixture(scope="function")
def prometheus_logger():
    """Create a PrometheusLogger instance for testing."""
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        REGISTRY.unregister(collector)
    logger = PrometheusLogger()
    assert logger._native_store is not None
    return logger


class ExceptionWithCode:
    """Exception-like object with 'code' attribute (ProxyException pattern)."""

    def __init__(self, code):
        self.code = code


class TestAsyncHooks:
    """Test how async hook methods treat invalid API key requests."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exception",
        [
            HTTPException(
                status_code=401,
                detail="LiteLLM Virtual Key expected. Received=nota****tall, expected to start with 'sk-'.",
            ),
            ProxyException(
                message="Authentication Error, Invalid proxy server token passed.",
                type=ProxyErrorTypes.token_not_found_in_db,
                param="key",
                code=401,
            ),
        ],
    )
    async def test_post_call_failure_hook_counts_401_without_key_hash(self, prometheus_logger, exception):
        unauthenticated = UserAPIKeyAuth(request_route="/v1/chat/completions")
        unauthenticated.api_key = "notakeyatall"

        await prometheus_logger.async_post_call_failure_hook(
            request_data={"model": "test-model"},
            original_exception=exception,
            user_api_key_dict=unauthenticated,
        )

        failed = series(prometheus_logger.litellm_proxy_failed_requests_metric.name)
        assert len(failed) == 1
        failed_labels, failed_count = failed[0]
        assert failed_labels["exception_status"] == "401"
        assert failed_labels["hashed_api_key"] == "None"
        assert failed_labels["route"] == "/v1/chat/completions"
        assert failed_count == 1
        total = series(prometheus_logger.litellm_proxy_total_requests_metric.name)
        assert len(total) == 1
        assert total[0][0]["status_code"] == "401"
        assert total[0][1] == 1

    @pytest.mark.asyncio
    async def test_post_call_failure_hook_keeps_resolved_identity_labels_for_401(self, prometheus_logger):
        expired_key = UserAPIKeyAuth(
            api_key="sk-expired",
            key_alias="expired-alias",
            team_id="team-1",
        )
        exception = ProxyException(
            message="Authentication Error - Expired Key.",
            type=ProxyErrorTypes.expired_key,
            param="key",
            code=401,
        )

        await prometheus_logger.async_post_call_failure_hook(
            request_data={"model": "test-model"},
            original_exception=exception,
            user_api_key_dict=expired_key,
        )

        failed = series(prometheus_logger.litellm_proxy_failed_requests_metric.name)
        assert len(failed) == 1
        failed_labels = failed[0][0]
        assert failed_labels["exception_status"] == "401"
        assert failed_labels["hashed_api_key"] == "None"
        assert failed_labels["api_key_alias"] == "expired-alias"
        assert failed_labels["team"] == "team-1"

    @pytest.mark.asyncio
    async def test_log_failure_event_skips_401_rejected_at_the_proxy_gate(self, prometheus_logger):
        exception = ExceptionWithCode("401")
        kwargs = {
            "model": "test-model",
            "standard_logging_object": {
                "metadata": {
                    "user_api_key_hash": "test-key",
                    "user_api_key_user_id": "test-user",
                },
                "model_group": "test-model",
            },
            "exception": exception,
            "litellm_params": {},
            LITELLM_LOGGING_NO_UPSTREAM_LLM_CALL: True,
        }

        await native_failure(prometheus_logger, kwargs=kwargs, response_obj=None, start_time=None, end_time=None)

        assert series(prometheus_logger.litellm_llm_api_failed_requests_metric.name) == []
        assert series(prometheus_logger.litellm_deployment_failure_responses.name) == []
        assert series(prometheus_logger.litellm_deployment_total_requests.name) == []
        assert series(prometheus_logger.litellm_deployment_state.name) == []


def _provider_auth_failure_kwargs() -> dict[str, object]:
    exception: Final = litellm.AuthenticationError(
        message="Incorrect API key provided: sk-rotat*****",
        llm_provider="openai",
        model="gpt-4o-mini",
    )
    return {
        "model": "gpt-4o-mini",
        "standard_logging_object": {
            "metadata": {
                "user_api_key_hash": "test-key",
                "user_api_key_alias": "test-alias",
                "user_api_key_user_id": "test-user",
                "user_api_key_team_id": "test-team",
                "user_api_key_team_alias": "test-team-alias",
            },
            "model_group": "rotated-credential",
            "model_id": "deployment-1",
            "api_base": "https://api.openai.com/v1",
            "custom_llm_provider": "openai",
        },
        "exception": exception,
        "litellm_params": {"custom_llm_provider": "openai", "metadata": {}},
        **CALLED_THE_PROVIDER,
    }


def _deployment_entra_rejection_kwargs() -> dict[str, object]:
    from litellm.llms.azure.common_utils import EntraRejectionError

    reason: Final = "Failed to get Azure AD token: AADSTS7000215: Invalid client secret provided."
    try:
        raise litellm.AuthenticationError(message=reason, llm_provider="azure", model="gpt-4.1") from (
            EntraRejectionError(status_code=401, message=reason)
        )
    except litellm.AuthenticationError as raised:
        return {
            key: value for key, value in _provider_auth_failure_kwargs().items() if key not in CALLED_THE_PROVIDER
        } | {"exception": raised}


def _metric_values(name: str, model_id: str) -> list[float]:
    return [
        sample.value
        for sample in _registry_samples()
        if sample.name == name and sample.labels.get("model_id") == model_id
    ]


def _registry_samples() -> Iterator[Sample]:
    for metric in REGISTRY.collect():
        yield from metric.samples


class TestProviderAuthFailures:
    """A 401 raised by the provider is an upstream failure and is recorded."""

    @pytest.mark.asyncio
    async def test_log_failure_event_records_a_provider_401(self, prometheus_logger):
        await native_failure(
            prometheus_logger,
            kwargs=_provider_auth_failure_kwargs(),
            response_obj=None,
            start_time=None,
            end_time=None,
        )

        assert _metric_values("litellm_llm_api_failed_requests_metric_total", "deployment-1") == [1.0]
        assert _metric_values("litellm_deployment_failure_responses_total", "deployment-1") == [1.0]
        assert _metric_values("litellm_deployment_state", "deployment-1") == [1.0]

    @pytest.mark.asyncio
    async def test_log_failure_event_records_entra_rejecting_the_deployment_credential_before_the_call(
        self, prometheus_logger
    ):
        await native_failure(
            prometheus_logger,
            kwargs=_deployment_entra_rejection_kwargs(),
            response_obj=None,
            start_time=None,
            end_time=None,
        )

        assert _metric_values("litellm_llm_api_failed_requests_metric_total", "deployment-1") == [1.0]
        assert _metric_values("litellm_deployment_failure_responses_total", "deployment-1") == [1.0]

    @pytest.mark.asyncio
    async def test_deployment_failure_metrics_record_a_provider_401(self, prometheus_logger):
        await native_failure(prometheus_logger, _provider_auth_failure_kwargs(), None)

        failures: Final = [
            sample for sample in _registry_samples() if sample.name == "litellm_deployment_failure_responses_total"
        ]
        assert [
            (
                sample.labels["model_id"],
                sample.labels["exception_status"],
                sample.labels["api_provider"],
                sample.value,
            )
            for sample in failures
        ] == [("deployment-1", "401", "openai", 1.0)]
        assert _metric_values("litellm_deployment_total_requests_total", "deployment-1") == [1.0]
        assert _metric_values("litellm_deployment_state", "deployment-1") == [1.0]
