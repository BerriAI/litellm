"""
Unit tests for Prometheus invalid API key request filtering.

Tests the 401 detection helpers, that LLM-level metrics skip invalid API key
requests rejected at the proxy gate while recording a provider's 401, and that
the proxy-level failed request counter still records them.
"""

from collections.abc import Iterator
from datetime import datetime
from typing import Final
from unittest.mock import Mock, patch

import pytest
from fastapi import HTTPException
from prometheus_client import REGISTRY
from prometheus_client.samples import Sample

import litellm
from litellm.constants import LITELLM_LOGGING_NO_UPSTREAM_LLM_CALL
from litellm.integrations.prometheus import PrometheusLogger
from litellm.llms.github_copilot.per_user_auth import GithubCopilotUserSession
from litellm.proxy._types import ProxyErrorTypes, ProxyException, UserAPIKeyAuth


PROVIDER_CREDENTIAL_FIELDS = [
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
    "aws_web_identity_token",
    "azure_ad_token",
    "azure_password",
    "azure_scope",
    "azure_username",
    "client_id",
    "client_secret",
    "gigachat_access_token",
    "oci_fingerprint",
    "oci_key",
    "oci_key_file",
    "oci_tenancy",
    "oci_user",
    "tenant_id",
    "token",
    "user_config",
    "vertex_ai_credentials",
    "vertex_credentials",
    "watsonx_credentials",
    "wx_credentials",
    "zen_api_key",
]
PROVIDER_ENDPOINT_FIELDS = [
    "aws_bedrock_runtime_endpoint",
    "aws_sts_endpoint",
    "deployment_url",
    "s3_endpoint_url",
    "sagemaker_base_url",
]
CALLED_THE_PROVIDER: Final = {"api_call_start_time": datetime(2026, 10, 19, 12, 0, 0)}


@pytest.fixture(scope="function")
def prometheus_logger():
    """Create a PrometheusLogger instance for testing."""
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        REGISTRY.unregister(collector)
    return PrometheusLogger()


class ExceptionWithCode:
    """Exception-like object with 'code' attribute (ProxyException pattern)."""

    def __init__(self, code):
        self.code = code


class ExceptionWithStatusCode:
    """Exception-like object with 'status_code' attribute."""

    def __init__(self, status_code):
        self.status_code = status_code


class TestExtractStatusCode:
    """Test status code extraction from various sources."""

    @pytest.mark.parametrize(
        "exception_class,code_value,expected",
        [
            (ExceptionWithCode, "401", 401),
            (ExceptionWithStatusCode, 401, 401),
        ],
    )
    def test_extract_from_exception(self, prometheus_logger, exception_class, code_value, expected):
        exception = exception_class(code_value)
        assert prometheus_logger._extract_status_code(exception=exception) == expected

    def test_extract_from_kwargs(self, prometheus_logger):
        exception = ExceptionWithCode("401")
        assert prometheus_logger._extract_status_code(kwargs={"exception": exception}) == 401

    def test_extract_from_enum_values(self, prometheus_logger):
        enum_values = Mock(status_code="401")
        assert prometheus_logger._extract_status_code(enum_values=enum_values) == 401


class TestInvalidAPIKeyDetection:
    """Test invalid API key request detection logic."""

    @pytest.mark.parametrize(
        "status_code,expected",
        [
            (401, True),
            (200, False),
            (500, False),
            (None, False),
        ],
    )
    def test_status_code_detection(self, prometheus_logger, status_code, expected):
        assert prometheus_logger._is_invalid_api_key_request(status_code=status_code) == expected

    def test_auth_error_message_detection(self, prometheus_logger):
        exception = AssertionError(
            "LiteLLM Virtual Key expected. Received=invalid-key-12345, expected to start with 'sk-'."
        )
        assert prometheus_logger._is_invalid_api_key_request(status_code=None, exception=exception) is True

    def test_non_auth_exception_not_detected(self, prometheus_logger):
        exception = ValueError("Some other error")
        assert prometheus_logger._is_invalid_api_key_request(status_code=None, exception=exception) is False


class TestSkipMetricsValidation:
    """Test high-level validation method that orchestrates detection and extraction."""

    @pytest.mark.parametrize(
        "exception,proxy_gate,expected",
        [
            (ExceptionWithCode("401"), True, True),
            (AssertionError("expected to start with 'sk-'"), True, True),
            (ExceptionWithStatusCode(429), True, False),
            (ExceptionWithCode("401"), False, True),
            (AssertionError("expected to start with 'sk-'"), False, True),
            (
                litellm.AuthenticationError(
                    message="Incorrect API key provided: sk-rotat*****",
                    llm_provider="openai",
                    model="gpt-4o-mini",
                ),
                False,
                False,
            ),
            (
                litellm.AuthenticationError(
                    message="API key not valid. Please pass a valid API key.",
                    llm_provider="gemini",
                    model="gemini-2.5-flash",
                ),
                False,
                False,
            ),
            (
                litellm.AuthenticationError(
                    message="Authentication Error, Invalid proxy server token passed.",
                    llm_provider="openai",
                    model="gpt-4o-mini",
                ),
                True,
                True,
            ),
            (
                litellm.CallerCredentialAuthenticationError(
                    message="GitHub Copilot connection is missing for this user",
                    llm_provider="github_copilot",
                    model="gpt-4o",
                ),
                False,
                True,
            ),
            (
                HTTPException(status_code=401, detail="caller did not present a token"),
                False,
                True,
            ),
            (
                litellm.BadRequestError(
                    message="API key not valid. Please pass a valid API key.",
                    llm_provider="gemini",
                    model="gemini-2.5-flash",
                ),
                False,
                False,
            ),
        ],
        ids=[
            "proxy-gate-401",
            "proxy-gate-auth-message",
            "proxy-gate-429",
            "unmarked-proxy-401",
            "unmarked-auth-message",
            "provider-401",
            "provider-auth-message",
            "proxy-gate-litellm-auth-error",
            "caller-credential-401",
            "unmarked-http-exception-401",
            "provider-400-auth-message",
        ],
    )
    def test_skip_except_for_provider_errors_the_proxy_did_not_raise(
        self, prometheus_logger, exception, proxy_gate, expected
    ):
        kwargs: Final[dict[str, object]] = {"exception": exception, **CALLED_THE_PROVIDER}
        if proxy_gate:
            kwargs[LITELLM_LOGGING_NO_UPSTREAM_LLM_CALL] = True
        assert prometheus_logger._should_skip_metrics_for_invalid_key(kwargs=kwargs) is expected

    @pytest.mark.parametrize(
        "logged_call,expected",
        [
            ({"litellm_params": {"proxy_server_request": {"caller_credential_fields": ("api_key",)}}}, True),
            (
                {
                    "litellm_params": {
                        "github_copilot_user_session": GithubCopilotUserSession(
                            token="caller-session-token", api_base="https://api.githubcopilot.com"
                        )
                    }
                },
                True,
            ),
            ({"standard_logging_object": {"metadata": {"used_client_oauth_token": True}}}, True),
            ({"litellm_params": {"proxy_server_request": {"caller_credential_fields": ("headers",)}}}, True),
            (
                {
                    "litellm_params": {
                        "proxy_server_request": {"caller_credential_fields": ("provider_specific_header",)}
                    }
                },
                True,
            ),
            ({"litellm_params": {"proxy_server_request": {"caller_credential_fields": ()}}}, False),
            ({"litellm_params": {"proxy_server_request": {"caller_credential_fields": "api_key"}}}, False),
            ({"standard_logging_object": {"metadata": {"used_client_oauth_token": False}}}, False),
            ({"litellm_params": {"proxy_server_request": {"body": {"azure_ad_token": "caller-token"}}}}, True),
            (
                {
                    "litellm_params": {
                        "proxy_server_request": {"body": {"extra_body": {"azure_ad_token": "caller-token"}}}
                    }
                },
                True,
            ),
            (
                {
                    "litellm_params": {
                        "azure_ad_token": "deployment-token",
                        "proxy_server_request": {"body": {"model": "azure-deployment"}},
                    }
                },
                False,
            ),
            ({"litellm_params": {"proxy_server_request": {"body": {"extra_body": "azure_ad_token"}}}}, False),
            (
                {
                    "litellm_params": {
                        "proxy_server_request": {"body": {"litellm_credential_name": "caller-credential"}}
                    }
                },
                True,
            ),
            (
                {
                    "litellm_params": {
                        "litellm_credential_name": "deployment-credential",
                        "proxy_server_request": {"body": {"model": "openai-deployment"}},
                    }
                },
                False,
            ),
            ({"litellm_params": {"proxy_server_request": {"caller_credential_fields": ("extra_headers",)}}}, True),
            ({"litellm_params": {"proxy_server_request": {"caller_credential_fields": ("organization",)}}}, True),
            (
                {
                    "litellm_params": {
                        "proxy_server_request": {"credential_fields": ("headers",), "caller_credential_fields": ()}
                    }
                },
                False,
            ),
            (
                {
                    "litellm_params": {
                        "proxy_server_request": {"body": {"extra_headers": {"Authorization": "Bearer sk-caller"}}}
                    }
                },
                False,
            ),
        ],
        ids=[
            "caller-supplied-api-key",
            "caller-copilot-session",
            "caller-anthropic-oauth-token",
            "caller-supplied-headers",
            "caller-supplied-provider-specific-header",
            "no-caller-credential-fields",
            "malformed-credential-fields",
            "deployment-anthropic-key",
            "caller-supplied-azure-ad-token",
            "caller-supplied-azure-ad-token-in-extra-body",
            "deployment-azure-ad-token",
            "malformed-extra-body",
            "caller-named-credential",
            "deployment-named-credential",
            "caller-supplied-extra-headers",
            "caller-supplied-organization",
            "headers-that-cannot-change-provider-auth",
            "extra-headers-not-stamped-by-the-proxy",
        ],
    )
    def test_skip_a_provider_401_on_a_credential_the_caller_owns(self, prometheus_logger, logged_call, expected):
        exception: Final = litellm.AuthenticationError(
            message="Incorrect API key provided: sk-calle*****",
            llm_provider="openai",
            model="gpt-4o-mini",
        )
        kwargs: Final = {"exception": exception, **CALLED_THE_PROVIDER, **logged_call}
        assert prometheus_logger._should_skip_metrics_for_invalid_key(kwargs=kwargs) is expected

    @pytest.mark.parametrize(
        "logged_call,expected",
        [(CALLED_THE_PROVIDER, False), ({}, True), ({"api_call_start_time": None}, True)],
        ids=["provider-401", "401-before-the-provider-call", "blank-provider-call-start"],
    )
    def test_skip_a_401_raised_before_the_provider_was_called(self, prometheus_logger, logged_call, expected):
        exception: Final = litellm.AuthenticationError(
            message="Missing Anthropic API Key - A call is being made to anthropic but no key is set",
            llm_provider="anthropic",
            model="claude-haiku-4-5",
        )
        kwargs: Final = {"exception": exception, **logged_call}
        assert prometheus_logger._should_skip_metrics_for_invalid_key(kwargs=kwargs) is expected

    @pytest.mark.parametrize("field", PROVIDER_CREDENTIAL_FIELDS + PROVIDER_ENDPOINT_FIELDS)
    @pytest.mark.parametrize("inside_extra_body", [False, True], ids=["body", "extra-body"])
    def test_skip_a_provider_401_on_a_provider_credential_field_the_caller_sent(
        self, prometheus_logger, field, inside_extra_body
    ):
        sent: Final = {field: "caller-credential"}
        body: Final = {"model": "shared-deployment", **({"extra_body": sent} if inside_extra_body else sent)}
        kwargs: Final = {
            "exception": litellm.AuthenticationError(message="Unauthorized", llm_provider="azure", model="gpt-4o"),
            "litellm_params": {"proxy_server_request": {"body": body}},
            **CALLED_THE_PROVIDER,
        }
        assert prometheus_logger._should_skip_metrics_for_invalid_key(kwargs=kwargs) is True

    @pytest.mark.parametrize("field", PROVIDER_CREDENTIAL_FIELDS + PROVIDER_ENDPOINT_FIELDS)
    def test_count_a_provider_401_on_a_provider_credential_field_the_deployment_owns(self, prometheus_logger, field):
        kwargs: Final = {
            "exception": litellm.AuthenticationError(message="Unauthorized", llm_provider="azure", model="gpt-4o"),
            **CALLED_THE_PROVIDER,
            "litellm_params": {
                field: "deployment-credential",
                "proxy_server_request": {"body": {"model": "shared-deployment"}},
            },
        }
        assert prometheus_logger._should_skip_metrics_for_invalid_key(kwargs=kwargs) is False

    @pytest.mark.parametrize(
        "token_exchange_endpoint,expected",
        [("https://login.example.com/oauth2/v2.0/token", True), ("", False), (None, False)],
        ids=["token-exchange-deployment", "blank-endpoint", "no-endpoint"],
    )
    def test_skip_a_provider_401_on_a_token_exchange_deployment(
        self, prometheus_logger, token_exchange_endpoint, expected
    ):
        kwargs: Final = {
            "exception": litellm.AuthenticationError(
                message="Unauthorized", llm_provider="microsoft_365_copilot", model="copilot"
            ),
            "litellm_params": {"token_exchange_endpoint": token_exchange_endpoint, "client_secret": "deployment"},
            **CALLED_THE_PROVIDER,
        }
        assert prometheus_logger._should_skip_metrics_for_invalid_key(kwargs=kwargs) is expected

    def test_no_skip_for_valid_request(self, prometheus_logger):
        assert prometheus_logger._should_skip_metrics_for_invalid_key() is False


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

        with (
            patch.object(prometheus_logger, "litellm_proxy_failed_requests_metric") as mock_failed,
            patch.object(prometheus_logger, "litellm_proxy_total_requests_metric") as mock_total,
        ):
            await prometheus_logger.async_post_call_failure_hook(
                request_data={"model": "test-model"},
                original_exception=exception,
                user_api_key_dict=unauthenticated,
            )

        failed_labels = mock_failed.labels.call_args.kwargs
        assert failed_labels["exception_status"] == "401"
        assert failed_labels["hashed_api_key"] is None
        assert failed_labels["route"] == "/v1/chat/completions"
        mock_failed.labels.return_value.inc.assert_called_once()
        assert mock_total.labels.call_args.kwargs["status_code"] == "401"
        mock_total.labels.return_value.inc.assert_called_once()

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

        with patch.object(prometheus_logger, "litellm_proxy_failed_requests_metric") as mock_failed:
            await prometheus_logger.async_post_call_failure_hook(
                request_data={"model": "test-model"},
                original_exception=exception,
                user_api_key_dict=expired_key,
            )

        failed_labels = mock_failed.labels.call_args.kwargs
        assert failed_labels["exception_status"] == "401"
        assert failed_labels["hashed_api_key"] is None
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

        with (
            patch.object(prometheus_logger, "litellm_llm_api_failed_requests_metric") as mock_failed,
            patch.object(prometheus_logger, "set_llm_deployment_failure_metrics") as mock_deployment,
        ):
            await prometheus_logger.async_log_failure_event(
                kwargs=kwargs, response_obj=None, start_time=None, end_time=None
            )

            mock_failed.labels.assert_not_called()
            mock_deployment.assert_not_called()


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
        await prometheus_logger.async_log_failure_event(
            kwargs=_provider_auth_failure_kwargs(),
            response_obj=None,
            start_time=None,
            end_time=None,
        )

        assert _metric_values("litellm_llm_api_failed_requests_metric_total", "deployment-1") == [1.0]
        assert _metric_values("litellm_deployment_failure_responses_total", "deployment-1") == [1.0]
        assert _metric_values("litellm_deployment_state", "deployment-1") == [1.0]

    def test_deployment_failure_metrics_record_a_provider_401(self, prometheus_logger):
        prometheus_logger.set_llm_deployment_failure_metrics(_provider_auth_failure_kwargs())

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
