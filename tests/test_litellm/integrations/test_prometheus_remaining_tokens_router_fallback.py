"""
LIT-2719 — `litellm_remaining_tokens_metric` and
`litellm_remaining_requests_metric` only fired for providers that return
`x-ratelimit-remaining-*` response headers (OpenAI, Azure, Anthropic).

This guarded the gauges behind a provider-specific code path, so Bedrock and
Vertex deployments — which never populate those headers — silently produced no
data even when the proxy router had `tpm`/`rpm` configured.

`_async_set_router_remaining_metrics` adds a provider-agnostic fallback that
asks `Router.get_remaining_model_group_usage` for the same model_group and
emits the gauges with `configured_limit - current_usage`.

Tests cover:
- Bedrock fallback emits both gauges.
- Vertex AI fallback emits both gauges.
- Already-present headers short-circuit the router lookup entirely.
- Partial header coverage (only requests) still triggers the missing tokens
  gauge.
- llm_router unavailable / model_group missing / router raises → silent no-op.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prometheus_client import REGISTRY

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.integrations.prometheus import PrometheusLogger
from litellm.types.integrations.prometheus import UserAPIKeyLabelValues


@pytest.fixture(scope="function")
def prometheus_logger():
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        REGISTRY.unregister(collector)
    return PrometheusLogger()


def _build_payload(
    model_group: str = "bedrock-claude-group",
    custom_llm_provider: str = "bedrock",
    additional_headers: dict | None = None,
    total_tokens: int = 10,
):
    return {
        "model_group": model_group,
        "custom_llm_provider": custom_llm_provider,
        "model": "anthropic.claude-3-sonnet-20240229-v1:0",
        "model_id": "deployment-id-1",
        "api_base": "https://bedrock-runtime.us-east-1.amazonaws.com",
        "total_tokens": total_tokens,
        "hidden_params": {
            "additional_headers": additional_headers or {},
        },
        "metadata": {
            "user_api_key_hash": "test-key",
            "user_api_key_alias": None,
            "user_api_key_team_id": None,
            "user_api_key_team_alias": None,
        },
    }


def _enum_values(model_group: str = "bedrock-claude-group"):
    return UserAPIKeyLabelValues(
        end_user=None,
        hashed_api_key="test-key",
        api_key_alias=None,
        team=None,
        team_alias=None,
        requested_model=model_group,
        model_group=model_group,
        model_id="deployment-id-1",
        api_base="https://bedrock-runtime.us-east-1.amazonaws.com",
        api_provider="bedrock",
        litellm_model_name="anthropic.claude-3-sonnet-20240229-v1:0",
    )


class TestRouterFallbackEmitsForBedrock:
    @pytest.mark.asyncio
    async def test_should_emit_both_gauges_for_bedrock_when_router_has_limits(
        self, prometheus_logger
    ):
        payload = _build_payload(custom_llm_provider="bedrock")
        enum_values = _enum_values()

        fake_router = MagicMock()
        fake_router.get_remaining_model_group_usage = AsyncMock(
            return_value={
                "x-ratelimit-remaining-tokens": 75,
                "x-ratelimit-limit-tokens": 100,
                "x-ratelimit-remaining-requests": 9,
                "x-ratelimit-limit-requests": 10,
            }
        )

        prometheus_logger.litellm_remaining_tokens_metric = MagicMock()
        prometheus_logger.litellm_remaining_requests_metric = MagicMock()

        with patch("litellm.proxy.proxy_server.llm_router", fake_router, create=True):
            await prometheus_logger._async_set_router_remaining_metrics(
                standard_logging_payload=payload,
                enum_values=enum_values,
            )

        fake_router.get_remaining_model_group_usage.assert_awaited_once_with(
            "bedrock-claude-group"
        )
        # Router returns pre-decrement counter (current request not yet
        # counted by deployment_callback_on_success). Subtract in-flight
        # delta so the value is comparable to vendor-header post-decrement
        # semantics: 75 - 10 total_tokens = 65, 9 - 1 request = 8.
        prometheus_logger.litellm_remaining_tokens_metric.labels.assert_called_once()
        prometheus_logger.litellm_remaining_tokens_metric.labels().set.assert_called_once_with(
            65
        )
        prometheus_logger.litellm_remaining_requests_metric.labels.assert_called_once()
        prometheus_logger.litellm_remaining_requests_metric.labels().set.assert_called_once_with(
            8
        )


class TestRouterFallbackEmitsForVertex:
    @pytest.mark.asyncio
    async def test_should_emit_both_gauges_for_vertex_when_router_has_limits(
        self, prometheus_logger
    ):
        payload = _build_payload(
            model_group="vertex-gemini-group",
            custom_llm_provider="vertex_ai",
        )
        enum_values = _enum_values(model_group="vertex-gemini-group")

        fake_router = MagicMock()
        fake_router.get_remaining_model_group_usage = AsyncMock(
            return_value={
                "x-ratelimit-remaining-tokens": 12345,
                "x-ratelimit-remaining-requests": 50,
            }
        )

        prometheus_logger.litellm_remaining_tokens_metric = MagicMock()
        prometheus_logger.litellm_remaining_requests_metric = MagicMock()

        with patch("litellm.proxy.proxy_server.llm_router", fake_router, create=True):
            await prometheus_logger._async_set_router_remaining_metrics(
                standard_logging_payload=payload,
                enum_values=enum_values,
            )

        fake_router.get_remaining_model_group_usage.assert_awaited_once_with(
            "vertex-gemini-group"
        )
        prometheus_logger.litellm_remaining_tokens_metric.labels().set.assert_called_once_with(
            12335
        )
        prometheus_logger.litellm_remaining_requests_metric.labels().set.assert_called_once_with(
            49
        )


class TestExistingHeadersShortCircuit:
    @pytest.mark.asyncio
    async def test_should_skip_router_lookup_when_both_headers_already_present(
        self, prometheus_logger
    ):
        payload = _build_payload(
            additional_headers={
                "x_ratelimit_remaining_tokens": 999,
                "x_ratelimit_remaining_requests": 99,
            }
        )

        fake_router = MagicMock()
        fake_router.get_remaining_model_group_usage = AsyncMock()

        prometheus_logger.litellm_remaining_tokens_metric = MagicMock()
        prometheus_logger.litellm_remaining_requests_metric = MagicMock()

        with patch("litellm.proxy.proxy_server.llm_router", fake_router, create=True):
            await prometheus_logger._async_set_router_remaining_metrics(
                standard_logging_payload=payload,
                enum_values=_enum_values(),
            )

        fake_router.get_remaining_model_group_usage.assert_not_called()
        prometheus_logger.litellm_remaining_tokens_metric.labels.assert_not_called()
        prometheus_logger.litellm_remaining_requests_metric.labels.assert_not_called()

    @pytest.mark.asyncio
    async def test_should_only_fill_missing_dimension_when_one_header_present(
        self, prometheus_logger
    ):
        payload = _build_payload(
            additional_headers={
                "x_ratelimit_remaining_requests": 7,
            }
        )

        fake_router = MagicMock()
        fake_router.get_remaining_model_group_usage = AsyncMock(
            return_value={
                "x-ratelimit-remaining-tokens": 555,
                "x-ratelimit-remaining-requests": 999,
            }
        )

        prometheus_logger.litellm_remaining_tokens_metric = MagicMock()
        prometheus_logger.litellm_remaining_requests_metric = MagicMock()

        with patch("litellm.proxy.proxy_server.llm_router", fake_router, create=True):
            await prometheus_logger._async_set_router_remaining_metrics(
                standard_logging_payload=payload,
                enum_values=_enum_values(),
            )

        prometheus_logger.litellm_remaining_tokens_metric.labels().set.assert_called_once_with(
            545
        )
        prometheus_logger.litellm_remaining_requests_metric.labels.assert_not_called()


class TestRouterFallbackInFlightAdjustment:
    """
    Router's TPM/RPM counter is incremented by
    ``Router.deployment_callback_on_success``, which races with this
    prometheus callback in the success-log fan-out. Prometheus wins the
    race, so the router-derived value is pre-decrement for the current
    request, while vendor-header values (OpenAI/Anthropic/Azure) are
    post-decrement. The fallback subtracts the in-flight delta so both
    paths report comparable values.
    """

    @pytest.mark.asyncio
    async def test_should_subtract_total_tokens_and_one_request(
        self, prometheus_logger
    ):
        payload = _build_payload(total_tokens=42)
        fake_router = MagicMock()
        fake_router.get_remaining_model_group_usage = AsyncMock(
            return_value={
                "x-ratelimit-remaining-tokens": 1000,
                "x-ratelimit-remaining-requests": 100,
            }
        )

        prometheus_logger.litellm_remaining_tokens_metric = MagicMock()
        prometheus_logger.litellm_remaining_requests_metric = MagicMock()

        with patch("litellm.proxy.proxy_server.llm_router", fake_router, create=True):
            await prometheus_logger._async_set_router_remaining_metrics(
                standard_logging_payload=payload,
                enum_values=_enum_values(),
            )

        prometheus_logger.litellm_remaining_tokens_metric.labels().set.assert_called_once_with(
            958
        )
        prometheus_logger.litellm_remaining_requests_metric.labels().set.assert_called_once_with(
            99
        )

    @pytest.mark.asyncio
    async def test_should_treat_missing_total_tokens_as_zero(self, prometheus_logger):
        payload = _build_payload()
        payload.pop("total_tokens", None)

        fake_router = MagicMock()
        fake_router.get_remaining_model_group_usage = AsyncMock(
            return_value={
                "x-ratelimit-remaining-tokens": 1000,
                "x-ratelimit-remaining-requests": 100,
            }
        )

        prometheus_logger.litellm_remaining_tokens_metric = MagicMock()
        prometheus_logger.litellm_remaining_requests_metric = MagicMock()

        with patch("litellm.proxy.proxy_server.llm_router", fake_router, create=True):
            await prometheus_logger._async_set_router_remaining_metrics(
                standard_logging_payload=payload,
                enum_values=_enum_values(),
            )

        prometheus_logger.litellm_remaining_tokens_metric.labels().set.assert_called_once_with(
            1000
        )
        prometheus_logger.litellm_remaining_requests_metric.labels().set.assert_called_once_with(
            99
        )


class TestRouterFallbackDefensivePaths:
    @pytest.mark.asyncio
    async def test_should_noop_when_llm_router_is_none(self, prometheus_logger):
        payload = _build_payload()

        prometheus_logger.litellm_remaining_tokens_metric = MagicMock()
        prometheus_logger.litellm_remaining_requests_metric = MagicMock()

        with patch("litellm.proxy.proxy_server.llm_router", None, create=True):
            await prometheus_logger._async_set_router_remaining_metrics(
                standard_logging_payload=payload,
                enum_values=_enum_values(),
            )

        prometheus_logger.litellm_remaining_tokens_metric.labels.assert_not_called()
        prometheus_logger.litellm_remaining_requests_metric.labels.assert_not_called()

    @pytest.mark.asyncio
    async def test_should_noop_when_model_group_missing(self, prometheus_logger):
        payload = _build_payload()
        payload["model_group"] = None

        fake_router = MagicMock()
        fake_router.get_remaining_model_group_usage = AsyncMock()

        prometheus_logger.litellm_remaining_tokens_metric = MagicMock()
        prometheus_logger.litellm_remaining_requests_metric = MagicMock()

        with patch("litellm.proxy.proxy_server.llm_router", fake_router, create=True):
            await prometheus_logger._async_set_router_remaining_metrics(
                standard_logging_payload=payload,
                enum_values=_enum_values(),
            )

        fake_router.get_remaining_model_group_usage.assert_not_called()
        prometheus_logger.litellm_remaining_tokens_metric.labels.assert_not_called()

    @pytest.mark.asyncio
    async def test_should_noop_when_router_returns_empty_dict(self, prometheus_logger):
        payload = _build_payload()

        fake_router = MagicMock()
        fake_router.get_remaining_model_group_usage = AsyncMock(return_value={})

        prometheus_logger.litellm_remaining_tokens_metric = MagicMock()
        prometheus_logger.litellm_remaining_requests_metric = MagicMock()

        with patch("litellm.proxy.proxy_server.llm_router", fake_router, create=True):
            await prometheus_logger._async_set_router_remaining_metrics(
                standard_logging_payload=payload,
                enum_values=_enum_values(),
            )

        prometheus_logger.litellm_remaining_tokens_metric.labels.assert_not_called()
        prometheus_logger.litellm_remaining_requests_metric.labels.assert_not_called()

    @pytest.mark.asyncio
    async def test_should_swallow_router_exception(self, prometheus_logger):
        payload = _build_payload()

        fake_router = MagicMock()
        fake_router.get_remaining_model_group_usage = AsyncMock(
            side_effect=RuntimeError("router boom")
        )

        prometheus_logger.litellm_remaining_tokens_metric = MagicMock()
        prometheus_logger.litellm_remaining_requests_metric = MagicMock()

        with patch("litellm.proxy.proxy_server.llm_router", fake_router, create=True):
            # Must not raise.
            await prometheus_logger._async_set_router_remaining_metrics(
                standard_logging_payload=payload,
                enum_values=_enum_values(),
            )

        prometheus_logger.litellm_remaining_tokens_metric.labels.assert_not_called()
