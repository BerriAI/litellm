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

from collections.abc import Iterator
from datetime import datetime
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prometheus_client import REGISTRY
from prometheus_client.samples import Sample


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
):
    return {
        "model_group": model_group,
        "custom_llm_provider": custom_llm_provider,
        "model": "anthropic.claude-3-sonnet-20240229-v1:0",
        "model_id": "deployment-id-1",
        "api_base": "https://bedrock-runtime.us-east-1.amazonaws.com",
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
        prometheus_logger.litellm_remaining_tokens_metric.labels.assert_called_once()
        prometheus_logger.litellm_remaining_tokens_metric.labels().set.assert_called_once_with(
            75
        )
        prometheus_logger.litellm_remaining_requests_metric.labels.assert_called_once()
        prometheus_logger.litellm_remaining_requests_metric.labels().set.assert_called_once_with(
            9
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
            12345
        )
        prometheus_logger.litellm_remaining_requests_metric.labels().set.assert_called_once_with(
            50
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
            555
        )
        prometheus_logger.litellm_remaining_requests_metric.labels.assert_not_called()


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


def _registry_samples() -> Iterator[Sample]:
    for metric in REGISTRY.collect():
        yield from metric.samples


def _sample_values(sample_name: str) -> tuple[float, ...]:
    return tuple(sample.value for sample in _registry_samples() if sample.name == sample_name)


@pytest.mark.parametrize("remaining_tokens", ["856, abc", 10**400], ids=["garbled", "too-large-for-a-float"])
def test_unparseable_provider_ratelimit_header_skips_only_its_own_gauge(
    prometheus_logger: PrometheusLogger, remaining_tokens: str | int
):
    payload: Final = _build_payload(
        model_group="groq-group",
        custom_llm_provider="groq",
        additional_headers={
            "x_ratelimit_remaining_requests": 0,
            "x_ratelimit_remaining_tokens": remaining_tokens,
        },
    )
    request_kwargs: Final = {
        "litellm_params": {"custom_llm_provider": "groq", "metadata": {}},
        "model": "llama-3.1-8b-instant",
        "standard_logging_object": payload,
    }

    prometheus_logger.set_llm_deployment_success_metrics(
        request_kwargs=request_kwargs,
        start_time=datetime(2026, 1, 1),
        end_time=datetime(2026, 1, 1, 0, 0, 1),
        enum_values=_enum_values(model_group="groq-group"),
        output_tokens=2.0,
    )

    assert _sample_values("litellm_remaining_requests_metric") == (0.0,)
    assert _sample_values("litellm_remaining_tokens_metric") == ()
    assert _sample_values("litellm_deployment_success_responses_total") == (1.0,)


@pytest.mark.parametrize(
    ("additional_headers", "remaining_requests", "remaining_tokens"),
    [
        ({"x_ratelimit_remaining_requests": "12.5", "x_ratelimit_remaining_tokens": "856.5"}, 12.5, 856.5),
        (
            {1: "non-string key", "x_ratelimit_remaining_requests": "12.5", "x_ratelimit_remaining_tokens": "856.5"},
            12.5,
            856.5,
        ),
        (
            {"x_ratelimit_remaining_requests": "inf", "x_ratelimit_remaining_tokens": "\u0661\u0662.\u0665"},
            float("inf"),
            12.5,
        ),
    ],
    ids=["decimal", "decimal-with-a-non-string-key", "infinity-and-arabic-indic-digits"],
)
def test_provider_ratelimit_headers_set_both_gauges_to_the_value_float_reads(
    prometheus_logger: PrometheusLogger,
    additional_headers: dict[object, object],
    remaining_requests: float,
    remaining_tokens: float,
):
    payload: Final = _build_payload(
        model_group="groq-group",
        custom_llm_provider="groq",
        additional_headers=additional_headers,
    )

    prometheus_logger.set_llm_deployment_success_metrics(
        request_kwargs={
            "litellm_params": {"custom_llm_provider": "groq", "metadata": {}},
            "model": "llama-3.1-8b-instant",
            "standard_logging_object": payload,
        },
        start_time=datetime(2026, 1, 1),
        end_time=datetime(2026, 1, 1, 0, 0, 1),
        enum_values=_enum_values(model_group="groq-group"),
        output_tokens=2.0,
    )

    assert _sample_values("litellm_remaining_requests_metric") == (remaining_requests,)
    assert _sample_values("litellm_remaining_tokens_metric") == (remaining_tokens,)
    assert _sample_values("litellm_deployment_success_responses_total") == (1.0,)
