"""
Tests for the Prometheus rate-limit labels added on top of PR #27687.

Covers two follow-up gaps to the unified rate-limit error work:

1. ``litellm_proxy_failed_requests_metric`` now carries
   ``rate_limit_category`` and ``rate_limit_type`` labels populated from
   :class:`litellm.RateLimitError` (vendor + ``ProxyRateLimitError``
   subclass). Closes the Prometheus side of LIT-2718.
2. ``_get_exception_class_name`` keeps emitting the literal string
   ``"HTTPException"`` for ``ProxyRateLimitError`` so existing dashboards
   that key off ``exception_class="HTTPException"`` for litellm-internal
   429s don't silently break when the new class lands.
"""

from collections.abc import Mapping
from unittest.mock import MagicMock, patch

import pytest

from litellm.exceptions import (
    RateLimitError,
    RateLimitErrorCategory,
    RateLimitType,
)
from litellm.integrations.prometheus import PrometheusLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
from litellm.types.integrations.prometheus import (
    PrometheusMetricLabels,
    UserAPIKeyLabelNames,
    UserAPIKeyLabelValues,
)


# ---------------------------------------------------------------------------
# Label / enum wiring
# ---------------------------------------------------------------------------


def test_should_register_rate_limit_label_names_on_enum():
    assert UserAPIKeyLabelNames.RATE_LIMIT_CATEGORY.value == "rate_limit_category"
    assert UserAPIKeyLabelNames.RATE_LIMIT_TYPE.value == "rate_limit_type"


def test_should_include_rate_limit_labels_on_failed_requests_metric():
    import litellm

    original = litellm.prometheus_emit_rate_limit_labels
    try:
        litellm.prometheus_emit_rate_limit_labels = True
        labels = PrometheusMetricLabels.get_labels(
            "litellm_proxy_failed_requests_metric"
        )
        assert "rate_limit_category" in labels
        assert "rate_limit_type" in labels
        # These must coexist with the legacy exception labels (back-compat).
        assert "exception_class" in labels
        assert "exception_status" in labels
    finally:
        litellm.prometheus_emit_rate_limit_labels = original


def test_should_omit_rate_limit_labels_by_default_for_back_compat():
    """Default-off preserves the metric's historical label set so existing
    dashboards / recording rules keyed on `litellm_proxy_failed_requests_metric`
    keep matching after upgrade."""
    import litellm

    assert litellm.prometheus_emit_rate_limit_labels is False
    labels = PrometheusMetricLabels.get_labels("litellm_proxy_failed_requests_metric")
    assert "rate_limit_category" not in labels
    assert "rate_limit_type" not in labels
    # Pre-PR labels must still be present.
    assert "exception_class" in labels
    assert "exception_status" in labels


def test_should_accept_rate_limit_fields_on_user_api_key_label_values():
    enum_values = UserAPIKeyLabelValues(
        rate_limit_category="litellm_rate_limit",
        rate_limit_type="requests",
    )
    assert enum_values.rate_limit_category == "litellm_rate_limit"
    assert enum_values.rate_limit_type == "requests"


# ---------------------------------------------------------------------------
# _extract_rate_limit_labels helper
# ---------------------------------------------------------------------------


def test_should_extract_vendor_category_for_vanilla_rate_limit_error():
    err = RateLimitError(message="vendor 429", llm_provider="openai", model="gpt-4o")
    category, rate_limit_type = PrometheusLogger._extract_rate_limit_labels(err)
    assert category == "vendor_rate_limit"
    assert rate_limit_type is None


def test_should_extract_litellm_category_and_type_for_proxy_rate_limit_error():
    err = ProxyRateLimitError(
        detail={"error": "tpm exceeded"},
        category=RateLimitErrorCategory.LITELLM_RATE_LIMIT,
        rate_limit_type=RateLimitType.TOKENS,
    )
    category, rate_limit_type = PrometheusLogger._extract_rate_limit_labels(err)
    assert category == "litellm_rate_limit"
    assert rate_limit_type == "tokens"


def test_should_return_none_for_non_rate_limit_exception():
    assert PrometheusLogger._extract_rate_limit_labels(ValueError("nope")) == (
        None,
        None,
    )


def test_should_return_none_for_none_exception():
    assert PrometheusLogger._extract_rate_limit_labels(None) == (None, None)


def test_should_extract_budget_dimension_for_budget_exceeded_error():
    # Virtual-key / team / org / end-user budget caps raise
    # `litellm.BudgetExceededError` (a bare Exception subclass), which sets
    # the same `.category` / `.rate_limit_type` attributes as the unified
    # RateLimitError path so Prometheus can split budget 429s from other
    # 429s without the customer parsing free-text error messages.
    import litellm

    err = litellm.BudgetExceededError(current_cost=0.5, max_budget=0.1)
    category, rate_limit_type = PrometheusLogger._extract_rate_limit_labels(err)
    assert category == "litellm_rate_limit"
    assert rate_limit_type == "budget"


@pytest.mark.parametrize(
    "category_enum,rate_limit_enum,expected_category,expected_type",
    [
        (
            RateLimitErrorCategory.LITELLM_RATE_LIMIT,
            RateLimitType.REQUESTS,
            "litellm_rate_limit",
            "requests",
        ),
        (
            RateLimitErrorCategory.LITELLM_RATE_LIMIT,
            RateLimitType.TOKENS,
            "litellm_rate_limit",
            "tokens",
        ),
        (
            RateLimitErrorCategory.LITELLM_RATE_LIMIT,
            RateLimitType.CONCURRENT_REQUESTS,
            "litellm_rate_limit",
            "concurrent_requests",
        ),
        (
            RateLimitErrorCategory.LITELLM_RATE_LIMIT,
            RateLimitType.BUDGET,
            "litellm_rate_limit",
            "budget",
        ),
        (
            RateLimitErrorCategory.LITELLM_RATE_LIMIT,
            RateLimitType.MAX_ITERATIONS,
            "litellm_rate_limit",
            "max_iterations",
        ),
        (
            RateLimitErrorCategory.LITELLM_BATCH_RATE_LIMIT,
            RateLimitType.REQUESTS,
            "litellm_batch_rate_limit",
            "requests",
        ),
    ],
)
def test_should_serialize_rate_limit_enums_as_underlying_string_values(
    category_enum, rate_limit_enum, expected_category, expected_type
):
    err = ProxyRateLimitError(
        detail="boom", category=category_enum, rate_limit_type=rate_limit_enum
    )
    category, rate_limit_type = PrometheusLogger._extract_rate_limit_labels(err)
    assert category == expected_category
    assert rate_limit_type == expected_type


# ---------------------------------------------------------------------------
# _get_exception_class_name back-compat
# ---------------------------------------------------------------------------


def test_should_emit_legacy_http_exception_label_for_proxy_rate_limit_error():
    """
    ``ProxyRateLimitError`` multi-inherits from ``HTTPException`` +
    ``RateLimitError``. The ``exception_class`` label MUST keep emitting
    "HTTPException" for back-compat with existing dashboards (see Slack
    thread + PR #27687 review). Distinguishing vendor vs. litellm 429s
    is now the job of the new ``rate_limit_category`` label.
    """
    err = ProxyRateLimitError(detail={"error": "boom"})
    assert PrometheusLogger._get_exception_class_name(err) == "HTTPException"


def test_should_keep_provider_prefixed_exception_class_for_vendor_rate_limit_errors():
    err = RateLimitError(message="vendor 429", llm_provider="openai", model="gpt-4o")
    # Vendor-side errors keep the historical "Provider.ClassName" formatting.
    assert PrometheusLogger._get_exception_class_name(err) == "Openai.RateLimitError"


def test_should_preserve_exception_class_name_for_unrelated_exceptions():
    assert PrometheusLogger._get_exception_class_name(ValueError("nope")) == (
        "ValueError"
    )


# ---------------------------------------------------------------------------
# End-to-end wiring through async_post_call_failure_hook
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_should_populate_rate_limit_labels_for_proxy_rate_limit_error_on_failure_hook():
    """
    When a proxy hook raises ``ProxyRateLimitError`` and the failure flows
    through ``async_post_call_failure_hook``, the resulting
    ``UserAPIKeyLabelValues`` must carry both new labels AND keep
    ``exception_class="HTTPException"`` for back-compat.
    """
    with patch(
        "litellm.integrations.prometheus.PrometheusLogger.__init__", return_value=None
    ):
        logger = PrometheusLogger()
        logger.litellm_proxy_failed_requests_metric = MagicMock()
        logger.litellm_proxy_total_requests_metric = MagicMock()
        logger.get_labels_for_metric = MagicMock(
            return_value=PrometheusMetricLabels.get_labels(
                "litellm_proxy_failed_requests_metric"
            )
        )

    err = ProxyRateLimitError(
        detail={"error": "rpm exceeded"},
        category=RateLimitErrorCategory.LITELLM_RATE_LIMIT,
        rate_limit_type=RateLimitType.REQUESTS,
    )

    with patch(
        "litellm.integrations.prometheus.prometheus_label_factory"
    ) as mock_label_factory:
        mock_label_factory.return_value = {}
        await logger.async_post_call_failure_hook(
            request_data={"model": "gpt-4o-mini", "metadata": {}},
            original_exception=err,
            user_api_key_dict=UserAPIKeyAuth(token="t"),
        )

    enum_values = mock_label_factory.call_args_list[0].kwargs["enum_values"]
    assert isinstance(enum_values, UserAPIKeyLabelValues)
    assert enum_values.rate_limit_category == "litellm_rate_limit"
    assert enum_values.rate_limit_type == "requests"
    # Back-compat: exception_class on a ProxyRateLimitError stays "HTTPException".
    assert enum_values.exception_class == "HTTPException"
    assert enum_values.exception_status == "429"


@pytest.mark.asyncio
async def test_should_populate_rate_limit_labels_for_vendor_rate_limit_error_on_failure_hook():
    with patch(
        "litellm.integrations.prometheus.PrometheusLogger.__init__", return_value=None
    ):
        logger = PrometheusLogger()
        logger.litellm_proxy_failed_requests_metric = MagicMock()
        logger.litellm_proxy_total_requests_metric = MagicMock()
        logger.get_labels_for_metric = MagicMock(
            return_value=PrometheusMetricLabels.get_labels(
                "litellm_proxy_failed_requests_metric"
            )
        )

    err = RateLimitError(message="upstream 429", llm_provider="openai", model="gpt-4o")

    with patch(
        "litellm.integrations.prometheus.prometheus_label_factory"
    ) as mock_label_factory:
        mock_label_factory.return_value = {}
        await logger.async_post_call_failure_hook(
            request_data={"model": "gpt-4o", "metadata": {}},
            original_exception=err,
            user_api_key_dict=UserAPIKeyAuth(token="t"),
        )

    enum_values = mock_label_factory.call_args_list[0].kwargs["enum_values"]
    assert isinstance(enum_values, UserAPIKeyLabelValues)
    assert enum_values.rate_limit_category == "vendor_rate_limit"
    assert enum_values.rate_limit_type is None
    # Vendor errors keep the historical Provider.ClassName label.
    assert enum_values.exception_class == "Openai.RateLimitError"
    assert enum_values.exception_status == "429"


@pytest.mark.asyncio
async def test_should_leave_rate_limit_labels_blank_for_non_rate_limit_failure():
    with patch(
        "litellm.integrations.prometheus.PrometheusLogger.__init__", return_value=None
    ):
        logger = PrometheusLogger()
        logger.litellm_proxy_failed_requests_metric = MagicMock()
        logger.litellm_proxy_total_requests_metric = MagicMock()
        logger.get_labels_for_metric = MagicMock(
            return_value=PrometheusMetricLabels.get_labels(
                "litellm_proxy_failed_requests_metric"
            )
        )

    with patch(
        "litellm.integrations.prometheus.prometheus_label_factory"
    ) as mock_label_factory:
        mock_label_factory.return_value = {}
        await logger.async_post_call_failure_hook(
            request_data={"model": "gpt-4o", "metadata": {}},
            original_exception=RuntimeError("boom"),
            user_api_key_dict=UserAPIKeyAuth(token="t"),
        )

    enum_values = mock_label_factory.call_args_list[0].kwargs["enum_values"]
    assert isinstance(enum_values, UserAPIKeyLabelValues)
    assert enum_values.rate_limit_category is None
    assert enum_values.rate_limit_type is None


def _logger_with_mock_virtual_key_gauges() -> PrometheusLogger:
    with patch(
        "litellm.integrations.prometheus.PrometheusLogger.__init__", return_value=None
    ):
        logger = PrometheusLogger()
    logger.litellm_remaining_api_key_requests_for_model = MagicMock()
    logger.litellm_remaining_api_key_tokens_for_model = MagicMock()
    logger.get_labels_for_metric = MagicMock(return_value=[])
    return logger


def _kwargs_with_v3_rate_limit_headers(additional_headers: dict) -> dict:
    return {
        "litellm_params": {"metadata": {"model_group": "gpt-4o-mini"}},
        "standard_logging_object": {
            "metadata": {},
            "hidden_params": {"additional_headers": additional_headers},
        },
    }


def _set_virtual_key_metrics(logger: PrometheusLogger, kwargs: dict) -> None:
    logger._set_virtual_key_rate_limit_metrics(
        user_api_key="test-hash",
        user_api_key_alias="test-alias",
        kwargs=kwargs,
        metadata=kwargs["litellm_params"]["metadata"],
        model_id="model-123",
    )


def test_should_read_v3_remaining_headers_when_metadata_keys_absent():
    """
    Regression for LIT-2577: the default v3 rate limiter writes remaining
    per-(key, model) values into
    ``standard_logging_object.hidden_params.additional_headers`` as
    ``x-ratelimit-model_per_key-remaining-{requests,tokens}`` and never sets
    the legacy ``litellm-key-remaining-*`` metadata keys, so the gauges were
    pinned to ``sys.maxsize``.
    """
    logger = _logger_with_mock_virtual_key_gauges()
    kwargs = _kwargs_with_v3_rate_limit_headers(
        {
            "x-ratelimit-model_per_key-remaining-requests": 42,
            "x-ratelimit-model_per_key-remaining-tokens": 900,
            "x-ratelimit-model_per_key-limit-requests": 100,
            "x-ratelimit-model_per_key-limit-tokens": 1000,
        }
    )

    _set_virtual_key_metrics(logger, kwargs)

    logger.litellm_remaining_api_key_requests_for_model.labels.return_value.set.assert_called_once_with(
        42
    )
    logger.litellm_remaining_api_key_tokens_for_model.labels.return_value.set.assert_called_once_with(
        900
    )


def test_should_prefer_legacy_metadata_keys_over_v3_headers():
    logger = _logger_with_mock_virtual_key_gauges()
    kwargs = _kwargs_with_v3_rate_limit_headers(
        {
            "x-ratelimit-model_per_key-remaining-requests": 42,
            "x-ratelimit-model_per_key-remaining-tokens": 900,
        }
    )
    kwargs["litellm_params"]["metadata"].update(
        {
            "litellm-key-remaining-requests-gpt-4o-mini": 3,
            "litellm-key-remaining-tokens-gpt-4o-mini": 200,
        }
    )

    _set_virtual_key_metrics(logger, kwargs)

    logger.litellm_remaining_api_key_requests_for_model.labels.return_value.set.assert_called_once_with(
        3
    )
    logger.litellm_remaining_api_key_tokens_for_model.labels.return_value.set.assert_called_once_with(
        200
    )


def test_should_treat_zero_v3_remaining_as_zero():
    logger = _logger_with_mock_virtual_key_gauges()
    kwargs = _kwargs_with_v3_rate_limit_headers(
        {
            "x-ratelimit-model_per_key-remaining-requests": 0,
            "x-ratelimit-model_per_key-remaining-tokens": 0,
        }
    )

    _set_virtual_key_metrics(logger, kwargs)

    logger.litellm_remaining_api_key_requests_for_model.labels.return_value.set.assert_called_once_with(
        0
    )
    logger.litellm_remaining_api_key_tokens_for_model.labels.return_value.set.assert_called_once_with(
        0
    )


def test_should_keep_maxsize_sentinel_when_no_rate_limit_source_present():
    import sys

    logger = _logger_with_mock_virtual_key_gauges()
    kwargs = {
        "litellm_params": {"metadata": {"model_group": "gpt-4o-mini"}},
        "standard_logging_object": {"metadata": {}, "hidden_params": {}},
    }

    _set_virtual_key_metrics(logger, kwargs)

    logger.litellm_remaining_api_key_requests_for_model.labels.return_value.set.assert_called_once_with(
        sys.maxsize
    )
    logger.litellm_remaining_api_key_tokens_for_model.labels.return_value.set.assert_called_once_with(
        sys.maxsize
    )


@pytest.mark.parametrize("bad_value", ["not-a-number", None, True])
def test_should_ignore_non_int_v3_header_values(bad_value):
    import sys

    logger = _logger_with_mock_virtual_key_gauges()
    kwargs = _kwargs_with_v3_rate_limit_headers(
        {
            "x-ratelimit-model_per_key-remaining-requests": bad_value,
            "x-ratelimit-model_per_key-remaining-tokens": bad_value,
        }
    )

    _set_virtual_key_metrics(logger, kwargs)

    logger.litellm_remaining_api_key_requests_for_model.labels.return_value.set.assert_called_once_with(
        sys.maxsize
    )
    logger.litellm_remaining_api_key_tokens_for_model.labels.return_value.set.assert_called_once_with(
        sys.maxsize
    )


KEY_AND_TEAM_RATE_LIMIT_METRICS = (
    "litellm_api_key_rate_limit_allowed_metric",
    "litellm_api_key_rate_limit_used_metric",
    "litellm_team_rate_limit_allowed_metric",
    "litellm_team_rate_limit_used_metric",
)


def _clear_prometheus_registry() -> None:
    from prometheus_client import REGISTRY

    for collector in list(REGISTRY._collector_to_names.keys()):
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass


def _collected_samples(metric_name: str) -> dict[tuple[tuple[str, str], ...], float]:
    from prometheus_client import REGISTRY

    return {
        tuple(sorted(sample.labels.items())): sample.value
        for metric in REGISTRY.collect()
        for sample in metric.samples
        if sample.name == metric_name
    }


def _success_kwargs_with_rate_limit_headers(additional_headers: Mapping[str, object] | None) -> dict[str, object]:
    return {
        "model": "claude-haiku-4-5",
        "litellm_params": {"metadata": {}},
        "standard_logging_object": {
            "id": "t",
            "call_type": "completion",
            "response_cost": 0.001,
            "status": "success",
            "total_tokens": 20,
            "prompt_tokens": 15,
            "completion_tokens": 5,
            "startTime": 1.0,
            "endTime": 2.0,
            "completionStartTime": 1.5,
            "model": "claude-haiku-4-5",
            "model_id": "model-123",
            "model_group": "anthropic-haiku-4-5",
            "api_base": "https://api.anthropic.com",
            "custom_llm_provider": "anthropic",
            "request_tags": [],
            "end_user": None,
            "cache_hit": False,
            "stream": False,
            "response": None,
            "model_parameters": None,
            "metadata": {
                "user_api_key_hash": "key-hash",
                "user_api_key_alias": "key-alias",
                "user_api_key_team_id": "team-id",
                "user_api_key_team_alias": "team-alias",
                "user_api_key_user_id": "u",
                "user_api_key_user_email": "e@x.com",
                "user_api_key_org_id": None,
                "user_api_key_org_alias": None,
                "requester_metadata": None,
                "user_api_key_end_user_id": None,
                "usage_object": None,
            },
            "hidden_params": {
                "litellm_overhead_time_ms": None,
                "additional_headers": additional_headers,
            },
        },
    }


async def _run_success_event(
    additional_headers: Mapping[str, object] | None, logger: PrometheusLogger | None = None
) -> None:
    import datetime

    now = datetime.datetime.now()
    await (logger or PrometheusLogger()).async_log_success_event(
        _success_kwargs_with_rate_limit_headers(additional_headers), None, now, now
    )


@pytest.mark.asyncio
async def test_should_emit_key_and_team_rate_limit_allowed_and_used_from_v3_headers():
    """
    LIT-1672: the v3 limiter mirrors ``x-ratelimit-{api_key,team}-{limit,remaining}-*``
    into the logging payload. The gauges must expose the configured limit as-is
    and the window consumption as ``limit - remaining`` for each key / team
    dimension, split by ``rate_limit_type``.
    """
    _clear_prometheus_registry()
    try:
        await _run_success_event(
            {
                "x-ratelimit-api_key-limit-requests": 10,
                "x-ratelimit-api_key-remaining-requests": 7,
                "x-ratelimit-api_key-limit-tokens": 20000,
                "x-ratelimit-api_key-remaining-tokens": 19947,
                "x-ratelimit-team-limit-requests": 50,
                "x-ratelimit-team-remaining-requests": 47,
                "x-ratelimit-team-limit-tokens": 40000,
                "x-ratelimit-team-remaining-tokens": 39960,
                "x-ratelimit-model_per_key-limit-requests": 5,
                "x-ratelimit-model_per_key-remaining-requests": 1,
            }
        )

        key_requests = (
            ("api_key_alias", "key-alias"),
            ("hashed_api_key", "key-hash"),
            ("rate_limit_type", "requests"),
        )
        key_tokens = (
            ("api_key_alias", "key-alias"),
            ("hashed_api_key", "key-hash"),
            ("rate_limit_type", "tokens"),
        )
        team_requests = (
            ("rate_limit_type", "requests"),
            ("team", "team-id"),
            ("team_alias", "team-alias"),
        )
        team_tokens = (
            ("rate_limit_type", "tokens"),
            ("team", "team-id"),
            ("team_alias", "team-alias"),
        )

        assert _collected_samples("litellm_api_key_rate_limit_allowed_metric") == {
            key_requests: 10,
            key_tokens: 20000,
        }
        assert _collected_samples("litellm_api_key_rate_limit_used_metric") == {
            key_requests: 3,
            key_tokens: 53,
        }
        assert _collected_samples("litellm_team_rate_limit_allowed_metric") == {
            team_requests: 50,
            team_tokens: 40000,
        }
        assert _collected_samples("litellm_team_rate_limit_used_metric") == {
            team_requests: 3,
            team_tokens: 40,
        }
    finally:
        _clear_prometheus_registry()


@pytest.mark.asyncio
async def test_should_emit_only_the_dimensions_the_limiter_enforced():
    """
    A key with only ``rpm_limit`` set and no team limits produces only the
    key/requests headers, so no tokens series and no team series may appear
    (a phantom 0 or sys.maxsize series would misreport an unlimited dimension).
    """
    _clear_prometheus_registry()
    try:
        await _run_success_event(
            {
                "x-ratelimit-api_key-limit-requests": 10,
                "x-ratelimit-api_key-remaining-requests": 10,
            }
        )

        key_requests = (
            ("api_key_alias", "key-alias"),
            ("hashed_api_key", "key-hash"),
            ("rate_limit_type", "requests"),
        )
        assert _collected_samples("litellm_api_key_rate_limit_allowed_metric") == {key_requests: 10}
        assert _collected_samples("litellm_api_key_rate_limit_used_metric") == {key_requests: 0}
        assert _collected_samples("litellm_team_rate_limit_allowed_metric") == {}
        assert _collected_samples("litellm_team_rate_limit_used_metric") == {}
    finally:
        _clear_prometheus_registry()


@pytest.mark.asyncio
async def test_should_drop_key_and_team_series_once_the_limiter_stops_reporting_a_limit():
    """
    Removing a key's ``rpm_limit`` / ``tpm_limit`` (or a team's ``tpm_limit``)
    makes the v3 limiter stop emitting that descriptor's headers on later
    requests. The old allowed/used samples must disappear instead of keeping
    a limit that no longer exists on the scrape.
    """
    _clear_prometheus_registry()
    try:
        logger = PrometheusLogger()
        await _run_success_event(
            {
                "x-ratelimit-api_key-limit-requests": 10,
                "x-ratelimit-api_key-remaining-requests": 7,
                "x-ratelimit-api_key-limit-tokens": 20000,
                "x-ratelimit-api_key-remaining-tokens": 19947,
                "x-ratelimit-team-limit-requests": 50,
                "x-ratelimit-team-remaining-requests": 47,
                "x-ratelimit-team-limit-tokens": 40000,
                "x-ratelimit-team-remaining-tokens": 39960,
            },
            logger=logger,
        )
        await _run_success_event(
            {
                "x-ratelimit-team-limit-requests": 50,
                "x-ratelimit-team-remaining-requests": 46,
            },
            logger=logger,
        )

        team_requests = (
            ("rate_limit_type", "requests"),
            ("team", "team-id"),
            ("team_alias", "team-alias"),
        )
        assert _collected_samples("litellm_api_key_rate_limit_allowed_metric") == {}
        assert _collected_samples("litellm_api_key_rate_limit_used_metric") == {}
        assert _collected_samples("litellm_team_rate_limit_allowed_metric") == {team_requests: 50}
        assert _collected_samples("litellm_team_rate_limit_used_metric") == {team_requests: 4}
    finally:
        _clear_prometheus_registry()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "additional_headers",
    [
        None,
        {"x-ratelimit-model_per_key-remaining-requests": 42},
        {"x-ratelimit-api_key-limit-requests": 10},
        {"x-ratelimit-api_key-limit-requests": "10", "x-ratelimit-api_key-remaining-requests": "7"},
        {"x-ratelimit-team-limit-tokens": True, "x-ratelimit-team-remaining-tokens": 5},
    ],
)
async def test_should_emit_no_key_or_team_rate_limit_series_without_a_complete_int_pair(
    additional_headers,
):
    _clear_prometheus_registry()
    try:
        await _run_success_event(additional_headers)

        for metric_name in KEY_AND_TEAM_RATE_LIMIT_METRICS:
            assert _collected_samples(metric_name) == {}, metric_name
    finally:
        _clear_prometheus_registry()
