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
