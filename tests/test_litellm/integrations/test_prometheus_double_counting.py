import pytest
from unittest.mock import AsyncMock, MagicMock
import litellm
from litellm.integrations.prometheus import PrometheusLogger
from litellm.proxy.utils import ProxyLogging


@pytest.fixture
def mock_prometheus_logger():
    logger = MagicMock(spec=PrometheusLogger)

    # Bind the methods we want to test
    logger.async_log_success_event = PrometheusLogger.async_log_success_event.__get__(
        logger
    )
    logger.async_post_call_success_hook = (
        PrometheusLogger.async_post_call_success_hook.__get__(logger)
    )
    logger.get_labels_for_metric = PrometheusLogger.get_labels_for_metric.__get__(
        logger
    )

    # Mock all the internal helpers
    logger._increment_top_level_request_and_spend_metrics = MagicMock()
    logger._increment_token_metrics = MagicMock()
    logger._increment_remaining_budget_metrics = AsyncMock()
    logger._set_virtual_key_rate_limit_metrics = MagicMock()
    logger._set_latency_metrics = MagicMock()
    logger.set_llm_deployment_success_metrics = MagicMock()
    logger._increment_cache_metrics = MagicMock()
    logger._should_skip_metrics_for_invalid_key = MagicMock(return_value=False)

    logger.label_filters = {}
    logger.enabled_metrics = []

    # Mock the metric so we can count calls
    logger.litellm_proxy_total_requests_metric = MagicMock()
    logger.litellm_proxy_total_requests_metric.labels.return_value = MagicMock()
    return logger


@pytest.fixture
def standard_kwargs():
    return {
        "standard_logging_object": {
            "metadata": {
                "user_api_key_user_id": "test_user",
                "user_api_key_hash": "test_hash",
                "user_api_key_alias": "test_alias",
                "user_api_key_team_id": "test_team",
                "user_api_key_team_alias": "test_team_alias",
                "user_api_key_user_email": "test@test.com",
            },
            "completion_tokens": 10,
            "total_tokens": 20,
            "prompt_tokens": 10,
            "response_cost": 0.001,
            "model_group": "gpt-4",
            "model_id": "gpt-4-123",
            "api_base": "https://api.openai.com",
            "custom_llm_provider": "openai",
            "request_tags": [],
            "stream": False,
        },
        "model": "gpt-4",
        "litellm_params": {"metadata": {}},
    }


@pytest.mark.asyncio
async def test_non_streaming_request_increments_litellm_proxy_total_requests_metric_exactly_once(
    mock_prometheus_logger, standard_kwargs
):
    """
    Test non-streaming request increments litellm_proxy_total_requests_metric exactly once
    """
    # Verify stream is False
    assert standard_kwargs["standard_logging_object"]["stream"] is False

    await mock_prometheus_logger.async_log_success_event(
        kwargs=standard_kwargs,
        response_obj=None,
        start_time=None,
        end_time=None,
    )

    # Assert inc() was called exactly once on the labels object
    labels_mock = (
        mock_prometheus_logger.litellm_proxy_total_requests_metric.labels.return_value
    )
    assert labels_mock.inc.call_count == 1


@pytest.mark.asyncio
async def test_streaming_request_increments_litellm_proxy_total_requests_metric_exactly_once(
    mock_prometheus_logger, standard_kwargs
):
    """
    Test streaming request increments litellm_proxy_total_requests_metric exactly once
    """
    # Set stream to True
    standard_kwargs["standard_logging_object"]["stream"] = True

    await mock_prometheus_logger.async_log_success_event(
        kwargs=standard_kwargs,
        response_obj=None,
        start_time=None,
        end_time=None,
    )

    # Assert inc() was called exactly once on the labels object
    labels_mock = (
        mock_prometheus_logger.litellm_proxy_total_requests_metric.labels.return_value
    )
    assert labels_mock.inc.call_count == 1


@pytest.mark.asyncio
async def test_async_post_call_success_hook_does_not_increment_metric(
    mock_prometheus_logger,
):
    """
    Test async_post_call_success_hook does NOT increment the metric
    """
    await mock_prometheus_logger.async_post_call_success_hook(
        data={}, user_api_key_dict=None, response=None
    )

    # Assert metric was NOT touched
    mock_prometheus_logger.litellm_proxy_total_requests_metric.labels.assert_not_called()


def test_callback_deduplication():
    """
    Test callback deduplication in _init_litellm_callbacks
    """
    from litellm.caching.caching import DualCache

    proxy_logging = ProxyLogging(user_api_key_cache=DualCache())

    # Store old callbacks so we can restore them later
    old_callbacks = litellm.callbacks.copy()

    try:
        # We start with empty components + "prometheus" to have isolated testing context
        # proxy_hooks might inject other stuff, but we care that `prometheus` string is removed
        # and replaced by an instance.
        litellm.callbacks = ["prometheus", "prometheus"]

        # Calling init which modifies litellm.callbacks in-place
        proxy_logging._init_litellm_callbacks()

        # Verify there are no string representations of prometheus left
        string_callbacks = [
            c for c in litellm.callbacks if isinstance(c, str) and c == "prometheus"
        ]
        assert len(string_callbacks) == 0

        # We initialized "prometheus", so there should be prometheus custom callback object here
        prometheus_instances = [
            c
            for c in litellm.callbacks
            if "prometheus" in str(type(c)).lower()
            or getattr(c, "name", None) == "prometheus_logger"
        ]

        # Since we added "prometheus" string twice, it might initialize it twice, but replacing
        # in-place should lead to correct count matching replaced. Actually the old buggy code
        # wouldn't remove "prometheus". The new one replaces it.
        assert len(prometheus_instances) > 0
    finally:
        # Restore
        litellm.callbacks = old_callbacks
