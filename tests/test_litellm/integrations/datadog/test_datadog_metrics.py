import os
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest
from httpx import Response

from litellm.integrations.datadog.datadog_metrics import DatadogMetricsLogger
from litellm.types.utils import StandardLoggingPayload


@pytest.fixture
def clean_env():
    """Set test env vars and restore originals after test."""
    keys = ["DD_API_KEY", "DD_APP_KEY", "DD_SITE", "DD_ENV", "DD_SERVICE", "DD_VERSION"]
    originals = {k: os.environ.get(k) for k in keys}

    os.environ["DD_API_KEY"] = "test_api_key"
    os.environ["DD_APP_KEY"] = "test_app_key"
    os.environ["DD_SITE"] = "test.datadoghq.com"
    os.environ["DD_ENV"] = "test-env"
    os.environ["DD_SERVICE"] = "test-service"
    os.environ["DD_VERSION"] = "1.0.0"

    yield

    for k, v in originals.items():
        if v is not None:
            os.environ[k] = v
        elif k in os.environ:
            del os.environ[k]


@pytest.mark.asyncio
async def test_init(clean_env):
    """Test initialization sets up clients and url correctly."""
    logger = DatadogMetricsLogger(start_periodic_flush=False)
    logger = DatadogMetricsLogger(start_periodic_flush=False)
    assert logger.upload_url == "https://api.test.datadoghq.com/api/v2/series"


@pytest.mark.asyncio
async def test_extract_tags(clean_env):
    """Test tag extraction from a StandardLoggingPayload."""
    logger = DatadogMetricsLogger()

    payload = StandardLoggingPayload(
        custom_llm_provider="openai",
        model="gpt-4o",
        model_group="gpt-4",
        metadata={"user_api_key_team_alias": "test-team"},
    )

    tags = logger._extract_tags(log=payload, status_code="200")

    assert "env:test-env" in tags
    assert "service:test-service" in tags
    assert "version:1.0.0" in tags
    assert "provider:openai" in tags
    assert "model_name:gpt-4o" in tags
    assert "model_group:gpt-4" in tags
    assert "status_code:200" in tags
    assert "team:test-team" in tags


@pytest.mark.asyncio
async def test_extract_tags_no_team(clean_env):
    """Test tag extraction when no team info is present."""
    logger = DatadogMetricsLogger()

    payload = StandardLoggingPayload(
        custom_llm_provider="anthropic",
        model="claude-3-sonnet",
    )

    tags = logger._extract_tags(log=payload, status_code="500")

    assert "provider:anthropic" in tags
    assert "model_name:claude-3-sonnet" in tags
    assert "status_code:500" in tags
    assert not any(tag.startswith("team:") for tag in tags)


@pytest.mark.asyncio
async def test_add_metrics_from_log(clean_env):
    """Test that _add_metrics_from_log appends the correct metric series to the queue."""
    logger = DatadogMetricsLogger(batch_size=100)

    now = datetime.now()
    start_time = now - timedelta(seconds=2)
    api_call_start_time = now - timedelta(seconds=1)

    payload = StandardLoggingPayload(
        custom_llm_provider="openai",
        model="gpt-4o",
    )

    kwargs = {
        "start_time": start_time,
        "api_call_start_time": api_call_start_time,
        "end_time": now,
    }

    logger._add_metrics_from_log(log=payload, kwargs=kwargs, status_code="200")

    # Should have 3 series: total_latency, llm_api_latency, request_count
    assert len(logger.log_queue) == 3

    metrics = {s["metric"]: s for s in logger.log_queue}

    # Total latency ~2s
    total = metrics["litellm.request.total_latency"]
    assert total["type"] == 3  # gauge
    assert abs(total["points"][0]["value"] - 2.0) < 0.1

    # LLM API latency ~1s
    llm = metrics["litellm.llm_api.latency"]
    assert llm["type"] == 3  # gauge
    assert abs(llm["points"][0]["value"] - 1.0) < 0.1

    # Request count
    count = metrics["litellm.llm_api.request_count"]
    assert count["type"] == 1  # count
    assert count["points"][0]["value"] == 1.0
    assert "status_code:200" in count["tags"]


@pytest.mark.asyncio
async def test_async_log_success_event(clean_env):
    """Test that success events are added to the queue."""
    logger = DatadogMetricsLogger(batch_size=100)

    now = datetime.now()
    start_time = now - timedelta(seconds=1)

    await logger.async_log_success_event(
        kwargs={
            "standard_logging_object": StandardLoggingPayload(
                custom_llm_provider="openai",
                model="gpt-4o",
            ),
            "start_time": start_time,
            "end_time": now,
        },
        response_obj=None,
        start_time=start_time,
        end_time=now,
    )

    # At least request_count and total_latency
    assert len(logger.log_queue) >= 2


@pytest.mark.asyncio
async def test_async_log_success_event_no_standard_logging_object(clean_env):
    """Test that events without standard_logging_object are skipped."""
    logger = DatadogMetricsLogger(batch_size=100)

    await logger.async_log_success_event(
        kwargs={},
        response_obj=None,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    assert len(logger.log_queue) == 0


@pytest.mark.asyncio
async def test_async_log_failure_event_extracts_status_code(clean_env):
    """Test that failure events extract the error status code."""
    logger = DatadogMetricsLogger(batch_size=100)

    now = datetime.now()
    start_time = now - timedelta(seconds=1)

    await logger.async_log_failure_event(
        kwargs={
            "standard_logging_object": StandardLoggingPayload(
                custom_llm_provider="openai",
                model="gpt-4o",
                error_information={"error_code": "429"},
            ),
            "start_time": start_time,
            "end_time": now,
        },
        response_obj=None,
        start_time=start_time,
        end_time=now,
    )

    count_series = next(
        (s for s in logger.log_queue if s["metric"] == "litellm.llm_api.request_count"),
        None,
    )
    assert count_series is not None
    assert "status_code:429" in count_series["tags"]


@pytest.mark.asyncio
async def test_async_log_failure_event_default_status_code(clean_env):
    """Test that failure events default to 500 when no error_code is present."""
    logger = DatadogMetricsLogger(batch_size=100)

    now = datetime.now()

    await logger.async_log_failure_event(
        kwargs={
            "standard_logging_object": StandardLoggingPayload(
                custom_llm_provider="openai",
                model="gpt-4o",
            ),
            "start_time": now,
            "end_time": now,
        },
        response_obj=None,
        start_time=now,
        end_time=now,
    )

    count_series = next(
        (s for s in logger.log_queue if s["metric"] == "litellm.llm_api.request_count"),
        None,
    )
    assert count_series is not None
    assert "status_code:500" in count_series["tags"]


@pytest.mark.asyncio
async def test_async_send_batch(clean_env):
    """Test that async_send_batch uploads metrics to Datadog."""
    logger = DatadogMetricsLogger()
    logger.async_client = AsyncMock()
    logger.async_client.post.return_value = Response(202, json={"status": "ok"})

    # Manually add a metric series to the queue
    logger.log_queue = [
        {
            "metric": "litellm.request.total_latency",
            "type": 3,
            "points": [{"timestamp": int(time.time()), "value": 1.5}],
            "tags": ["env:test"],
        }
    ]

    await logger.async_send_batch()

    assert logger.async_client.post.called
    call_args = logger.async_client.post.call_args
    assert call_args[0][0] == "https://api.test.datadoghq.com/api/v2/series"

    # Verify gzip + JSON payload
    import gzip
    import json

    compressed = call_args[1]["content"]
    payload = json.loads(gzip.decompress(compressed).decode("utf-8"))
    assert len(payload["series"]) == 1
    assert payload["series"][0]["metric"] == "litellm.request.total_latency"


@pytest.mark.asyncio
async def test_async_send_batch_empty_queue(clean_env):
    """Test that async_send_batch does nothing when queue is empty."""
    logger = DatadogMetricsLogger()
    logger.async_client = AsyncMock()

    await logger.async_send_batch()

    assert not logger.async_client.post.called
