import gzip
import json
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest
from httpx import Request, Response

from litellm.integrations.datadog.datadog_metrics import DatadogMetricsLogger
from litellm.types.utils import StandardLoggingPayload


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in (
        ("DD_API_KEY", "test_api_key"),
        ("DD_APP_KEY", "test_app_key"),
        ("DD_SITE", "test.datadoghq.com"),
        ("DD_ENV", "test-env"),
        ("DD_SERVICE", "test-service"),
        ("DD_VERSION", "1.0.0"),
    ):
        monkeypatch.setenv(key, value)


@pytest.mark.asyncio
async def test_init(clean_env):
    """Test initialization sets up clients and url correctly."""
    logger = DatadogMetricsLogger(start_periodic_flush=False)
    assert logger.upload_url == "https://api.test.datadoghq.com/api/v2/series"


@pytest.mark.asyncio
async def test_extract_tags(clean_env):
    """Test tag extraction from a StandardLoggingPayload."""
    logger = DatadogMetricsLogger(start_periodic_flush=False)

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
async def test_extract_tags_normalizes_team_alias(clean_env):
    """Team aliases with uppercase or special characters match what Datadog stores."""
    logger = DatadogMetricsLogger(start_periodic_flush=False)

    payload = StandardLoggingPayload(
        custom_llm_provider="openai",
        model="gpt-4o",
        metadata={"user_api_key_team_alias": "P&T CTO-B2B"},
    )

    tags = logger._extract_tags(log=payload, status_code="200")

    assert "team:p_t_cto-b2b" in tags


@pytest.mark.asyncio
async def test_extract_tags_keeps_non_string_team_id(clean_env):
    """A numeric team id still produces a team tag instead of aborting the metric."""
    logger = DatadogMetricsLogger(start_periodic_flush=False)

    payload = StandardLoggingPayload(
        custom_llm_provider="openai",
        model="gpt-4o",
        metadata={"user_api_key_team_id": 67890},
    )

    tags = logger._extract_tags(log=payload, status_code="200")

    assert "team:67890" in tags


@pytest.mark.asyncio
async def test_extract_tags_no_team(clean_env):
    """Test tag extraction when no team info is present."""
    logger = DatadogMetricsLogger(start_periodic_flush=False)

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
    logger = DatadogMetricsLogger(batch_size=100, start_periodic_flush=False)

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

    # total_latency and llm_api_latency (each as gauge + distribution) plus request_count
    # (no overhead metric because payload has no hidden_params litellm_overhead_time_ms)
    assert len(logger.log_queue) == 5

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
async def test_latency_metrics_also_emitted_as_distributions(clean_env):
    """Each latency sample is queued as a distribution point so Datadog can compute per-request percentiles."""
    logger = DatadogMetricsLogger(batch_size=100, start_periodic_flush=False)

    now = datetime.now()
    payload = StandardLoggingPayload(
        custom_llm_provider="openai",
        model="gpt-4o",
        hidden_params={"litellm_overhead_time_ms": 250},
    )
    kwargs = {
        "start_time": now - timedelta(seconds=2),
        "api_call_start_time": now - timedelta(seconds=1),
        "end_time": now,
    }

    logger._add_metrics_from_log(log=payload, kwargs=kwargs, status_code="200")

    distributions = {s["metric"]: s for s in logger.log_queue if s["type"] == "distribution"}
    assert set(distributions) == {
        "litellm.request.total_latency.distribution",
        "litellm.llm_api.latency.distribution",
        "litellm.overhead.latency.distribution",
    }

    expected_seconds = {
        "litellm.request.total_latency.distribution": 2.0,
        "litellm.llm_api.latency.distribution": 1.0,
        "litellm.overhead.latency.distribution": 0.25,
    }
    for metric, seconds in expected_seconds.items():
        ((timestamp, values),) = distributions[metric]["points"]
        assert timestamp == int(now.timestamp())
        assert len(values) == 1
        assert abs(values[0] - seconds) < 0.1
        assert "provider:openai" in distributions[metric]["tags"]
        assert "model_name:gpt-4o" in distributions[metric]["tags"]

    assert "status_code:200" in distributions["litellm.request.total_latency.distribution"]["tags"]
    assert not any(
        tag.startswith("status_code:") for tag in distributions["litellm.overhead.latency.distribution"]["tags"]
    )


@pytest.mark.asyncio
async def test_extract_tags_omits_hostname_when_unset(clean_env, monkeypatch: pytest.MonkeyPatch):
    """An unset HOSTNAME must not produce an empty `HOSTNAME:` tag."""
    monkeypatch.delenv("HOSTNAME", raising=False)
    logger = DatadogMetricsLogger(start_periodic_flush=False)

    tags = logger._extract_tags(log=StandardLoggingPayload(model="gpt-4o"))

    assert not any(tag.startswith("HOSTNAME") for tag in tags)

    monkeypatch.setenv("HOSTNAME", "pod-abc")
    assert "HOSTNAME:pod-abc" in logger._extract_tags(log=StandardLoggingPayload(model="gpt-4o"))


@pytest.mark.asyncio
async def test_overhead_latency_metric_emitted(clean_env):
    """Test that litellm.overhead.latency is emitted when hidden_params contains litellm_overhead_time_ms."""
    logger = DatadogMetricsLogger(batch_size=100, start_periodic_flush=False)

    now = datetime.now()
    start_time = now - timedelta(seconds=2)
    api_call_start_time = now - timedelta(seconds=1)

    payload = StandardLoggingPayload(
        custom_llm_provider="openai",
        model="gpt-4o",
        hidden_params={
            "litellm_overhead_time_ms": 250.0,  # 250 ms of overhead
        },
    )

    kwargs = {
        "start_time": start_time,
        "api_call_start_time": api_call_start_time,
        "end_time": now,
    }

    logger._add_metrics_from_log(log=payload, kwargs=kwargs, status_code="200")

    metrics = {s["metric"]: s for s in logger.log_queue}

    # Overhead metric must be present
    assert "litellm.overhead.latency" in metrics, (
        f"Expected 'litellm.overhead.latency' in emitted metrics, got: {list(metrics.keys())}"
    )
    overhead = metrics["litellm.overhead.latency"]
    assert overhead["type"] == 3  # gauge
    # 250 ms → 0.25 s
    assert abs(overhead["points"][0]["value"] - 0.25) < 1e-6
    # status_code should NOT be in overhead tags (it is a latency metric, not a request count)
    assert not any(tag.startswith("status_code:") for tag in overhead["tags"])


@pytest.mark.asyncio
async def test_overhead_latency_metric_absent_when_no_hidden_params(clean_env):
    """Test that litellm.overhead.latency is NOT emitted when hidden_params has no overhead value."""
    logger = DatadogMetricsLogger(batch_size=100, start_periodic_flush=False)

    now = datetime.now()
    start_time = now - timedelta(seconds=2)
    api_call_start_time = now - timedelta(seconds=1)

    payload = StandardLoggingPayload(
        custom_llm_provider="openai",
        model="gpt-4o",
        # No hidden_params / no litellm_overhead_time_ms
    )

    kwargs = {
        "start_time": start_time,
        "api_call_start_time": api_call_start_time,
        "end_time": now,
    }

    logger._add_metrics_from_log(log=payload, kwargs=kwargs, status_code="200")

    metrics = {s["metric"]: s for s in logger.log_queue}
    assert "litellm.overhead.latency" not in metrics


@pytest.mark.asyncio
async def test_async_log_success_event(clean_env):
    """Test that success events are added to the queue."""
    logger = DatadogMetricsLogger(batch_size=100, start_periodic_flush=False)

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
    logger = DatadogMetricsLogger(batch_size=100, start_periodic_flush=False)

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
    logger = DatadogMetricsLogger(batch_size=100, start_periodic_flush=False)

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
    logger = DatadogMetricsLogger(batch_size=100, start_periodic_flush=False)

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
    logger = DatadogMetricsLogger(start_periodic_flush=False)
    logger.async_client = AsyncMock()
    mock_request = Request("POST", "https://api.test.datadoghq.com/api/v2/series")
    logger.async_client.post.return_value = Response(202, json={"status": "ok"}, request=mock_request)

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
async def test_async_send_batch_routes_distributions_to_v1_endpoint(clean_env):
    """Distribution series go to /api/v1/distribution_points (deflate), gauges/counts stay on /api/v2/series."""
    import gzip
    import json
    import zlib

    logger = DatadogMetricsLogger(start_periodic_flush=False)
    logger.async_client = AsyncMock()
    logger.async_client.post.return_value = Response(
        202, json={"status": "ok"}, request=Request("POST", "https://api.test.datadoghq.com")
    )

    timestamp = int(time.time())
    logger.log_queue = [
        {
            "metric": "litellm.request.total_latency",
            "type": 3,
            "points": [{"timestamp": timestamp, "value": 1.5}],
            "tags": ["env:test"],
        },
        {
            "metric": "litellm.request.total_latency.distribution",
            "type": "distribution",
            "points": ((timestamp, (1.5,)),),
            "tags": ("env:test",),
        },
    ]

    await logger.async_send_batch()

    calls = {call.args[0]: call.kwargs for call in logger.async_client.post.call_args_list}
    assert set(calls) == {
        "https://api.test.datadoghq.com/api/v2/series",
        "https://api.test.datadoghq.com/api/v1/distribution_points",
    }

    series_call = calls["https://api.test.datadoghq.com/api/v2/series"]
    assert series_call["headers"]["Content-Encoding"] == "gzip"
    series_payload = json.loads(gzip.decompress(series_call["content"]))
    assert [s["metric"] for s in series_payload["series"]] == ["litellm.request.total_latency"]

    distribution_call = calls["https://api.test.datadoghq.com/api/v1/distribution_points"]
    assert distribution_call["headers"]["Content-Encoding"] == "deflate"
    assert distribution_call["headers"]["DD-API-KEY"] == "test_api_key"
    distribution_payload = json.loads(zlib.decompress(distribution_call["content"]))
    assert distribution_payload == {
        "series": [
            {
                "metric": "litellm.request.total_latency.distribution",
                "type": "distribution",
                "points": [[timestamp, [1.5]]],
                "tags": ["env:test"],
            }
        ]
    }


@pytest.mark.asyncio
async def test_async_send_batch_skips_v2_when_only_distributions_queued(clean_env):
    logger = DatadogMetricsLogger(start_periodic_flush=False)
    logger.async_client = AsyncMock()
    logger.async_client.post.return_value = Response(
        202, json={"status": "ok"}, request=Request("POST", "https://api.test.datadoghq.com")
    )
    logger.log_queue = [
        {
            "metric": "litellm.llm_api.latency.distribution",
            "type": "distribution",
            "points": ((int(time.time()), (0.4,)),),
            "tags": ("env:test",),
        }
    ]

    await logger.async_send_batch()

    assert [call.args[0] for call in logger.async_client.post.call_args_list] == [
        "https://api.test.datadoghq.com/api/v1/distribution_points"
    ]


@pytest.mark.asyncio
async def test_flush_retries_only_distributions_after_v1_failure(clean_env):
    logger = DatadogMetricsLogger(start_periodic_flush=False)
    logger.async_client = AsyncMock()
    v2_url = "https://api.test.datadoghq.com/api/v2/series"
    v1_url = "https://api.test.datadoghq.com/api/v1/distribution_points"
    responses = {
        v2_url: Response(202, json={"status": "ok"}, request=Request("POST", v2_url)),
        v1_url: Response(503, json={"errors": ["down"]}, request=Request("POST", v1_url)),
    }
    timestamp = int(time.time())
    late_count = {
        "metric": "litellm.llm_api.request_count",
        "type": 1,
        "points": [{"timestamp": timestamp, "value": 1}],
        "tags": ["env:test"],
    }

    def post(url, **_):
        if url == v1_url and late_count not in logger.log_queue:
            logger.log_queue.append(late_count)
        return responses[url]

    logger.async_client.post.side_effect = post

    gauge = {
        "metric": "litellm.request.total_latency",
        "type": 3,
        "points": [{"timestamp": timestamp, "value": 1.5}],
        "tags": ["env:test"],
    }
    distribution = {
        "metric": "litellm.request.total_latency.distribution",
        "type": "distribution",
        "points": ((timestamp, (1.5,)),),
        "tags": ("env:test",),
    }
    logger.log_queue = [gauge, distribution]

    await logger.flush_queue()

    assert logger.log_queue == [distribution, late_count]

    responses[v1_url] = Response(202, json={"status": "ok"}, request=Request("POST", v1_url))
    await logger.flush_queue()

    assert logger.log_queue == []
    assert [call.args[0] for call in logger.async_client.post.call_args_list] == [v2_url, v1_url, v2_url, v1_url]
    second_v2 = json.loads(gzip.decompress(logger.async_client.post.call_args_list[2].kwargs["content"]))
    assert [s["metric"] for s in second_v2["series"]] == ["litellm.llm_api.request_count"]


@pytest.mark.asyncio
async def test_async_send_batch_empty_queue(clean_env):
    """Test that async_send_batch does nothing when queue is empty."""
    logger = DatadogMetricsLogger(start_periodic_flush=False)
    logger.async_client = AsyncMock()

    await logger.async_send_batch()

    assert not logger.async_client.post.called
