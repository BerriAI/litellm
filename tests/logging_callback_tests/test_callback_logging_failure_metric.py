import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

from unittest.mock import AsyncMock

import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger


class FakePrometheusLogger(CustomLogger):
    """Stand-in for the prometheus logger that records failure-metric increments."""

    def __init__(self):
        super().__init__()
        self.failures: list[str] = []

    def increment_callback_logging_failure(self, callback_name: str):
        self.failures.append(callback_name)


@pytest.fixture(autouse=True)
def reset_callbacks():
    original = litellm.callbacks
    litellm.callbacks = []
    yield
    litellm.callbacks = original


def _register(prometheus, logger):
    litellm.callbacks = [prometheus, logger]


def _make_payload():
    return {"id": "test", "response_cost": 1.0, "messages": [], "response": {}}


@pytest.mark.asyncio
async def test_pubsub_inline_flush_failure_increments_metric():
    from litellm.integrations.gcs_pubsub.pub_sub import GcsPubSubLogger

    prometheus = FakePrometheusLogger()
    logger = GcsPubSubLogger(
        project_id="P", topic_id="T", flush_interval=999, batch_size=1
    )
    logger.construct_request_headers = AsyncMock(return_value={})
    failing_post = AsyncMock()
    failing_post.return_value.status_code = 500
    failing_post.return_value.text = "boom"
    logger.async_httpx_client.post = failing_post
    _register(prometheus, logger)

    await logger.async_log_success_event(
        kwargs={"standard_logging_object": _make_payload()},
        response_obj=None,
        start_time=0,
        end_time=0,
    )

    assert prometheus.failures == ["GcsPubSubLogger"]
    # queue retained for retry
    assert len(logger.log_queue) == 1


@pytest.mark.asyncio
async def test_failure_metric_dedup_and_reset():
    from litellm.integrations.gcs_pubsub.pub_sub import GcsPubSubLogger

    prometheus = FakePrometheusLogger()
    logger = GcsPubSubLogger(
        project_id="P", topic_id="T", flush_interval=999, batch_size=1
    )
    logger.construct_request_headers = AsyncMock(return_value={})
    failing_post = AsyncMock()
    failing_post.return_value.status_code = 500
    failing_post.return_value.text = "boom"
    logger.async_httpx_client.post = failing_post
    _register(prometheus, logger)

    payload = _make_payload()

    # two consecutive failing flushes -> only one increment
    await logger.async_log_success_event(
        kwargs={"standard_logging_object": payload},
        response_obj=None,
        start_time=0,
        end_time=0,
    )
    await logger.flush_queue()
    assert prometheus.failures == ["GcsPubSubLogger"]

    # a success resets failure state
    ok_post = AsyncMock()
    ok_post.return_value.status_code = 200
    ok_post.return_value.text = "ok"
    ok_post.return_value.json = lambda: {}
    logger.async_httpx_client.post = ok_post
    await logger.flush_queue()
    assert logger._is_in_failure_state is False

    # next failure increments again
    logger.async_httpx_client.post = failing_post
    logger.log_queue.append(payload)
    await logger.flush_queue()
    assert prometheus.failures == ["GcsPubSubLogger", "GcsPubSubLogger"]


@pytest.mark.asyncio
async def test_generic_api_inline_flush_failure_increments_metric():
    from litellm.integrations.generic_api.generic_api_callback import GenericAPILogger

    prometheus = FakePrometheusLogger()
    logger = GenericAPILogger(
        endpoint="https://example.invalid",
        headers={},
        flush_interval=999,
        batch_size=1,
    )
    logger._post_with_retries = AsyncMock(side_effect=Exception("network down"))
    _register(prometheus, logger)

    await logger.async_log_success_event(
        kwargs={"standard_logging_object": _make_payload()},
        response_obj=None,
        start_time=0,
        end_time=0,
    )

    assert prometheus.failures == ["GenericAPILogger"]
    assert len(logger.log_queue) == 1


@pytest.mark.asyncio
async def test_datadog_flush_failure_increments_and_retains_queue():
    os.environ["DD_API_KEY"] = "fake"
    os.environ["DD_SITE"] = "us5.datadoghq.com"
    from litellm.integrations.datadog.datadog import DataDogLogger

    prometheus = FakePrometheusLogger()
    logger = DataDogLogger(flush_interval=999)
    logger.async_send_compressed_data = AsyncMock(side_effect=Exception("dd down"))
    _register(prometheus, logger)

    logger.log_queue = [{"id": "1"}]
    await logger.flush_queue()

    assert prometheus.failures == ["DataDogLogger"]
    assert len(logger.log_queue) == 1
