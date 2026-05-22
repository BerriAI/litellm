from unittest.mock import AsyncMock, Mock, patch

import pytest
from httpx import Request, Response

from litellm.integrations.datadog.datadog import DataDogLogger
from litellm.types.integrations.datadog import DatadogPayload


@pytest.fixture
def datadog_env(monkeypatch):
    monkeypatch.setenv("DD_API_KEY", "test_api_key")
    monkeypatch.setenv("DD_SITE", "test.datadoghq.com")


@pytest.mark.asyncio
async def test_async_send_batch_keeps_events_appended_during_send(datadog_env):
    with patch("asyncio.create_task"):
        logger = DataDogLogger()

    logger.log_queue = [
        DatadogPayload(
            ddsource="litellm",
            ddtags="env:test",
            hostname="host",
            message=f'{{"event": {i}}}',
            service="svc",
            status="info",
        )
        for i in range(2)
    ]

    async def _mock_send(data):
        logger.log_queue.append(
            DatadogPayload(
                ddsource="litellm",
                ddtags="env:test",
                hostname="host",
                message='{"event": 2}',
                service="svc",
                status="info",
            )
        )
        return Response(
            202, request=Request("POST", "https://example.com"), text="Accepted"
        )

    logger.async_send_compressed_data = AsyncMock(side_effect=_mock_send)

    await logger.async_send_batch()

    assert logger.async_send_compressed_data.await_count == 1
    sent_batch = logger.async_send_compressed_data.await_args.args[0]
    assert len(sent_batch) == 2
    assert len(logger.log_queue) == 1
    assert logger.log_queue[0]["message"] == '{"event": 2}'


@pytest.mark.asyncio
async def test_failure_hook_threshold_flush_uses_flush_queue(datadog_env):
    with patch("asyncio.create_task"):
        logger = DataDogLogger()

    logger.batch_size = 1
    logger.flush_queue = AsyncMock()

    await logger.async_post_call_failure_hook(
        request_data={},
        original_exception=Exception("boom"),
        user_api_key_dict=type("UserKey", (), {})(),
        traceback_str="trace",
    )

    logger.flush_queue.assert_awaited_once()


@pytest.mark.asyncio
async def test_async_send_batch_requeues_events_on_413(datadog_env):
    with patch("asyncio.create_task"):
        logger = DataDogLogger()

    logger.log_queue = [
        DatadogPayload(
            ddsource="litellm",
            ddtags="env:test",
            hostname="host",
            message=f'{{"event": {i}}}',
            service="svc",
            status="info",
        )
        for i in range(2)
    ]

    logger.async_send_compressed_data = AsyncMock(
        return_value=Response(
            413,
            request=Request("POST", "https://example.com"),
            text="Payload Too Large",
        )
    )

    await logger.async_send_batch()

    assert logger.async_send_compressed_data.await_count == 1
    assert len(logger.log_queue) == 2
    assert [event["message"] for event in logger.log_queue] == [
        '{"event": 0}',
        '{"event": 1}',
    ]


@pytest.mark.asyncio
async def test_async_send_batch_handles_empty_queue(datadog_env):
    with patch("asyncio.create_task"):
        logger = DataDogLogger()

    logger.log_queue = []
    logger.async_send_compressed_data = AsyncMock()

    await logger.async_send_batch()

    logger.async_send_compressed_data.assert_not_awaited()


@pytest.mark.asyncio
async def test_async_send_batch_requeues_events_on_exception(datadog_env):
    with patch("asyncio.create_task"):
        logger = DataDogLogger()

    logger.log_queue = [
        DatadogPayload(
            ddsource="litellm",
            ddtags="env:test",
            hostname="host",
            message=f'{{"event": {i}}}',
            service="svc",
            status="info",
        )
        for i in range(2)
    ]

    logger.async_send_compressed_data = AsyncMock(side_effect=RuntimeError("boom"))

    await logger.async_send_batch()

    assert [event["message"] for event in logger.log_queue] == [
        '{"event": 0}',
        '{"event": 1}',
    ]


@pytest.mark.asyncio
async def test_log_async_event_threshold_flush_uses_flush_queue(datadog_env):
    with patch("asyncio.create_task"):
        logger = DataDogLogger()

    logger.batch_size = 1
    logger.flush_queue = AsyncMock()
    logger.create_datadog_logging_payload = Mock(
        return_value=DatadogPayload(
            ddsource="litellm",
            ddtags="env:test",
            hostname="host",
            message='{"event": 0}',
            service="svc",
            status="info",
        )
    )

    await logger._log_async_event(
        kwargs={},
        response_obj={},
        start_time=None,
        end_time=None,
    )

    logger.flush_queue.assert_awaited_once()


@pytest.mark.asyncio
async def test_flush_queue_updates_last_flush_time(datadog_env):
    with patch("asyncio.create_task"):
        logger = DataDogLogger()

    logger.log_queue = [
        DatadogPayload(
            ddsource="litellm",
            ddtags="env:test",
            hostname="host",
            message='{"event": 0}',
            service="svc",
            status="info",
        )
    ]
    logger.last_flush_time = 0

    async def _successful_send():
        logger.log_queue = []

    logger.async_send_batch = AsyncMock(side_effect=_successful_send)

    await logger.flush_queue()

    logger.async_send_batch.assert_awaited_once()
    assert logger.last_flush_time > 0


@pytest.mark.asyncio
async def test_flush_queue_does_not_update_last_flush_time_when_send_requeues(
    datadog_env,
):
    with patch("asyncio.create_task"):
        logger = DataDogLogger()

    logger.log_queue = [
        DatadogPayload(
            ddsource="litellm",
            ddtags="env:test",
            hostname="host",
            message='{"event": 0}',
            service="svc",
            status="info",
        )
    ]
    logger.last_flush_time = 123.0

    async def _requeue_batch():
        logger.log_queue = [
            DatadogPayload(
                ddsource="litellm",
                ddtags="env:test",
                hostname="host",
                message='{"event": 0}',
                service="svc",
                status="info",
            )
        ]

    logger.async_send_batch = AsyncMock(side_effect=_requeue_batch)

    await logger.flush_queue()

    logger.async_send_batch.assert_awaited_once()
    assert logger.last_flush_time == 123.0


@pytest.mark.asyncio
async def test_flush_queue_returns_without_lock(datadog_env):
    with patch("asyncio.create_task"):
        logger = DataDogLogger()

    logger.flush_lock = None
    logger.log_queue = [
        DatadogPayload(
            ddsource="litellm",
            ddtags="env:test",
            hostname="host",
            message='{"event": 0}',
            service="svc",
            status="info",
        )
    ]
    logger.async_send_batch = AsyncMock()

    await logger.flush_queue()

    logger.async_send_batch.assert_not_awaited()
