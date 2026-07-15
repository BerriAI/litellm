from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from httpx import Request, Response

from litellm.integrations.datadog.datadog import DataDogLogger
from litellm.llms.custom_httpx.http_handler import MaskedHTTPStatusError
from litellm.types.integrations.datadog import DD_MAX_BATCH_SIZE, DatadogPayload


def _payloads(n):
    return [
        DatadogPayload(
            ddsource="litellm",
            ddtags="env:test",
            hostname="host",
            message=f'{{"event": {i}}}',
            service="svc",
            status="info",
        )
        for i in range(n)
    ]


def _raised_413():
    request = Request("POST", "https://example.com")
    response = Response(413, request=request, text="Payload Too Large")
    return MaskedHTTPStatusError(
        httpx.HTTPStatusError("413", request=request, response=response)
    )


def _make_send(max_ok, delivered, *, raise_413=True):
    """Datadog double: 413 batches larger than max_ok, 202 (recording delivery) otherwise."""

    async def _send(data):
        request = Request("POST", "https://example.com")
        if len(data) > max_ok:
            if raise_413:
                raise _raised_413()
            return Response(413, request=request, text="Payload Too Large")
        delivered.extend(event["message"] for event in data)
        return Response(202, request=request, text="Accepted")

    return _send


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
async def test_413_splits_oversized_batch_and_delivers_every_event(datadog_env):
    """A raised 413 (the real httpx path) halves the batch until each piece is accepted."""
    with patch("asyncio.create_task"):
        logger = DataDogLogger()

    logger.log_queue = _payloads(4)
    delivered: list = []
    logger.async_send_compressed_data = AsyncMock(side_effect=_make_send(1, delivered))

    await logger.async_send_batch()

    assert sorted(delivered) == [f'{{"event": {i}}}' for i in range(4)]
    assert logger.log_queue == []


@pytest.mark.asyncio
async def test_413_does_not_requeue_oversized_batch(datadog_env):
    """Regression for the infinite 413 loop: an undeliverable batch must not be re-queued."""
    with patch("asyncio.create_task"):
        logger = DataDogLogger()

    logger.log_queue = _payloads(4)
    logger.async_send_compressed_data = AsyncMock(side_effect=_make_send(0, []))

    await logger.async_send_batch()
    await logger.async_send_batch()

    assert logger.log_queue == []


@pytest.mark.asyncio
async def test_413_drops_single_oversized_event(datadog_env):
    with patch("asyncio.create_task"):
        logger = DataDogLogger()

    logger.log_queue = _payloads(1)
    send = AsyncMock(side_effect=_make_send(0, []))
    logger.async_send_compressed_data = send

    await logger.async_send_batch()

    assert send.await_count == 1
    assert logger.log_queue == []


@pytest.mark.asyncio
async def test_413_returned_response_also_splits(datadog_env):
    """Defensive path: a 413 returned (not raised) is handled the same way."""
    with patch("asyncio.create_task"):
        logger = DataDogLogger()

    logger.log_queue = _payloads(4)
    delivered: list = []
    logger.async_send_compressed_data = AsyncMock(
        side_effect=_make_send(1, delivered, raise_413=False)
    )

    await logger.async_send_batch()

    assert sorted(delivered) == [f'{{"event": {i}}}' for i in range(4)]
    assert logger.log_queue == []


@pytest.mark.asyncio
async def test_partial_delivery_then_transient_error_requeues_only_undelivered(
    datadog_env,
):
    """A transient error after a partial split delivery must not duplicate delivered events."""
    with patch("asyncio.create_task"):
        logger = DataDogLogger()

    logger.log_queue = _payloads(4)
    delivered: list = []

    async def _send(data):
        messages = [event["message"] for event in data]
        if len(data) > 2:
            raise _raised_413()
        if messages == ['{"event": 2}', '{"event": 3}']:
            raise RuntimeError("transient network error")
        delivered.extend(messages)
        return Response(
            202, request=Request("POST", "https://example.com"), text="Accepted"
        )

    logger.async_send_compressed_data = AsyncMock(side_effect=_send)

    await logger.async_send_batch()

    assert delivered == ['{"event": 0}', '{"event": 1}']
    assert [event["message"] for event in logger.log_queue] == [
        '{"event": 2}',
        '{"event": 3}',
    ]


@pytest.mark.asyncio
async def test_unexpected_non_202_status_requeues(datadog_env):
    """A non-413, non-202 response is treated as undelivered and re-queued."""
    with patch("asyncio.create_task"):
        logger = DataDogLogger()

    logger.log_queue = _payloads(2)
    logger.async_send_compressed_data = AsyncMock(
        return_value=Response(
            200, request=Request("POST", "https://example.com"), text="OK"
        )
    )

    await logger.async_send_batch()

    assert [event["message"] for event in logger.log_queue] == [
        '{"event": 0}',
        '{"event": 1}',
    ]


@pytest.mark.parametrize(
    "value, expected",
    [
        ("50", 50),
        ("1", 1),
        ("0", 1),
        ("-5", 1),
        (str(DD_MAX_BATCH_SIZE + 100), DD_MAX_BATCH_SIZE),
        ("not_an_int", DD_MAX_BATCH_SIZE),
    ],
)
def test_dd_batch_size_env_resolution(monkeypatch, value, expected):
    monkeypatch.setenv("DD_API_KEY", "test_api_key")
    monkeypatch.setenv("DD_SITE", "test.datadoghq.com")
    monkeypatch.setenv("DD_BATCH_SIZE", value)
    with patch("asyncio.create_task"):
        logger = DataDogLogger()
    assert logger.batch_size == expected


def test_dd_batch_size_defaults_to_max(monkeypatch):
    monkeypatch.setenv("DD_API_KEY", "test_api_key")
    monkeypatch.setenv("DD_SITE", "test.datadoghq.com")
    monkeypatch.delenv("DD_BATCH_SIZE", raising=False)
    with patch("asyncio.create_task"):
        logger = DataDogLogger()
    assert logger.batch_size == DD_MAX_BATCH_SIZE


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
