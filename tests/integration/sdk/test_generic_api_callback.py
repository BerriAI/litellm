from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator, Sequence
from typing import Final, TypedDict

import httpx
import pytest
from pydantic import TypeAdapter
from typing_extensions import ReadOnly

from litellm.integrations.generic_api.generic_api_callback import LOG_FORMAT_TYPES, GenericAPILogger
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from tests.integration._support.wire import Reply as WireReply, Request as WireRequest, wire_server


class _QueuedLog(TypedDict):
    id: ReadOnly[str]


_QUEUED_LOG: Final = TypeAdapter(_QueuedLog)
_LOG_BATCH: Final = TypeAdapter(tuple[_QueuedLog, ...])


def _batch_events(body: bytes, log_format: LOG_FORMAT_TYPES) -> tuple[_QueuedLog, ...]:
    match log_format:
        case "json_array":
            return _LOG_BATCH.validate_json(body)
        case "ndjson":
            return tuple(_QUEUED_LOG.validate_json(line) for line in body.splitlines())
        case "single":
            return (_QUEUED_LOG.validate_json(body),)


def _received_log_ids(requests: Sequence[WireRequest], log_format: LOG_FORMAT_TYPES) -> Iterator[str]:
    for request in requests:
        for event in _batch_events(request.body, log_format):
            yield event["id"]


@pytest.mark.asyncio
@pytest.mark.parametrize("log_format", ("json_array", "ndjson", "single"))
async def test_generic_api_keeps_failure_arriving_during_success_batch(log_format: LOG_FORMAT_TYPES) -> None:
    loop: Final = asyncio.get_running_loop()
    first_arrived: Final = asyncio.Event()
    release: Final = threading.Event()

    def respond(request: WireRequest) -> WireReply:
        events: Final = _batch_events(request.body, log_format)
        if any(event["id"] == "before" for event in events):
            loop.call_soon_threadsafe(first_arrived.set)
            assert release.wait(5), "The first HTTP batch was never released"
        return WireReply()

    with wire_server(respond) as wire:
        handler: Final = AsyncHTTPHandler(transport=httpx.AsyncHTTPTransport(trust_env=False))
        before_tasks: Final = frozenset(asyncio.all_tasks())
        logger: Final = GenericAPILogger(
            endpoint=wire.url,
            async_httpx_client=handler,
            log_format=log_format,
            batch_size=1000,
            flush_interval=3600,
        )
        owned_tasks: Final = frozenset(asyncio.all_tasks()) - before_tasks
        try:
            async with asyncio.TaskGroup() as flushes:
                await logger.async_log_success_event({"standard_logging_object": {"id": "before"}}, None, None, None)
                first_flush: Final = flushes.create_task(logger.flush_queue())
                try:
                    await first_arrived.wait()
                    await logger.async_log_failure_event({"standard_logging_object": {"id": "during"}}, None, None, None)
                finally:
                    release.set()
                await first_flush
                await logger.flush_queue()
                assert tuple(_received_log_ids(wire.drain(), log_format)) == ("before", "during")
        finally:
            release.set()
            for task in owned_tasks:
                task.cancel()
            await asyncio.gather(*owned_tasks, return_exceptions=True)
            await handler.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("log_format", ("json_array", "ndjson", "single"))
async def test_generic_api_concurrent_flushes_deliver_each_event_once(log_format: LOG_FORMAT_TYPES) -> None:
    with wire_server(lambda request: WireReply()) as wire:
        handler: Final = AsyncHTTPHandler(transport=httpx.AsyncHTTPTransport(trust_env=False))
        before_tasks: Final = frozenset(asyncio.all_tasks())
        logger: Final = GenericAPILogger(
            endpoint=wire.url,
            async_httpx_client=handler,
            log_format=log_format,
            batch_size=1000,
            flush_interval=3600,
        )
        owned_tasks: Final = frozenset(asyncio.all_tasks()) - before_tasks
        try:
            await logger.async_log_success_event({"standard_logging_object": {"id": "once"}}, None, None, None)
            await asyncio.gather(logger.flush_queue(), logger.async_send_batch())
            assert tuple(_received_log_ids(wire.drain(), log_format)) == ("once",)
        finally:
            for task in owned_tasks:
                task.cancel()
            await asyncio.gather(*owned_tasks, return_exceptions=True)
            await handler.close()
