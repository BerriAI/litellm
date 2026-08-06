import asyncio
import json
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any, Final, Protocol, runtime_checkable

import httpx
from pydantic import TypeAdapter
from typing_extensions import TypedDict

from litellm.constants import (
    ANTHROPIC_MESSAGES_MAX_DETACHED_STREAM_DRAINS,
    ANTHROPIC_MESSAGES_STREAM_RELAY_QUEUE_MAXSIZE,
)
from litellm.litellm_core_utils.core_helpers import process_response_headers
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy.pass_through_endpoints.success_handler import (
    PassThroughEndpointLogging,
)
from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType
from litellm.types.utils import GenericStreamingChunk, ModelResponseStream

GLOBAL_PASS_THROUGH_SUCCESS_HANDLER_OBJ: Final = PassThroughEndpointLogging()

# asyncio holds only a weak reference to a bare create_task() result, so a
# fire-and-forget task can be garbage-collected mid-run. The upstream pump
# below must outlive the client-facing generator (which is closed on client
# disconnect), so root every pump task in a module-level set per the stdlib
# guidance and drop it again from the done callback.
_UPSTREAM_PUMP_TASKS: Final[set[asyncio.Task[None]]] = set()  # mutable-ok: stdlib strong-ref set for pump tasks

# Rooted set of pumps still draining upstream AFTER their client disconnected.
# Bounds how many detached drains run at once so a burst of slow/abandoned
# streams can't pin unbounded worker memory; a pump over the cap bills what it
# already collected instead of continuing to drain. Only ever touched from the
# event loop, so a plain set + len() check needs no lock.
_DETACHED_STREAM_DRAINS: Final[set[asyncio.Task[None]]] = set()  # mutable-ok: bounded strong-ref set, detached drains

INCOMPLETE_STREAM_ERROR_MESSAGE: Final = (
    "Provider stream ended before emitting a message_stop event; "
    "the response is incomplete and any partial content (e.g. tool_use input JSON) may be truncated."
)


def _is_message_stop_chunk(chunk: object) -> bool:
    if isinstance(chunk, dict):
        return chunk.get("type") == "message_stop"
    if isinstance(chunk, (bytes, bytearray)):
        return any(line == b"event: message_stop" for line in chunk.splitlines())
    return False


def _is_provider_error_chunk(chunk: object) -> bool:
    if isinstance(chunk, dict):
        return chunk.get("type") == "error"
    if isinstance(chunk, (bytes, bytearray)):
        return any(line == b"event: error" for line in chunk.splitlines())
    return False


def _is_terminal_stream_chunk(chunk: object) -> bool:
    return _is_message_stop_chunk(chunk) or _is_provider_error_chunk(chunk)


def _try_claim_detached_drain_slot() -> bool:
    """Claim a detached-drain slot for the current task, bounding concurrency.

    Returns True if a slot was claimed (the caller may keep draining upstream
    for billing) or False if the cap is already reached (the caller should stop
    and bill what it has). Only touched from the event loop, so the check +
    insert need no lock.
    """
    if len(_DETACHED_STREAM_DRAINS) >= ANTHROPIC_MESSAGES_MAX_DETACHED_STREAM_DRAINS:
        return False
    current_task: Final = asyncio.current_task()
    if current_task is not None:
        _DETACHED_STREAM_DRAINS.add(current_task)
        current_task.add_done_callback(_DETACHED_STREAM_DRAINS.discard)
    return True


def _incomplete_stream_error_sse_event() -> bytes:
    payload: Final = json.dumps(
        {
            "type": "error",
            "error": {"type": "api_error", "message": INCOMPLETE_STREAM_ERROR_MESSAGE},
        }
    )
    return f"event: error\ndata: {payload}\n\n".encode()


class AnthropicMessagesStreamHiddenParams(TypedDict):
    additional_headers: dict[str, str]


@runtime_checkable
class SupportsAclose(Protocol):
    async def aclose(self) -> None: ...


async def aclose_if_supported(stream: object) -> None:
    if isinstance(stream, SupportsAclose):
        await stream.aclose()


_RESPONSE_HEADERS_ADAPTER: Final[TypeAdapter[dict[str, str]]] = TypeAdapter(dict[str, str])


def anthropic_messages_stream_hidden_params(
    response_headers: httpx.Headers,
) -> AnthropicMessagesStreamHiddenParams:
    return AnthropicMessagesStreamHiddenParams(
        additional_headers=_RESPONSE_HEADERS_ADAPTER.validate_python(process_response_headers(response_headers))
    )


class AnthropicMessagesStreamingResponse:
    """
    Wraps the /v1/messages SSE byte stream so upstream provider response
    headers (e.g. Bedrock's x-amzn-requestid / x-amzn-trace-id) survive as
    ``_hidden_params["additional_headers"]``, which the proxy forwards to
    clients as ``llm_provider-*`` response headers. Bare async generators
    cannot carry attributes, so header context was previously dropped.
    """

    def __init__(
        self,
        completion_stream: AsyncIterator[bytes],
        hidden_params: AnthropicMessagesStreamHiddenParams,
    ) -> None:
        self.completion_stream = completion_stream
        self._hidden_params = hidden_params

    def __aiter__(self) -> "AnthropicMessagesStreamingResponse":
        return self

    async def __anext__(self) -> bytes:
        return await self.completion_stream.__anext__()

    async def aclose(self) -> None:
        await aclose_if_supported(self.completion_stream)


class BaseAnthropicMessagesStreamingIterator:
    """
    Base class for Anthropic Messages streaming iterators that provides common logic
    for streaming response handling and logging.
    """

    def __init__(
        self,
        litellm_logging_obj: LiteLLMLoggingObj,
        request_body: dict,
    ):
        self.litellm_logging_obj = litellm_logging_obj
        self.request_body = request_body
        self.start_time = datetime.now()
        self.completion_start_time: datetime | None = None

    async def _handle_streaming_logging(self, collected_chunks: list[bytes]):
        """Handle the logging after all chunks have been collected."""
        from litellm.proxy.pass_through_endpoints.streaming_handler import (
            PassThroughStreamingHandler,
        )

        end_time: Final = datetime.now()
        # Set completion_start_time so TTFT is calculated from the first
        # chunk rather than falling back to end_time in async_success_handler.
        if self.completion_start_time is not None:
            self.litellm_logging_obj.completion_start_time = self.completion_start_time
            self.litellm_logging_obj.model_call_details["completion_start_time"] = self.completion_start_time
        asyncio.create_task(
            PassThroughStreamingHandler._route_streaming_logging_to_handler(
                litellm_logging_obj=self.litellm_logging_obj,
                passthrough_success_handler_obj=GLOBAL_PASS_THROUGH_SUCCESS_HANDLER_OBJ,
                url_route="/v1/messages",
                request_body=self.request_body or {},
                endpoint_type=EndpointType.ANTHROPIC,
                start_time=self.start_time,
                raw_bytes=collected_chunks,
                end_time=end_time,
            )
        )

    def get_async_streaming_response_iterator(
        self,
        httpx_response,
        request_body: dict,
        litellm_logging_obj: LiteLLMLoggingObj,
    ) -> AsyncIterator:
        """Helper function to handle Anthropic streaming responses using the existing logging handlers"""
        from litellm.proxy.pass_through_endpoints.streaming_handler import (
            PassThroughStreamingHandler,
        )

        # Use the existing streaming handler for Anthropic
        return PassThroughStreamingHandler.chunk_processor(
            response=httpx_response,
            request_body=request_body,
            litellm_logging_obj=litellm_logging_obj,
            endpoint_type=EndpointType.ANTHROPIC,
            start_time=self.start_time,
            passthrough_success_handler_obj=GLOBAL_PASS_THROUGH_SUCCESS_HANDLER_OBJ,
            url_route="/v1/messages",
        )

    def _convert_chunk_to_sse_format(self, chunk: dict | Any) -> bytes:
        """
        Convert a chunk to Server-Sent Events format.

        This method should be overridden by subclasses if they need custom
        chunk formatting logic.
        """
        if isinstance(chunk, dict):
            event_type: Final[str] = str(chunk.get("type", "message"))
            payload: Final = f"event: {event_type}\ndata: {json.dumps(chunk)}\n\n"
            return payload.encode()
        else:
            # For non-dict chunks, return as is
            return chunk

    async def async_sse_wrapper(
        self,
        completion_stream: AsyncIterator[bytes | GenericStreamingChunk | ModelResponseStream | dict],
    ) -> AsyncIterator[bytes]:
        """
        Generic async SSE wrapper that converts streaming chunks to SSE format
        and handles logging.

        The upstream read runs in a detached background task (``_pump_upstream``)
        so that a client disconnect tears down only this client-facing generator,
        never the upstream drain + billing. The provider (e.g. Bedrock) keeps
        generating and billing the full response regardless of the client, so
        draining it to completion is what lets spend tracking see the real
        terminal ``message_delta`` / ``message_stop`` usage instead of a
        truncated placeholder count.

        Chunks reach the client through a bounded queue. While the client is
        connected the pump blocks on a full queue (racing the disconnect
        signal), so a slow reader throttles the upstream read exactly as the old
        direct ``yield`` did instead of letting the whole response buffer in
        memory. Once the client goes away the pump stops enqueueing and only
        keeps a single ``collected_chunks`` copy for billing, and the number of
        such post-disconnect drains running at once is capped so client behavior
        can't create unbounded worker state; over the cap the pump bills what it
        has rather than draining further. Detached-drain lifetime is otherwise
        bounded by the upstream stream/read timeout.

        An upstream failure (Bedrock read / decode / chunk-conversion error)
        that happens while the client is still connected is forwarded through
        the queue and re-raised here, so the original provider exception (and
        its status) reaches the proxy's failure handling unchanged rather than
        being masked by a generic incomplete-stream event.

        This method provides the common logic for both Anthropic and Bedrock implementations.
        """
        queue: Final[asyncio.Queue[bytes | None | BaseException]] = asyncio.Queue(
            maxsize=ANTHROPIC_MESSAGES_STREAM_RELAY_QUEUE_MAXSIZE
        )
        client_detached: Final = asyncio.Event()

        pump_task: Final = asyncio.create_task(self._pump_upstream_to_queue(completion_stream, queue, client_detached))
        _UPSTREAM_PUMP_TASKS.add(pump_task)
        pump_task.add_done_callback(_UPSTREAM_PUMP_TASKS.discard)

        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            # Client-facing generator is being torn down (normal end, a
            # re-raised upstream error, or a disconnect GeneratorExit). Signal
            # the pump to stop enqueueing and unblock any backpressure-blocked
            # put; the pump then either finishes billing or drains detached
            # (subject to the cap) for accurate usage.
            client_detached.set()

    async def _bill_collected_chunks(
        self,
        collected_chunks: list[bytes],  # mutable-ok: SSE buffer forwarded to list-typed _handle_streaming_logging
    ) -> None:
        from litellm._logging import verbose_proxy_logger

        try:
            await self._handle_streaming_logging(collected_chunks)
        except Exception as exc:  # noqa: BLE001  # billing is best-effort; never crash the pump
            verbose_proxy_logger.warning(
                "async_sse_wrapper billing failed after %d chunks: %s(%s)",
                len(collected_chunks),
                type(exc).__name__,
                exc,
            )

    @staticmethod
    async def _enqueue_for_client(
        queue: "asyncio.Queue[bytes | None | BaseException]",
        client_detached: "asyncio.Event",
        item: bytes | None | BaseException,
    ) -> bool:
        """Deliver one item to the client, applying backpressure.

        Returns True if the item was queued, False if the client disconnected
        before there was room (the item is then dropped, since a gone client
        can't receive it). Never blocks once the client has detached.
        """
        if client_detached.is_set():
            return False
        try:
            queue.put_nowait(item)
            return True
        except asyncio.QueueFull:
            pass
        put_task: Final = asyncio.ensure_future(queue.put(item))
        detached_task: Final = asyncio.ensure_future(client_detached.wait())
        try:
            await asyncio.wait(frozenset((put_task, detached_task)), return_when=asyncio.FIRST_COMPLETED)
        finally:
            if not detached_task.done():
                detached_task.cancel()
        if put_task.done() and not put_task.cancelled():
            return True
        put_task.cancel()
        return False

    async def _pump_upstream_to_queue(
        self,
        completion_stream: AsyncIterator[bytes | GenericStreamingChunk | ModelResponseStream | dict],
        queue: "asyncio.Queue[bytes | None | BaseException]",
        client_detached: "asyncio.Event",
    ) -> None:
        """Drain the whole upstream into ``queue`` (backpressured) and bill once.

        Runs detached so a client disconnect can't interrupt the upstream read;
        see ``async_sse_wrapper`` for the full rationale. Returns after billing.
        """
        from litellm._logging import verbose_proxy_logger

        collected_chunks: Final[list[bytes]] = []  # mutable-ok: SSE billing buffer appended to across the drain
        saw_terminal_event = False  # rebind-ok: accumulates across the upstream loop
        draining_detached = False  # rebind-ok: set once this pump claims a detached-drain slot
        try:
            async for chunk in completion_stream:
                if self.completion_start_time is None:
                    self.completion_start_time = datetime.now()
                saw_terminal_event = saw_terminal_event or _is_terminal_stream_chunk(chunk)
                encoded_chunk = self._convert_chunk_to_sse_format(chunk)
                collected_chunks.append(encoded_chunk)
                if not client_detached.is_set():
                    await self._enqueue_for_client(queue, client_detached, encoded_chunk)
                    continue
                # Client has gone: keep draining only to reach the terminal usage
                # event for billing, but claim a detached-drain slot first; over
                # the cap, bill what we have rather than pinning more memory.
                if not draining_detached:
                    if not _try_claim_detached_drain_slot():
                        verbose_proxy_logger.warning(
                            "async_sse_wrapper: detached-drain cap (%d) reached; billing %d partial "
                            "chunks without draining the rest of the upstream stream",
                            ANTHROPIC_MESSAGES_MAX_DETACHED_STREAM_DRAINS,
                            len(collected_chunks),
                        )
                        await self._bill_collected_chunks(collected_chunks)
                        return
                    draining_detached = True
        except Exception as exc:  # noqa: BLE001  # forward the provider error to a live client, else salvage spend
            await self._handle_pump_upstream_error(queue, client_detached, collected_chunks, exc)
            return

        if not client_detached.is_set():
            if not saw_terminal_event:
                await self._enqueue_for_client(queue, client_detached, _incomplete_stream_error_sse_event())
            await self._enqueue_for_client(queue, client_detached, None)
        await self._bill_collected_chunks(collected_chunks)

    async def _handle_pump_upstream_error(
        self,
        queue: "asyncio.Queue[bytes | None | BaseException]",
        client_detached: "asyncio.Event",
        collected_chunks: list[bytes],  # mutable-ok: SSE buffer forwarded to list-typed _bill_collected_chunks
        exc: BaseException,
    ) -> None:
        from litellm._logging import verbose_proxy_logger

        if not client_detached.is_set() and await self._enqueue_for_client(queue, client_detached, exc):
            # Preserve the provider-specific failure: the client-facing
            # generator re-raises it and the proxy's failure handling (status
            # code, post_call_failure_hook) runs. That path owns logging, so
            # don't also success-bill.
            return
        # Client already gone (or disconnected before the error reached it): no
        # failure hook will run, so salvage the partial spend instead of
        # dropping the request entirely.
        verbose_proxy_logger.warning(
            "async_sse_wrapper upstream pump failed after client disconnect (%d chunks): %s(%s)",
            len(collected_chunks),
            type(exc).__name__,
            exc,
        )
        await self._bill_collected_chunks(collected_chunks)
