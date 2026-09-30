import asyncio
import json
import math
import ssl as ssl_module
from collections.abc import AsyncIterator, Mapping
from contextlib import AbstractAsyncContextManager, suppress
from itertools import count
from types import MappingProxyType
from typing import Final, Protocol, cast
from urllib.parse import urlsplit, urlunsplit

from pydantic import TypeAdapter, ValidationError
from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm._logging import redact_secrets
from litellm.constants import (
    OPENAI_LIVE_SESSION_CLOSE_TIMEOUT_SECONDS,
    REALTIME_SESSION_SUCCESS_LOGGED_KEY,
    REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES,
)
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER, LoggingWorker
from litellm.litellm_core_utils.realtime_errors import (
    close_after_upstream_handshake_refusal,
    realtime_error_event,
    websocket_close_reason,
)
from litellm.litellm_core_utils.realtime_streaming import backend_close_from
from litellm.llms.openai.realtime.handler import OpenAIRealtime
from litellm.types.llms.openai import OpenAIRealtimeStreamList
from litellm.types.utils import LiteLLMRealtimeStreamLoggingObject, Usage

_LIVE_EVENT_FRAME_ADAPTER: Final = TypeAdapter(dict[str, object])
_LIVE_TERMINAL_RESPONSE_EVENT_TYPES: Final = frozenset({"response.completed", "response.incomplete", "response.failed"})
_LIVE_USAGE_EVENT_TYPES: Final = frozenset({"session.usage.updated", "session.closed"})
_LIVE_WEBSOCKET_SCHEMES: Final = MappingProxyType({"https": "wss", "http": "ws"})
_LIVE_SESSION_CLOSE_FRAME: Final = '{"type": "session.close"}'


class _LiveSessionModel(TypedDict):
    model: ReadOnly[str]


class _LiveSessionStartedEvent(TypedDict):
    type: ReadOnly[str]
    session: NotRequired[ReadOnly[_LiveSessionModel]]


class _LiveUsageSeconds(TypedDict):
    seconds: ReadOnly[object]


class _LiveUsageEvent(TypedDict):
    type: ReadOnly[str]
    usage: NotRequired[ReadOnly[_LiveUsageSeconds]]


class _LiveResponseSummary(TypedDict):
    id: ReadOnly[str]
    model: ReadOnly[object]
    usage: ReadOnly[object]


class _LiveTerminalResponseEvent(TypedDict):
    type: ReadOnly[str]
    response: ReadOnly[_LiveResponseSummary]


class _LiveResponseEvent(TypedDict):
    type: ReadOnly[str]
    event: ReadOnly[_LiveTerminalResponseEvent]


class _LivePreCallInput(TypedDict):
    session_start: ReadOnly[Mapping[str, object]]


class _LivePreCallArgs(TypedDict):
    api_base: ReadOnly[str]
    headers: ReadOnly[Mapping[str, str]]
    complete_input_dict: ReadOnly[_LivePreCallInput]


class LiveClientWebSocket(Protocol):
    async def receive_text(self) -> str: ...

    async def send_text(self, data: str) -> None: ...

    async def close(self, code: int = 1000, reason: str | None = None) -> None: ...


class LiveBackendWebSocket(Protocol):
    async def send(self, data: str) -> None: ...

    async def recv(self) -> str | bytes: ...

    async def close(self) -> None: ...


class LiveWebSocketConnector(Protocol):
    def __call__(
        self,
        url: str,
        *,
        additional_headers: Mapping[str, str],
        max_size: int | None,
        ssl: bool | str | ssl_module.SSLContext | None,
    ) -> AbstractAsyncContextManager[LiveBackendWebSocket]: ...


def _openai_live_websocket_connect(
    url: str,
    *,
    additional_headers: Mapping[str, str],
    max_size: int | None,
    ssl: bool | str | ssl_module.SSLContext | None,
) -> AbstractAsyncContextManager[LiveBackendWebSocket]:
    import websockets

    return websockets.connect(
        url,
        additional_headers=additional_headers,
        max_size=max_size,
        ssl=ssl,
    )


def _is_finite_usage_seconds(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and (not isinstance(value, float) or math.isfinite(value))
    )


def _session_started_event(event: Mapping[str, object]) -> _LiveSessionStartedEvent:
    session: Final = event.get("session")
    session_model: Final = session.get("model") if isinstance(session, Mapping) else None
    if isinstance(session_model, str):
        started_with_model: Final[_LiveSessionStartedEvent] = {
            "type": "session.started",
            "session": {"model": session_model},
        }
        return started_with_model
    started: Final[_LiveSessionStartedEvent] = {"type": "session.started"}
    return started


def _usage_event(event: Mapping[str, object], event_type: str) -> _LiveUsageEvent:
    usage: Final = event.get("usage")
    seconds: Final = usage.get("seconds") if isinstance(usage, Mapping) else None
    if _is_finite_usage_seconds(seconds):
        usage_with_seconds: Final[_LiveUsageEvent] = {"type": event_type, "usage": {"seconds": seconds}}
        return usage_with_seconds
    usage_without_seconds: Final[_LiveUsageEvent] = {"type": event_type}
    return usage_without_seconds


def _terminal_response_event(event: Mapping[str, object]) -> _LiveResponseEvent | None:
    nested_event: Final = event.get("event")
    if not isinstance(nested_event, Mapping):
        return None
    nested_event_type: Final = nested_event.get("type")
    if not isinstance(nested_event_type, str) or nested_event_type not in _LIVE_TERMINAL_RESPONSE_EVENT_TYPES:
        return None
    response: Final = nested_event.get("response")
    response_id: Final = response.get("id") if isinstance(response, Mapping) else None
    if not isinstance(response_id, str) or not isinstance(response, Mapping):
        return None
    response_event: Final[_LiveResponseEvent] = {
        "type": "response.event",
        "event": {
            "type": nested_event_type,
            "response": {
                "id": response_id,
                "model": response.get("model"),
                "usage": response.get("usage"),
            },
        },
    }
    return response_event


def _metering_event(message: str) -> Mapping[str, object] | None:
    try:
        event: Final = _LIVE_EVENT_FRAME_ADAPTER.validate_json(message)
    except ValidationError:
        return None
    event_type: Final = event.get("type")
    if event_type == "session.started":
        return _session_started_event(event)
    if isinstance(event_type, str) and event_type in _LIVE_USAGE_EVENT_TYPES:
        return _usage_event(event, event_type)
    if event_type == "response.event":
        return _terminal_response_event(event)
    return None


def _is_session_close_message(message: str) -> bool:
    try:
        event: Final = _LIVE_EVENT_FRAME_ADAPTER.validate_json(message)
    except ValidationError:
        return False
    return event.get("type") == "session.close"


_RetainedEvent = tuple[int, Mapping[str, object]]


class _LiveEventRetention:
    def __init__(self) -> None:
        self._event_sequence = count()
        self._session_started: _RetainedEvent | None = None
        self._latest_usage: _RetainedEvent | None = None
        self._session_closed: _RetainedEvent | None = None
        self._response_events: dict[str, _RetainedEvent] = {}  # mutable-ok: one insert per response id, no copy

    def retain(self, event: Mapping[str, object]) -> None:
        retained_item: Final = (next(self._event_sequence), event)
        event_type: Final = event.get("type")
        if event_type == "session.started":
            if self._session_started is None:
                self._session_started = retained_item
            return
        if event_type == "session.usage.updated":
            usage: Final = event.get("usage")
            seconds: Final = usage.get("seconds") if isinstance(usage, Mapping) else None
            if _is_finite_usage_seconds(seconds):
                self._latest_usage = retained_item
            return
        if event_type == "session.closed":
            self._session_closed = retained_item
            return
        nested_event: Final = event.get("event")
        response: Final = nested_event.get("response") if isinstance(nested_event, Mapping) else None
        response_id: Final = response.get("id") if isinstance(response, Mapping) else None
        if isinstance(response_id, str):
            self._response_events.setdefault(response_id, retained_item)

    def results(self) -> tuple[Mapping[str, object], ...]:
        retained_items: Final = tuple(
            item for item in (self._session_started, self._latest_usage, self._session_closed) if item is not None
        ) + tuple(self._response_events.values())
        return tuple(event for _, event in sorted(retained_items, key=lambda item: item[0]))


async def _forward_live_client_frames(
    *,
    websocket: LiveClientWebSocket,
    backend_websocket: LiveBackendWebSocket,
    client_gone: asyncio.Event,
    client_close_sent: asyncio.Event,
) -> None:
    async for message in _live_client_messages(websocket, client_gone):
        await backend_websocket.send(message)
        if _is_session_close_message(message):
            client_close_sent.set()


async def _live_client_messages(
    websocket: LiveClientWebSocket,
    client_gone: asyncio.Event,
) -> AsyncIterator[str]:
    from starlette.websockets import WebSocketDisconnect

    try:
        while True:
            yield await websocket.receive_text()
    except WebSocketDisconnect:
        client_gone.set()


async def _forward_live_backend_frames(
    *,
    websocket: LiveClientWebSocket,
    backend_websocket: LiveBackendWebSocket,
    client_gone: asyncio.Event,
    event_retention: _LiveEventRetention,
) -> None:
    from starlette.websockets import WebSocketDisconnect
    from websockets.exceptions import ConnectionClosed

    async for message in _live_backend_messages(websocket, backend_websocket):
        if (event := _metering_event(message)) is not None:
            event_retention.retain(event)
        if client_gone.is_set():
            continue
        try:
            await websocket.send_text(message)
        except (WebSocketDisconnect, RuntimeError, ConnectionClosed, OSError):
            client_gone.set()


async def _live_backend_messages(
    websocket: LiveClientWebSocket,
    backend_websocket: LiveBackendWebSocket,
) -> AsyncIterator[str]:
    from starlette.websockets import WebSocketDisconnect
    from websockets.exceptions import ConnectionClosed

    try:
        while True:
            yield _decode_live_websocket_frame(await backend_websocket.recv())
    except ConnectionClosed as error:
        upstream_close: Final = backend_close_from(error)
        with suppress(WebSocketDisconnect, RuntimeError, ConnectionClosed, OSError):
            await websocket.close(code=upstream_close.code, reason=upstream_close.reason)


def _decode_live_websocket_frame(frame: str | bytes) -> str:
    return frame.decode("utf-8") if isinstance(frame, bytes) else frame


class OpenAILiveSessions(OpenAIRealtime):
    def __init__(
        self,
        logging_worker: LoggingWorker = GLOBAL_LOGGING_WORKER,
        websocket_connector: LiveWebSocketConnector | None = None,
    ) -> None:
        super().__init__()
        self._logging_worker = logging_worker
        self._websocket_connector = (
            websocket_connector if websocket_connector is not None else _openai_live_websocket_connect
        )

    @staticmethod
    def _construct_live_url(api_base: str) -> str:
        parsed_url: Final = urlsplit(api_base)
        websocket_scheme: Final = _LIVE_WEBSOCKET_SCHEMES.get(parsed_url.scheme, parsed_url.scheme)
        return urlunsplit((websocket_scheme, parsed_url.netloc, "/v1/live/sessions", "", ""))

    async def async_live_session(
        self,
        *,
        model: str,
        websocket: LiveClientWebSocket,
        logging_obj: LiteLLMLogging,
        session_start: Mapping[str, object],
        api_base: str | None,
        api_key: str | None,
    ) -> None:
        if api_key is None:
            raise ValueError("api_key is required for OpenAI Live session calls")
        from websockets.exceptions import InvalidStatus

        resolved_api_base: Final = api_base or self._get_default_api_base()
        url: Final = self._construct_live_url(resolved_api_base)
        headers: Final = self._get_additional_headers(api_key)
        ssl_config: Final = self._get_ssl_config(url)
        session: Final = session_start.get("session")
        if not isinstance(session, Mapping):
            raise ValueError("session_start.session must be a mapping")
        rewritten_start: Final = {  # mutable-ok: json.dumps frame
            **session_start,
            "session": {**session, "model": model},  # mutable-ok: json.dumps frame
        }
        upstream_start: Final = json.dumps(rewritten_start)
        pre_call_args: Final[_LivePreCallArgs] = {
            "api_base": url,
            "headers": headers,
            "complete_input_dict": {"session_start": rewritten_start},
        }
        event_retention: Final = _LiveEventRetention()
        try:
            logging_obj.pre_call(input=None, api_key=api_key, additional_args=pre_call_args)
            async with self._websocket_connector(
                url,
                additional_headers=headers,
                max_size=REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES,
                ssl=ssl_config,
            ) as backend_websocket:
                upstream_connected: Final = True
                try:
                    await backend_websocket.send(upstream_start)
                    await self._bidirectional_forward(
                        websocket=websocket,
                        backend_websocket=backend_websocket,
                        event_retention=event_retention,
                    )
                finally:
                    if upstream_connected:
                        retained_events: Final = event_retention.results()
                        logging_result: Final = LiteLLMRealtimeStreamLoggingObject(
                            usage=Usage(),
                            results=cast(OpenAIRealtimeStreamList, list(retained_events)),  # mutable-ok: list field
                        )
                        try:
                            self._logging_worker.ensure_initialized_and_enqueue(
                                logging_obj.dispatch_success_handlers(logging_result, prefer_async_handlers=True)
                            )
                        finally:
                            logging_obj.model_call_details[REALTIME_SESSION_SUCCESS_LOGGED_KEY] = True
        except InvalidStatus as error:
            await close_after_upstream_handshake_refusal(websocket, error.response.status_code)
        except Exception as error:
            redacted_error: Final = redact_secrets(str(error))
            with suppress(Exception):
                await websocket.send_text(realtime_error_event(redacted_error, error_type="server_error"))
            try:
                await websocket.close(
                    code=1011,
                    reason=websocket_close_reason(redacted_error, fallback="Internal server error"),
                )
            except RuntimeError as close_error:
                if "already completed" not in str(close_error) and "websocket.close" not in str(close_error):
                    raise Exception(f"Unexpected error while closing WebSocket: {close_error}") from close_error

    async def _bidirectional_forward(
        self,
        *,
        websocket: LiveClientWebSocket,
        backend_websocket: LiveBackendWebSocket,
        event_retention: _LiveEventRetention,
    ) -> None:
        from websockets.exceptions import ConnectionClosed

        client_gone: Final = asyncio.Event()
        client_close_sent: Final = asyncio.Event()
        client_forward_task: Final = asyncio.create_task(
            _forward_live_client_frames(
                websocket=websocket,
                backend_websocket=backend_websocket,
                client_gone=client_gone,
                client_close_sent=client_close_sent,
            )
        )
        backend_forward_task: Final = asyncio.create_task(
            _forward_live_backend_frames(
                websocket=websocket,
                backend_websocket=backend_websocket,
                client_gone=client_gone,
                event_retention=event_retention,
            )
        )
        client_gone_task: Final = asyncio.create_task(client_gone.wait())
        forwarding_tasks: Final = (client_forward_task, backend_forward_task, client_gone_task)
        try:
            completed_tasks: Final = await asyncio.wait(
                forwarding_tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            done_tasks: Final = completed_tasks[0]
            if backend_forward_task in done_tasks:
                backend_forward_task.result()
                return
            if client_forward_task in done_tasks:
                client_forward_task.result()
            elif client_gone_task in done_tasks:
                client_forward_task.cancel()
            event_loop: Final = asyncio.get_running_loop()
            close_deadline: Final = event_loop.time() + OPENAI_LIVE_SESSION_CLOSE_TIMEOUT_SECONDS
            if not client_close_sent.is_set():
                remaining_send_timeout: Final = max(close_deadline - event_loop.time(), 0)
                with suppress(ConnectionClosed, TimeoutError, asyncio.TimeoutError):
                    await asyncio.wait_for(
                        backend_websocket.send(_LIVE_SESSION_CLOSE_FRAME),
                        timeout=remaining_send_timeout,
                    )
            remaining_wait_timeout: Final = max(close_deadline - event_loop.time(), 0)
            close_wait_result: Final = await asyncio.wait(
                (backend_forward_task,),
                timeout=remaining_wait_timeout,
            )
            remaining_close_timeout: Final = max(close_deadline - event_loop.time(), 0)
            with suppress(TimeoutError, asyncio.TimeoutError):
                await asyncio.wait_for(
                    backend_websocket.close(),
                    timeout=remaining_close_timeout,
                )
            if backend_forward_task in close_wait_result[0]:
                backend_forward_task.result()
            elif not backend_forward_task.done():
                backend_forward_task.cancel()
        finally:
            for task in forwarding_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*forwarding_tasks, return_exceptions=True)
