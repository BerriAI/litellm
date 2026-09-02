from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import json
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from typing import Final, Protocol
from urllib.parse import urlparse, urlunparse

from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.constants import REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from litellm.litellm_core_utils.realtime_streaming import RealTimeStreaming
from litellm.llms.custom_httpx.http_handler import get_shared_realtime_ssl_context
from litellm.types.realtime import RealtimeInputAudioTranscriptionUsage, RealtimeQueryParams

from .transformation import (
    MUSE_MODEL,
    MuseEventTransformer,
    MuseProtocolError,
    MuseSessionConfig,
    encode_event,
    error_event,
    parse_session_update,
    session_created_event,
    session_updated_event,
)

DEFAULT_MUSE_REALTIME_URL: Final = "wss://api.meta.ai/v1/asr/realtime"
_MAX_AUDIO_BACKLOG_SECONDS: Final = 4
_MAX_PENDING_PROVIDER_EVENTS: Final = 256
_JSON_ADAPTER: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)


class _ProviderWebSocket(Protocol):
    async def send(self, message: str | bytes) -> None: ...

    async def recv(self, decode: bool | None = None) -> str | bytes: ...

    async def close(self, code: int = 1000, reason: str = "") -> None: ...


class _ClientWebSocketExceptions(Protocol):
    ConnectionClosed: type[Exception]


class _ClientWebSocket(Protocol):
    exceptions: _ClientWebSocketExceptions

    @property
    def scope(self) -> Mapping[str, object]: ...

    async def send_text(self, data: str) -> None: ...

    async def receive_text(self) -> str: ...

    async def close(self, code: int = 1000, reason: str | None = None) -> None: ...


class WebSocketConnect(Protocol):
    def __call__(
        self,
        url: str,
        *,
        open_timeout: float,
        max_size: int | None,
        ssl: object | None,
    ) -> Awaitable[_ProviderWebSocket]: ...


class MuseAdapterError(RuntimeError):
    def __init__(self, message: str, *, close_code: int) -> None:
        super().__init__(message)
        self.close_code: Final = close_code


class MuseRealtimeAdapter:
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        api_base: str | None = None,
        timeout: float | None = None,
        websocket_connect: WebSocketConnect | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        terminate_client: Callable[[int], Awaitable[None]] | None = None,
    ) -> None:
        if model.removeprefix("meta/") != MUSE_MODEL:
            raise ValueError("unsupported Meta realtime model")
        self._model: Final = model.removeprefix("meta/")
        self._access_token: Final = normalize_access_token(api_key)
        self._url: Final = build_muse_realtime_url(api_base)
        self._timeout: Final = timeout or 10.0
        self._websocket_connect = websocket_connect
        self._monotonic: Final = monotonic
        self._sleep: Final = sleep
        self._terminate_client: Final = terminate_client
        self._provider_ws: _ProviderWebSocket | None = None
        self._config: MuseSessionConfig | None = None
        self._session_id: str = f"sess_{uuid.uuid4().hex}"
        self._events: Final[asyncio.Queue[str | BaseException]] = asyncio.Queue(maxsize=_MAX_PENDING_PROVIDER_EVENTS)
        self._events.put_nowait(encode_event(session_created_event(self._model, self._session_id)))
        self._transformer: Final = MuseEventTransformer()
        self._audio_condition: Final = asyncio.Condition()
        self._pending_audio: bytearray = bytearray()
        self._audio_generation: int = 0
        self._flush_requested: bool = False
        self._end_requested: bool = False
        self._end_stream_sent: bool = False
        self._audio_consumed: bool = False
        self._closed: bool = False
        self._resources_closed: bool = False
        self._sender_task: asyncio.Task[None] | None = None
        self._receiver_task: asyncio.Task[None] | None = None
        self.close_code: int = 1000
        self.close_reason: str = "Session closed"

    async def send(self, message: str | bytes) -> None:
        if self._closed:
            raise MuseAdapterError("Meta Muse realtime session is closed", close_code=self.close_code)
        if isinstance(message, bytes):
            await self._reject("invalid_request_error", "invalid_event", "Client events must be JSON text")
            return
        try:
            event: Final = _parse_client_event(message)
            event_type: Final = event.get("type")
            if event_type in ("session.update", "transcription_session.update"):
                await self._handle_session_update(message)
                return
            if event_type == "input_audio_buffer.append":
                await self._handle_audio_append(event)
                return
            if event_type == "input_audio_buffer.clear":
                await self._clear_audio()
                return
            if event_type == "input_audio_buffer.commit":
                await self._commit_audio()
                return
            if event_type == "input_audio_buffer.end":
                await self._end_audio()
                return
            await self._emit(
                error_event(
                    "invalid_request_error",
                    "unsupported_event",
                    f"Event type {event_type!r} is not supported for Meta Muse transcription",
                )
            )
        except MuseProtocolError as exc:
            await self._reject("invalid_request_error", "invalid_event", str(exc))

    async def recv(self, decode: bool | None = None) -> str | bytes:
        event: Final = await self._events.get()
        if isinstance(event, BaseException):
            close_code: Final = _exception_close_code(event) if isinstance(event, Exception) else 1011
            if self._terminate_client is not None:
                await self._terminate_client(close_code)
            raise event
        return event.encode("utf-8") if decode is False else event

    async def close(self, code: int = 1000, reason: str = "") -> None:
        if self._resources_closed:
            return
        self._closed = True
        self._resources_closed = True
        self.close_code = sanitize_close_code(code)
        self.close_reason = safe_close_reason(self.close_code)
        async with self._audio_condition:
            self._end_requested = True
            self._audio_condition.notify_all()
        tasks: Final = tuple(task for task in (self._sender_task, self._receiver_task) if task is not None)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        provider_ws: Final = self._provider_ws
        if provider_ws is not None:
            with contextlib.suppress(Exception):
                await provider_ws.close(code=self.close_code, reason=self.close_reason)

    def unbilled_usage_on_session_close(self, model: str) -> RealtimeInputAudioTranscriptionUsage | None:
        return self._transformer.take_unbilled_usage()

    async def _handle_session_update(self, message: str) -> None:
        config: Final = parse_session_update(message, self._model)
        if self._config is not None:
            if config != self._config:
                await self._reject(
                    "invalid_request_error",
                    "session_configuration_locked",
                    "Meta Muse session configuration cannot change after setup",
                )
                return
            await self._emit(session_updated_event(config, self._session_id))
            return
        await self._connect(config)

    async def _connect(self, config: MuseSessionConfig) -> None:
        connector: Final = self._websocket_connect or _default_websocket_connect
        last_error: Exception | None = None  # rebind-ok: records the latest bounded handshake attempt
        for attempt in range(2):
            provider_ws: _ProviderWebSocket | None = None
            try:
                provider_ws = await connector(
                    self._url,
                    open_timeout=self._timeout,
                    max_size=REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES,
                    ssl=_ssl_config(self._url),
                )
                await provider_ws.send(json.dumps(config.handshake(self._access_token), separators=(",", ":")))
                raw_ack: str | bytes = await asyncio.wait_for(  # rebind-ok: one response per handshake attempt
                    provider_ws.recv(), timeout=self._timeout
                )
                session_id: str = _parse_handshake_ack(raw_ack)  # rebind-ok: one ID per handshake attempt
                self._provider_ws = provider_ws
                self._config = config
                self._transformer.configure(config)
                self._session_id = session_id
                self._sender_task = asyncio.create_task(self._send_audio(), name="meta-muse-realtime-send")
                self._receiver_task = asyncio.create_task(self._receive_events(), name="meta-muse-realtime-receive")
                await self._emit(session_updated_event(config, session_id))
                return
            except asyncio.CancelledError:
                if provider_ws is not None:
                    with contextlib.suppress(Exception):
                        await provider_ws.close()
                raise
            except Exception as exc:  # noqa: BLE001  # connector implementations expose heterogeneous transport errors
                last_error = exc
                if provider_ws is not None:
                    with contextlib.suppress(Exception):
                        await provider_ws.close()
                close_code: int = _exception_close_code(exc)  # rebind-ok: classified per handshake attempt
                retryable_transport_error: bool = not isinstance(  # rebind-ok: classified per handshake attempt
                    exc, (MuseAdapterError, MuseProtocolError)
                )
                if attempt == 0 and retryable_transport_error and close_code in (1011, 1013):
                    continue
                self.close_code = close_code
                self.close_reason = safe_close_reason(close_code)
                await self._emit(
                    error_event(
                        "server_error" if close_code != 1008 else "invalid_request_error",
                        "provider_connection_error",
                        "Meta Muse realtime handshake failed",
                    )
                )
                await self._events.put(MuseAdapterError("Meta Muse realtime handshake failed", close_code=close_code))
                await self._mark_terminated(close_code)
                return
        assert last_error is not None
        raise MuseAdapterError("Meta Muse realtime handshake failed", close_code=1011)

    async def _handle_audio_append(self, event: Mapping[str, JsonValue]) -> None:
        config: Final = self._require_configured()
        if self._end_requested or self._end_stream_sent:
            await self._reject("invalid_request_error", "input_ended", "Audio input has already ended")
            return
        audio_value: Final = event.get("audio")
        if not isinstance(audio_value, str):
            await self._reject("invalid_request_error", "invalid_audio", "Audio must be a base64 string")
            return
        try:
            audio: Final = base64.b64decode(audio_value, validate=True)
        except (binascii.Error, ValueError):
            await self._reject("invalid_request_error", "invalid_audio", "Audio must be valid base64")
            return
        if len(audio) % 2:
            await self._reject("invalid_request_error", "invalid_audio", "PCM16 audio must contain complete samples")
            return
        if not audio:
            return
        max_backlog_bytes: Final = config.bytes_per_second * _MAX_AUDIO_BACKLOG_SECONDS
        if len(audio) > max_backlog_bytes:
            await self._reject(
                "invalid_request_error",
                "audio_backlog_exceeded",
                "Audio append exceeds the four-second Muse backlog limit",
            )
            return
        async with self._audio_condition:
            await self._audio_condition.wait_for(
                lambda: self._closed or len(self._pending_audio) + len(audio) <= max_backlog_bytes
            )
            if self._closed:
                raise MuseAdapterError("Meta Muse realtime session is closed", close_code=self.close_code)
            self._pending_audio.extend(audio)
            self._audio_condition.notify_all()

    async def _clear_audio(self) -> None:
        self._require_configured()
        async with self._audio_condition:
            self._pending_audio.clear()
            self._audio_generation += 1
            self._flush_requested = False
            self._audio_condition.notify_all()
        await self._emit(
            {  # mutable-ok: OpenAI-compatible JSON event
                "type": "input_audio_buffer.cleared",
                "event_id": f"event_{uuid.uuid4().hex}",
            }
        )

    async def _commit_audio(self) -> None:
        config: Final = self._require_configured()
        previous_item_id, item_id = self._transformer.commit_item()
        async with self._audio_condition:
            self._flush_requested = True
            if config.mode == "PUSH_TO_TALK":
                self._end_requested = True
            self._audio_condition.notify_all()
        await self._emit(
            {  # mutable-ok: OpenAI-compatible JSON event
                "type": "input_audio_buffer.committed",
                "event_id": f"event_{uuid.uuid4().hex}",
                "previous_item_id": previous_item_id,
                "item_id": item_id,
            }
        )

    async def _end_audio(self) -> None:
        self._require_configured()
        async with self._audio_condition:
            self._flush_requested = True
            self._end_requested = True
            self._audio_condition.notify_all()

    async def _send_audio(self) -> None:
        config: Final = self._require_configured()
        provider_ws: Final = self._require_provider_ws()
        pacing_origin: float | None = None  # rebind-ok: initialized when the first packet is ready
        sent_duration: float = 0.0  # rebind-ok: absolute pacing clock advances after each packet
        try:
            while True:
                packet, pacing_origin, ended = await self._next_audio_packet(
                    config,
                    pacing_origin,
                    sent_duration,
                )
                if ended:
                    break
                if packet is None:
                    continue
                await provider_ws.send(packet)
                self._audio_consumed = True
                sent_duration += len(packet) / config.bytes_per_second
            await self._send_end_stream()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001  # WebSocket implementations expose heterogeneous transport errors
            await self._fail_provider(exc, phase="audio send")

    async def _next_audio_packet(
        self,
        config: MuseSessionConfig,
        pacing_origin: float | None,
        sent_duration: float,
    ) -> tuple[bytes | None, float | None, bool]:
        async with self._audio_condition:
            await self._audio_condition.wait_for(
                lambda: (
                    self._closed
                    or len(self._pending_audio) >= config.packet_bytes
                    or (self._flush_requested and bool(self._pending_audio))
                    or (self._end_requested and not self._pending_audio)
                )
            )
            if self._closed or (self._end_requested and not self._pending_audio):
                return None, pacing_origin, True
            packet_size: Final = min(config.packet_bytes, len(self._pending_audio))
            if packet_size < config.packet_bytes and not self._flush_requested:
                return None, pacing_origin, False
            generation: Final = self._audio_generation
            current_time: Final = self._monotonic()
            effective_origin: Final = (
                current_time - sent_duration
                if pacing_origin is None or current_time > pacing_origin + sent_duration
                else pacing_origin
            )
            deadline: Final = effective_origin + sent_duration
        delay: Final = deadline - self._monotonic()
        if delay > 0:
            await self._sleep(delay)
        async with self._audio_condition:
            if generation != self._audio_generation:
                return None, effective_origin, False
            actual_size: Final = min(packet_size, len(self._pending_audio))
            packet: Final = bytes(self._pending_audio[:actual_size])
            del self._pending_audio[:actual_size]
            if not self._pending_audio:
                self._flush_requested = False
            self._audio_condition.notify_all()
        return packet or None, effective_origin, False

    async def _send_end_stream(self) -> None:
        if self._end_stream_sent:
            return
        provider_ws: Final = self._require_provider_ws()
        await provider_ws.send('{"type":"endStream"}')
        self._end_stream_sent = True

    async def _receive_events(self) -> None:
        provider_ws: Final = self._require_provider_ws()
        try:
            while True:
                raw: str | bytes = await provider_ws.recv()  # rebind-ok: one provider frame per iteration
                if not isinstance(raw, str):
                    raise MuseProtocolError("provider returned a non-text event")
                for event in self._transformer.transform(raw):
                    await self._emit(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001  # provider close exceptions vary by WebSocket implementation
            close_code: Final = _exception_close_code(exc)
            if close_code == 1000 and self._end_stream_sent:
                await self._mark_terminated(1000)
                await self._events.put(MuseAdapterError("Meta Muse realtime session completed", close_code=1000))
                return
            failure: Final = MuseAdapterError(
                "Meta Muse realtime closed before input ended",
                close_code=1011 if close_code == 1000 else close_code,
            )
            await self._fail_provider(failure, phase="receive")

    async def _fail_provider(self, exc: Exception, *, phase: str) -> None:
        close_code: Final = _exception_close_code(exc)
        self.close_code = close_code
        self.close_reason = safe_close_reason(close_code)
        await self._emit(
            error_event(
                "server_error",
                "provider_connection_error",
                f"Meta Muse realtime {phase} failed",
            )
        )
        await self._events.put(MuseAdapterError(f"Meta Muse realtime {phase} failed", close_code=close_code))
        await self._mark_terminated(close_code)

    async def _mark_terminated(self, close_code: int) -> None:
        self._closed = True
        self.close_code = sanitize_close_code(close_code)
        self.close_reason = safe_close_reason(self.close_code)
        async with self._audio_condition:
            self._audio_condition.notify_all()

    async def _terminate(self, close_code: int) -> None:
        await self._mark_terminated(close_code)
        if self._terminate_client is not None:
            await self._terminate_client(self.close_code)

    async def _reject(self, error_type: str, code: str, message: str) -> None:
        self.close_code = 1008
        self.close_reason = safe_close_reason(1008)
        await self._emit(error_event(error_type, code, message))
        await self._events.put(MuseAdapterError(message, close_code=1008))
        await self._mark_terminated(1008)

    async def _emit(self, event: Mapping[str, object]) -> None:
        await self._events.put(encode_event(event))

    def _require_configured(self) -> MuseSessionConfig:
        if self._config is None:
            raise MuseProtocolError("send session.update before audio events")
        return self._config

    def _require_provider_ws(self) -> _ProviderWebSocket:
        if self._provider_ws is None:
            raise MuseProtocolError("Meta Muse provider connection is not ready")
        return self._provider_ws


class MetaRealtime:
    async def async_realtime(
        self,
        model: str,
        websocket: _ClientWebSocket,
        logging_obj: LiteLLMLogging,
        api_base: str | None = None,
        api_key: str | None = None,
        client: object | None = None,
        timeout: float | None = None,
        query_params: RealtimeQueryParams | None = None,
        user_api_key_dict: object | None = None,
        litellm_metadata: Mapping[str, object] | None = None,
        websocket_connect: WebSocketConnect | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        **kwargs: object,  # kwargs-ok: realtime dispatcher forwards provider-neutral options
    ) -> None:
        if api_key is None or not api_key.strip():
            await _send_client_error_and_close(websocket, "Meta Model API key is required")
            return
        try:
            adapter: Final = MuseRealtimeAdapter(
                model=model,
                api_key=api_key,
                api_base=api_base,
                timeout=timeout,
                websocket_connect=websocket_connect,
                monotonic=monotonic,
                sleep=sleep,
                terminate_client=lambda code: _close_client(websocket, code),
            )
        except ValueError:
            await _send_client_error_and_close(websocket, "Invalid Meta Muse realtime configuration")
            return
        realtime_streaming: Final = RealTimeStreaming(
            websocket,
            adapter,  # pyright: ignore[reportArgumentType]  # raw adapter intentionally matches the websocket surface
            logging_obj,
            model=model,
            user_api_key_dict=user_api_key_dict,
            request_data={  # mutable-ok: relay request metadata payload
                "litellm_metadata": dict(litellm_metadata or {})  # mutable-ok: relay owns its metadata copy
            },
            force_transcription_model=model,
            usage_provider=adapter,
            exclude_private_content_from_logs=True,
        )
        try:
            await realtime_streaming.bidirectional_forward()
        except MuseAdapterError as exc:
            adapter.close_code = exc.close_code
            adapter.close_reason = safe_close_reason(exc.close_code)
        except Exception:  # noqa: BLE001  # relay errors are normalized before closing the accepted client socket
            adapter.close_code = 1011
            adapter.close_reason = safe_close_reason(1011)
            verbose_proxy_logger.exception("Meta Muse realtime session failed")
        finally:
            await adapter.close(code=adapter.close_code)
            await _close_client(websocket, adapter.close_code)


def normalize_access_token(api_key: str) -> str:
    stripped: Final = api_key.strip()
    if not stripped:
        raise ValueError("Meta Model API key is required")
    parts: Final = stripped.split(None, 1)
    if parts[0].casefold() == "bearer":
        if len(parts) != 2 or not parts[1].strip():
            raise ValueError("Meta Model API key must include a token after Bearer")
        return f"Bearer {parts[1].strip()}"
    return f"Bearer {stripped}"


def build_muse_realtime_url(api_base: str | None) -> str:
    if api_base is None:
        return DEFAULT_MUSE_REALTIME_URL
    parsed: Final = urlparse(api_base.strip())
    scheme: Final = "wss" if parsed.scheme == "https" else parsed.scheme
    if (
        scheme != "wss"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ValueError("Meta api_base must be an absolute wss:// or https:// URL without credentials or a fragment")
    netloc: Final = f"{parsed.hostname}:{parsed.port}" if parsed.port is not None else parsed.hostname
    return urlunparse((scheme, netloc, "/v1/asr/realtime", "", "", ""))


def sanitize_close_code(code: int | None) -> int:
    if code is not None and code in (1000, 1008, 1011, 1013):
        return code
    return 1011


def safe_close_reason(code: int) -> str:
    return {  # mutable-ok: immutable-by-convention close-reason lookup
        1000: "Session closed",
        1008: "Invalid realtime transcription request",
        1011: "Realtime transcription service error",
        1013: "Realtime transcription service unavailable",
    }.get(code, "Realtime transcription service error")


def _parse_client_event(payload: str) -> Mapping[str, JsonValue]:
    try:
        value: Final = _JSON_ADAPTER.validate_json(payload)
    except ValidationError:
        raise MuseProtocolError("invalid JSON object") from None
    if not isinstance(value, dict):
        raise MuseProtocolError("message must be a JSON object")
    event_type: Final = value.get("type")
    if not isinstance(event_type, str) or not event_type:
        raise MuseProtocolError("message type must be a non-empty string")
    return value


def _parse_handshake_ack(raw: str | bytes) -> str:
    if not isinstance(raw, str):
        raise MuseProtocolError("provider returned a non-text handshake response")
    message: Final = _parse_json_object(raw)
    if message.get("type") == "error":
        raise MuseAdapterError("Meta Muse realtime handshake was rejected", close_code=1008)
    session_id: Final = message.get("sessionId")
    if not isinstance(session_id, str) or not session_id.strip():
        raise MuseProtocolError("provider returned an invalid handshake response")
    return session_id.strip()


def _parse_json_object(payload: str) -> Mapping[str, JsonValue]:
    try:
        value: Final = _JSON_ADAPTER.validate_json(payload)
    except ValidationError:
        raise MuseProtocolError("invalid provider JSON object") from None
    if not isinstance(value, dict):
        raise MuseProtocolError("provider message must be a JSON object")
    return value


def _exception_close_code(exc: Exception) -> int:
    code: Final = getattr(exc, "code", None)
    if isinstance(exc, MuseAdapterError):
        return sanitize_close_code(exc.close_code)
    return sanitize_close_code(code if isinstance(code, int) else None)


def _ssl_config(url: str) -> object | None:
    if not url.startswith("wss://"):
        return None
    config: Final = get_shared_realtime_ssl_context()
    return True if config is False else config


async def _default_websocket_connect(
    url: str,
    *,
    open_timeout: float,
    max_size: int | None,
    ssl: object | None,
) -> _ProviderWebSocket:
    import websockets

    connection: Final = await websockets.connect(
        url,
        open_timeout=open_timeout,
        max_size=max_size,
        ssl=ssl,  # pyright: ignore[reportArgumentType]  # shared SSL helper returns the library-supported union
    )
    return connection


async def _send_client_error_and_close(websocket: _ClientWebSocket, message: str) -> None:
    with contextlib.suppress(Exception):
        await websocket.send_text(encode_event(error_event("invalid_request_error", "invalid_configuration", message)))
    await _close_client(websocket, 1008)


async def _close_client(websocket: _ClientWebSocket, code: int) -> None:
    with contextlib.suppress(Exception):
        await websocket.close(code=sanitize_close_code(code), reason=safe_close_reason(sanitize_close_code(code)))
