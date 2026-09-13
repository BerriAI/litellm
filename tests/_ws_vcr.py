"""Record and replay realtime WebSocket traffic in the shared VCR Redis store.

The HTTP VCR layer (``tests/_vcr_redis_persister.py`` /
``tests/_vcr_conftest_common.py``) only intercepts httpx/aiohttp, so the
realtime suite always reached the live provider. This module intercepts at the
``websockets.connect`` boundary instead and caches whole WebSocket sessions
under a distinct ``litellm:vcr:wscassette:`` key, reusing the same Redis client,
24h TTL, save-on-pass, and best-effort degradation semantics.

Record mode logs every frame in order with its direction, a text/binary flag,
and, for each server frame, the number of client frames seen before it. That
count is the causal gate for replay: a recorded server frame is only released
once the client has sent at least that many frames, so the deterministic replay
reproduces the same interleaving without a live connection. Client frames are
matched against the recording with volatile fields (ids, timestamps) normalized
away; a structurally different client frame is contract drift and raises loudly
rather than hanging, and every replay wait is bounded by a timeout.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import warnings
from typing import AsyncIterator, Callable, Literal, Optional, Protocol, Union

from pydantic import BaseModel, ConfigDict, ValidationError
from websockets.exceptions import ConnectionClosedOK

from tests._vcr_redis_persister import (
    CASSETTE_TTL_SECONDS,
    VCRCassetteCacheWarning,
    _build_default_client,
    _record_cache_failure,
)

WS_REDIS_KEY_PREFIX = "litellm:vcr:wscassette:"
WS_MAX_SESSIONS_PER_CASSETTE = 20
WS_MAX_FRAMES_PER_SESSION = 2000
WS_REPLAY_TIMEOUT_ENV = "LITELLM_WS_VCR_REPLAY_TIMEOUT"
WS_DEFAULT_REPLAY_TIMEOUT_SECONDS = 15.0
WS_CASSETTE_SCHEMA_VERSION = 1

_log = logging.getLogger(__name__)

Message = Union[str, bytes]
Direction = Literal["client_to_server", "server_to_client"]
Opcode = Literal["text", "binary"]


class WsConnectionLike(Protocol):
    async def recv(self, decode: Optional[bool] = None) -> Message: ...

    async def send(self, message: Message, *args: object, **kwargs: object) -> None: ...

    async def close(self, *args: object, **kwargs: object) -> None: ...

    def __aiter__(self) -> AsyncIterator[Message]: ...


class WsConnectContextLike(Protocol):
    async def __aenter__(self) -> WsConnectionLike: ...

    async def __aexit__(self, *exc_info: object) -> Optional[bool]: ...


class RedisLike(Protocol):
    def get(self, key: str) -> Optional[bytes]: ...

    def set(self, key: str, value: bytes, ex: int) -> object: ...


class WsFrame(BaseModel):
    model_config = ConfigDict(frozen=True)

    direction: Direction
    opcode: Opcode
    text: Optional[str] = None
    binary_b64: Optional[str] = None
    client_frames_before: Optional[int] = None


class WsSession(BaseModel):
    model_config = ConfigDict(frozen=True)

    frames: tuple[WsFrame, ...]


class WsCassette(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int = WS_CASSETTE_SCHEMA_VERSION
    sessions: tuple[WsSession, ...]


class WsVcrReplayError(Exception): ...


class WsVcrContractDrift(WsVcrReplayError): ...


class WsVcrReplayTimeout(WsVcrReplayError): ...


_UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_OPENAI_ID_RE = re.compile(
    r"\b(?:evt|event|item|msg|resp|response|sess|session|call|fc|rs|conv|ce|audio)_[A-Za-z0-9]{6,}"
)
_ISO_TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?")
_EPOCH_RE = re.compile(r"(?<![\d.])1[0-9]{9,12}(?![\d.])")

_VOLATILE_KEYS = frozenset(
    {
        "event_id",
        "item_id",
        "previous_item_id",
        "response_id",
        "id",
        "session_id",
        "call_id",
        "conversation_id",
    }
)

_BEARER_RE = re.compile(r"Bearer\s+[A-Za-z0-9._\-]+")
_OPENAI_KEY_RE = re.compile(r"sk-[A-Za-z0-9_\-]{8,}")
_XAI_KEY_RE = re.compile(r"xai-[A-Za-z0-9_\-]{8,}")


def scrub_secrets(text: str) -> str:
    scrubbed = _BEARER_RE.sub("Bearer <redacted>", text)
    scrubbed = _OPENAI_KEY_RE.sub("<redacted-key>", scrubbed)
    scrubbed = _XAI_KEY_RE.sub("<redacted-key>", scrubbed)
    return scrubbed


def _normalize_json_for_match(obj: object) -> object:
    if isinstance(obj, dict):
        return {
            str(key): ("<vcr-id>" if key in _VOLATILE_KEYS else _normalize_json_for_match(value))
            for key, value in sorted(obj.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(obj, list):
        return [_normalize_json_for_match(item) for item in obj]
    if isinstance(obj, str):
        return _normalize_scalar_string(obj)
    return obj


def _normalize_scalar_string(text: str) -> str:
    normalized = _UUID_RE.sub("<vcr-uuid>", text)
    normalized = _OPENAI_ID_RE.sub("<vcr-id>", normalized)
    normalized = _ISO_TS_RE.sub("<vcr-iso-ts>", normalized)
    normalized = _EPOCH_RE.sub("<vcr-epoch>", normalized)
    return normalized


def normalize_text_for_match(text: str) -> str:
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return _normalize_scalar_string(text)
    return json.dumps(_normalize_json_for_match(parsed), sort_keys=True, separators=(",", ":"))


def text_frames_match(recorded: str, incoming: str) -> bool:
    return normalize_text_for_match(recorded) == normalize_text_for_match(incoming)


def _frame_payload(message: Message) -> tuple[Opcode, Optional[str], Optional[str]]:
    if isinstance(message, str):
        return "text", scrub_secrets(message), None
    try:
        decoded = message.decode("utf-8")
    except UnicodeDecodeError:
        return "binary", None, base64.b64encode(message).decode("ascii")
    return "text", scrub_secrets(decoded), None


def ws_redis_key_for(nodeid: str) -> str:
    rel = nodeid.replace("::", "/").replace("\\", "/").lstrip("./")
    return f"{WS_REDIS_KEY_PREFIX}{rel}"


def replay_timeout_seconds() -> float:
    raw = os.environ.get(WS_REPLAY_TIMEOUT_ENV)
    if not raw:
        return WS_DEFAULT_REPLAY_TIMEOUT_SECONDS
    try:
        return float(raw)
    except ValueError:
        return WS_DEFAULT_REPLAY_TIMEOUT_SECONDS


class WsSessionRecorder:
    def __init__(self) -> None:
        self._frames: list[WsFrame] = []
        self._client_count = 0

    def record_client_frame(self, message: Message) -> None:
        opcode, text, binary_b64 = _frame_payload(message)
        self._frames.append(WsFrame(direction="client_to_server", opcode=opcode, text=text, binary_b64=binary_b64))
        self._client_count += 1

    def record_server_frame(self, message: Message) -> None:
        opcode, text, binary_b64 = _frame_payload(message)
        self._frames.append(
            WsFrame(
                direction="server_to_client",
                opcode=opcode,
                text=text,
                binary_b64=binary_b64,
                client_frames_before=self._client_count,
            )
        )

    def to_session(self) -> WsSession:
        return WsSession(frames=tuple(self._frames))


class RecordingConnection:
    def __init__(self, real: WsConnectionLike, recorder: WsSessionRecorder) -> None:
        self._real = real
        self._recorder = recorder

    async def recv(self, decode: Optional[bool] = None) -> Message:
        result = await self._real.recv(decode=decode)
        self._recorder.record_server_frame(result)
        return result

    async def send(self, message: Message, *args: object, **kwargs: object) -> None:
        self._recorder.record_client_frame(message)
        await self._real.send(message, *args, **kwargs)

    async def close(self, *args: object, **kwargs: object) -> None:
        await self._real.close(*args, **kwargs)

    def __aiter__(self) -> AsyncIterator[Message]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[Message]:
        async for message in self._real:
            self._recorder.record_server_frame(message)
            yield message


class ReplayConnection:
    def __init__(
        self,
        session: WsSession,
        timeout: float,
        on_error: Callable[[WsVcrReplayError], None],
    ) -> None:
        self._server_frames = tuple(f for f in session.frames if f.direction == "server_to_client")
        self._client_frames = tuple(f for f in session.frames if f.direction == "client_to_server")
        self._timeout = timeout
        self._on_error = on_error
        self._server_cursor = 0
        self._client_cursor = 0
        self._client_sent = 0
        self._closed = False
        self._progress = asyncio.Event()

    async def recv(self, decode: Optional[bool] = None) -> Message:
        want_bytes = decode is False
        while True:
            if self._closed or self._server_cursor >= len(self._server_frames):
                raise ConnectionClosedOK(None, None)
            frame = self._server_frames[self._server_cursor]
            needed = frame.client_frames_before or 0
            if self._client_sent >= needed:
                self._server_cursor += 1
                return _materialize_frame(frame, want_bytes)
            await self._await_client_progress(needed)

    async def _await_client_progress(self, needed: int) -> None:
        waiter = self._progress
        try:
            await asyncio.wait_for(waiter.wait(), timeout=self._timeout)
        except asyncio.TimeoutError:
            error = WsVcrReplayTimeout(
                f"WS-VCR replay stalled: server frame #{self._server_cursor} needs "
                f"{needed} client frame(s) but only {self._client_sent} were sent within "
                f"{self._timeout}s. The client stopped driving the recorded session."
            )
            self._on_error(error)
            raise error

    async def send(self, message: Message, *args: object, **kwargs: object) -> None:
        if self._client_cursor >= len(self._client_frames):
            error = WsVcrContractDrift(
                "WS-VCR contract drift: client sent frame "
                f"#{self._client_cursor + 1} but the recording has only "
                f"{len(self._client_frames)} client frame(s). Extra frame: {_preview(message)}"
            )
            self._on_error(error)
            raise error
        recorded = self._client_frames[self._client_cursor]
        if not _client_frame_matches(recorded, message):
            error = WsVcrContractDrift(
                "WS-VCR contract drift on client frame "
                f"#{self._client_cursor + 1}:\n  recorded: {_preview_frame(recorded)}\n"
                f"  got:      {_preview(message)}"
            )
            self._on_error(error)
            raise error
        self._client_cursor += 1
        self._client_sent += 1
        self._signal_progress()

    async def close(self, *args: object, **kwargs: object) -> None:
        self._closed = True
        self._signal_progress()

    def _signal_progress(self) -> None:
        previous = self._progress
        self._progress = asyncio.Event()
        previous.set()

    def __aiter__(self) -> AsyncIterator[Message]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[Message]:
        while True:
            try:
                yield await self.recv()
            except ConnectionClosedOK:
                return


def _materialize_frame(frame: WsFrame, want_bytes: bool) -> Message:
    if frame.opcode == "text":
        text = frame.text or ""
        return text.encode("utf-8") if want_bytes else text
    return base64.b64decode(frame.binary_b64 or "")


def _client_frame_matches(recorded: WsFrame, message: Message) -> bool:
    opcode, text, binary_b64 = _frame_payload(message)
    if recorded.opcode != opcode:
        return False
    if opcode == "text":
        return text_frames_match(recorded.text or "", text or "")
    return recorded.binary_b64 == binary_b64


def _preview(message: Message) -> str:
    text = message if isinstance(message, str) else message.decode("utf-8", errors="replace")
    return scrub_secrets(text)[:200]


def _preview_frame(frame: WsFrame) -> str:
    if frame.opcode == "text":
        return (frame.text or "")[:200]
    return f"<binary {len(frame.binary_b64 or '')} b64 chars>"


class _RecordingConnect:
    def __init__(
        self,
        real_context: WsConnectContextLike,
        recorder: WsSessionRecorder,
        on_done: Callable[[WsSessionRecorder], None],
    ) -> None:
        self._real_context = real_context
        self._recorder = recorder
        self._on_done = on_done

    async def __aenter__(self) -> RecordingConnection:
        real = await self._real_context.__aenter__()
        return RecordingConnection(real, self._recorder)

    async def __aexit__(self, *exc_info: object) -> Optional[bool]:
        try:
            return await self._real_context.__aexit__(*exc_info)
        finally:
            self._on_done(self._recorder)


class _ReplayConnect:
    def __init__(
        self,
        session: WsSession,
        timeout: float,
        on_error: Callable[[WsVcrReplayError], None],
    ) -> None:
        self._session = session
        self._timeout = timeout
        self._on_error = on_error

    async def __aenter__(self) -> ReplayConnection:
        return ReplayConnection(self._session, self._timeout, self._on_error)

    async def __aexit__(self, *exc_info: object) -> bool:
        return False


class WsVcrController:
    def __init__(
        self,
        original_connect: Callable[..., WsConnectContextLike],
        cassette: Optional[WsCassette],
        timeout: float,
    ) -> None:
        self._original_connect = original_connect
        self._cassette = cassette
        self._timeout = timeout
        self._replay_cursor = 0
        self._recorded_sessions: list[WsSession] = []
        self._errors: list[WsVcrReplayError] = []
        self._replayed = False
        self._recorded = False

    def connect(self, *args: object, **kwargs: object) -> object:
        if self._cassette is not None and self._replay_cursor < len(self._cassette.sessions):
            session = self._cassette.sessions[self._replay_cursor]
            self._replay_cursor += 1
            self._replayed = True
            return _ReplayConnect(session, self._timeout, self._errors.append)
        self._recorded = True
        recorder = WsSessionRecorder()
        return _RecordingConnect(self._original_connect(*args, **kwargs), recorder, self._finish_recorder)

    def _finish_recorder(self, recorder: WsSessionRecorder) -> None:
        self._recorded_sessions.append(recorder.to_session())

    @property
    def errors(self) -> tuple[WsVcrReplayError, ...]:
        return tuple(self._errors)

    @property
    def replayed(self) -> bool:
        return self._replayed

    @property
    def recorded(self) -> bool:
        return self._recorded

    def built_cassette(self) -> Optional[WsCassette]:
        if not self._recorded_sessions:
            return None
        return WsCassette(sessions=tuple(self._recorded_sessions))

    def verdict(self) -> str:
        if self._replayed and not self._recorded:
            return f"[WS-VCR HIT] sessions={self._replay_cursor} frames={self._played_frame_count()}"
        if self._recorded:
            cassette = self.built_cassette()
            frames = _cassette_frame_count(cassette) if cassette is not None else 0
            return f"[WS-VCR MISS] recorded sessions={len(self._recorded_sessions)} frames={frames}"
        return "[WS-VCR NOOP] (no websocket traffic)"

    def _played_frame_count(self) -> int:
        if self._cassette is None:
            return 0
        return sum(len(s.frames) for s in self._cassette.sessions[: self._replay_cursor])


def _cassette_frame_count(cassette: WsCassette) -> int:
    return sum(len(s.frames) for s in cassette.sessions)


def load_ws_cassette(client: RedisLike, key: str) -> Optional[WsCassette]:
    from redis.exceptions import RedisError

    try:
        data = client.get(key)
    except RedisError as exc:
        _record_cache_failure("load", exc)
        message = f"WS-VCR redis load failed for {key}; treating as cache miss: {type(exc).__name__}: {exc}"
        _log.warning(message)
        warnings.warn(message, VCRCassetteCacheWarning, stacklevel=2)
        return None
    if data is None:
        return None
    try:
        raw = data.decode("utf-8") if isinstance(data, (bytes, bytearray)) else data
        return WsCassette.model_validate_json(raw)
    except (ValidationError, ValueError, TypeError) as exc:
        _record_cache_failure("load", exc)
        message = (
            f"WS-VCR redis load failed for {key}; cached payload is corrupt, "
            f"treating as cache miss: {type(exc).__name__}: {exc}"
        )
        _log.warning(message)
        warnings.warn(message, VCRCassetteCacheWarning, stacklevel=2)
        return None


def save_ws_cassette(
    client: RedisLike,
    key: str,
    cassette: WsCassette,
    passed: bool,
    ttl_seconds: int = CASSETTE_TTL_SECONDS,
) -> bool:
    from redis.exceptions import RedisError

    if not passed:
        _log.info("WS-VCR redis save skipped for %s; test did not pass - leaving any prior cassette intact", key)
        return False
    if len(cassette.sessions) > WS_MAX_SESSIONS_PER_CASSETTE:
        _log.warning(
            "WS-VCR redis save refused for %s; %d sessions (> WS_MAX_SESSIONS_PER_CASSETTE=%d)",
            key,
            len(cassette.sessions),
            WS_MAX_SESSIONS_PER_CASSETTE,
        )
        return False
    if any(len(session.frames) > WS_MAX_FRAMES_PER_SESSION for session in cassette.sessions):
        _log.warning(
            "WS-VCR redis save refused for %s; a session exceeds WS_MAX_FRAMES_PER_SESSION=%d",
            key,
            WS_MAX_FRAMES_PER_SESSION,
        )
        return False
    payload = cassette.model_dump_json().encode("utf-8")
    try:
        client.set(key, payload, ex=ttl_seconds)
    except RedisError as exc:
        _record_cache_failure("save", exc)
        message = f"WS-VCR redis save failed for {key}; cassette not persisted: {type(exc).__name__}: {exc}"
        _log.warning(message)
        warnings.warn(message, VCRCassetteCacheWarning, stacklevel=2)
        return False
    return True


def build_ws_cassette_client(
    builder: Callable[[], RedisLike] = _build_default_client,
) -> Optional[RedisLike]:
    try:
        return builder()
    except Exception as exc:
        _record_cache_failure("load", exc)
        message = (
            f"WS-VCR redis client unavailable; realtime tests fall back to live "
            f"websocket traffic: {type(exc).__name__}: {exc}"
        )
        _log.warning(message)
        warnings.warn(message, VCRCassetteCacheWarning, stacklevel=2)
        return None
