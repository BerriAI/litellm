import asyncio
import threading
import uuid
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from dataclasses import dataclass
from itertools import accumulate
from typing import Final

import litellm
import pytest
from integration._support.client import held
from integration._support.wire import Reply, Request, wire_server
from litellm.integrations.custom_logger import CustomLogger

_BACKEND: Final = "openai/gpt-4o-mini-tts"
_PROVIDER_KEY: Final = "synthetic-tts-key"
_FIRST: Final = b"audio-frame-0:" + bytes(range(64))
_REST: Final = (
    b"audio-frame-1:" + bytes(range(64, 192)),
    b"audio-frame-2:" + bytes(range(192, 256)),
)
_CLIP: Final = _FIRST + b"".join(_REST)
_TIMEOUT_SECONDS: Final = 20


@dataclass(frozen=True, slots=True)
class Logged:
    call_id: object
    error_code: object
    exception: str


def _logged(kwargs: Mapping[str, object]) -> Logged:
    payload: Final = kwargs.get("standard_logging_object")
    error_information: Final = payload.get("error_information") if isinstance(payload, dict) else None
    return Logged(
        kwargs.get("litellm_call_id"),
        error_information.get("error_code") if isinstance(error_information, dict) else None,
        type(kwargs.get("exception")).__name__,
    )


class _Probe(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self._lock: Final = threading.Lock()
        self._events: Final[list[tuple[str, Logged]]] = []

    def _record(self, kind: str, kwargs: Mapping[str, object]) -> None:
        with self._lock:
            self._events.append((kind, _logged(kwargs)))

    async def async_log_success_event(
        self,
        kwargs: dict[str, object],
        response_obj: object,
        start_time: object,
        end_time: object,
    ) -> None:
        self._record("async_success", kwargs)

    async def async_log_failure_event(
        self,
        kwargs: dict[str, object],
        response_obj: object,
        start_time: object,
        end_time: object,
    ) -> None:
        self._record("async_failure", kwargs)

    def log_success_event(
        self,
        kwargs: dict[str, object],
        response_obj: object,
        start_time: object,
        end_time: object,
    ) -> None:
        self._record("sync_success", kwargs)

    def log_failure_event(
        self,
        kwargs: dict[str, object],
        response_obj: object,
        start_time: object,
        end_time: object,
    ) -> None:
        self._record("sync_failure", kwargs)

    def events(self) -> tuple[tuple[str, Logged], ...]:
        with self._lock:
            return tuple(self._events)


def _peer(release: threading.Event | None = None, abort_after: int | None = None) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", "/audio/speech"), request.target
        return Reply(
            content_type="audio/mpeg",
            chunks=(_FIRST, *_REST),
            gate_after_first=release,
            abort_after=abort_after,
        )

    return respond


def _first_bytes(chunks: Iterator[bytes], size: int) -> bytes:
    return next(heard for heard in accumulate(chunks) if len(heard) >= size)


async def _afirst_bytes(chunks: AsyncIterator[bytes], size: int, heard: bytes = b"") -> bytes:
    if len(heard) >= size:
        return heard
    return await _afirst_bytes(chunks, size, heard + await anext(chunks))


def _probe() -> _Probe:
    probe: Final = _Probe()
    litellm.callbacks.append(probe)
    return probe


def _nothing_logged(events: tuple[tuple[str, Logged], ...]) -> bool:
    return events == ()


def _logged_exactly(probe: _Probe, count: int) -> tuple[tuple[str, Logged], ...]:
    return held(probe.events, lambda events: len(events) == count, holding=2)


async def _aspeech(
    api_base: str, call_id: str, metadata: Mapping[str, str] | None = None
) -> litellm.HttpxBinaryResponseContent:
    return await asyncio.wait_for(
        litellm.aspeech(
            model=_BACKEND,
            input=f"streamed speech {call_id}",
            voice="alloy",
            api_base=api_base,
            api_key=_PROVIDER_KEY,
            stream=True,
            metadata=metadata,
            litellm_call_id=call_id,
            timeout=_TIMEOUT_SECONDS,
        ),
        _TIMEOUT_SECONDS,
    )


def _speech(api_base: str, call_id: str) -> litellm.HttpxBinaryResponseContent:
    return litellm.speech(
        model=_BACKEND,
        input=f"streamed speech {call_id}",
        voice="alloy",
        api_base=api_base,
        api_key=_PROVIDER_KEY,
        stream=True,
        litellm_call_id=call_id,
        timeout=_TIMEOUT_SECONDS,
    )


async def test_aspeech_stream_returns_at_the_first_audio_and_logs_success_once_after_the_clip() -> None:
    probe: Final = _probe()
    release: Final = threading.Event()
    call_id: Final = str(uuid.uuid4())
    with wire_server(_peer(release)) as wire:
        response: Final = await _aspeech(wire.url, call_id)
        chunks: Final = await response.aiter_bytes()
        first: Final = await _afirst_bytes(chunks, len(_FIRST))
        await asyncio.to_thread(held, probe.events, _nothing_logged, holding=2)
        release.set()
        assert first + b"".join([chunk async for chunk in chunks]) == _CLIP
        logged: Final = await asyncio.to_thread(_logged_exactly, probe, 1)
    assert logged == (("async_success", Logged(call_id, "", "NoneType")),)


async def test_aspeech_stream_closed_early_logs_success_once_with_a_499_marker() -> None:
    probe: Final = _probe()
    release: Final = threading.Event()
    call_id: Final = str(uuid.uuid4())
    with wire_server(_peer(release)) as wire:
        response: Final = await _aspeech(wire.url, call_id)
        assert await _afirst_bytes(await response.aiter_bytes(), len(_FIRST)) == _FIRST
        await response.aclose()
        logged: Final = await asyncio.to_thread(_logged_exactly, probe, 1)
        release.set()
    assert logged == (("async_success", Logged(call_id, "499", "NoneType")),)


async def test_aspeech_stream_closed_early_leaves_the_callers_metadata_unmarked_for_its_next_call() -> None:
    probe: Final = _probe()
    release: Final = threading.Event()
    caller_metadata: Final = {"tenant": "caller-owned"}
    closed_early: Final = str(uuid.uuid4())
    completed: Final = str(uuid.uuid4())
    with wire_server(_peer(release)) as wire:
        first: Final = await _aspeech(wire.url, closed_early, caller_metadata)
        assert await _afirst_bytes(await first.aiter_bytes(), len(_FIRST)) == _FIRST
        await first.aclose()
        await asyncio.to_thread(_logged_exactly, probe, 1)
        release.set()
        assert caller_metadata.keys() & {"client_disconnected", "error_information"} == set(), caller_metadata
        second: Final = await _aspeech(wire.url, completed, caller_metadata)
        assert b"".join([chunk async for chunk in await second.aiter_bytes()]) == _CLIP
        logged: Final = await asyncio.to_thread(_logged_exactly, probe, 2)
    assert caller_metadata["tenant"] == "caller-owned"
    assert logged == (
        ("async_success", Logged(closed_early, "499", "NoneType")),
        ("async_success", Logged(completed, "", "NoneType")),
    )


def test_speech_takes_base_url_in_its_positional_slot() -> None:
    probe: Final = _probe()
    call_id: Final = str(uuid.uuid4())
    text: Final = f"positional base_url {call_id}"
    with wire_server(_peer()) as wire:
        positional_through_base_url: Final = (
            _BACKEND,
            text,
            "alloy",
            _PROVIDER_KEY,
            *(None,) * 6,
            _TIMEOUT_SECONDS,
            *(None,) * 7,
            wire.url,
        )
        response: Final = litellm.speech(*positional_through_base_url, litellm_call_id=call_id)
        assert response.content == _CLIP
        received: Final = wire.drain()
        logged: Final = _logged_exactly(probe, 1)
    assert [(request.method, request.target) for request in received] == [("POST", "/audio/speech")]
    assert text.encode() in received[0].body
    assert logged == (("sync_success", Logged(call_id, "", "NoneType")),)


async def test_aspeech_stream_cut_off_raises_a_mapped_error_and_logs_one_failure() -> None:
    probe: Final = _probe()
    call_id: Final = str(uuid.uuid4())
    with wire_server(_peer(abort_after=1)) as wire:
        response: Final = await _aspeech(wire.url, call_id)
        chunks: Final = await response.aiter_bytes()
        assert await _afirst_bytes(chunks, len(_FIRST)) == _FIRST
        with pytest.raises(litellm.APIConnectionError):
            await anext(chunks)
        logged: Final = await asyncio.to_thread(_logged_exactly, probe, 2)
    assert sorted(logged) == [
        (kind, Logged(call_id, "500", "APIConnectionError")) for kind in ("async_failure", "sync_failure")
    ]


def test_speech_stream_returns_at_the_first_audio_and_logs_success_once_after_the_clip() -> None:
    probe: Final = _probe()
    release: Final = threading.Event()
    call_id: Final = str(uuid.uuid4())
    with wire_server(_peer(release)) as wire:
        response: Final = _speech(wire.url, call_id)
        chunks: Final = response.iter_bytes()
        first: Final = _first_bytes(chunks, len(_FIRST))
        held(probe.events, _nothing_logged, holding=2)
        release.set()
        assert first + b"".join(chunks) == _CLIP
        logged: Final = _logged_exactly(probe, 1)
    assert logged == (("sync_success", Logged(call_id, "", "NoneType")),)


def test_speech_stream_cut_off_raises_a_mapped_error_and_logs_one_failure() -> None:
    probe: Final = _probe()
    call_id: Final = str(uuid.uuid4())
    with wire_server(_peer(abort_after=1)) as wire:
        response: Final = _speech(wire.url, call_id)
        chunks: Final = response.iter_bytes()
        assert _first_bytes(chunks, len(_FIRST)) == _FIRST
        with pytest.raises(litellm.APIConnectionError):
            next(chunks)
        logged: Final = _logged_exactly(probe, 1)
    assert logged == (("sync_failure", Logged(call_id, "500", "APIConnectionError")),)
