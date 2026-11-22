from __future__ import annotations

import asyncio
import contextlib
import datetime
from collections.abc import Callable
from typing import Final
from unittest.mock import patch

import httpx
import pytest

import litellm
from litellm.router_rust.invoke.slots import hold_slot_until_closed
from litellm.router_rust.invoke.streaming import _RECORDINGS
from litellm.router_utils.router_callbacks.track_deployment_metrics import (
    get_deployment_failures_for_current_minute,
    get_deployment_successes_for_current_minute,
)
from litellm.types.llms.openai import HttpxBinaryResponseContent
from litellm.utils import calculate_max_parallel_requests

_ROUTER_CLOCK: Final = "litellm.router_strategy.lowest_tpm_rpm_v2.get_utc_datetime"
_FIRST_MINUTE: Final = datetime.datetime(2026, 1, 1, 12, 0, 59, tzinfo=datetime.timezone.utc)
_NEXT_MINUTE: Final = datetime.datetime(2026, 1, 1, 12, 1, 0, tzinfo=datetime.timezone.utc)


def _rpm_limited_router() -> litellm.Router:
    return litellm.Router(
        model_list=[
            {
                "model_name": "limited",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "test-key",
                    "rpm": 1,
                },
            }
        ],
        routing_strategy="usage-based-routing-v2",
        enable_pre_call_checks=True,
        num_retries=0,
    )


_RPM_LIMITED_REQUEST: Final = {
    "model": "limited",
    "messages": [{"role": "user", "content": "rate limit"}],
    "mock_response": "response",
}


@pytest.mark.parametrize(
    ("max_parallel_requests", "tpm", "rpm", "default_max_parallel_requests", "expected"),
    [
        pytest.param(10, 300_000, 30, 40, 10, id="explicit-limit"),
        pytest.param(None, 300_000, 30, 40, 30, id="rpm"),
        pytest.param(None, 300_000, None, 40, 1_800, id="tpm"),
        pytest.param(None, 20, None, 40, 1, id="minimum-tpm-limit"),
        pytest.param(None, None, None, 40, 40, id="router-default"),
        pytest.param(None, None, None, None, None, id="unset"),
    ],
)
def test_scenario(
    max_parallel_requests: int | None,
    tpm: int | None,
    rpm: int | None,
    default_max_parallel_requests: int | None,
    expected: int | None,
) -> None:
    calculated: Final = calculate_max_parallel_requests(
        max_parallel_requests=max_parallel_requests,
        rpm=rpm,
        tpm=tpm,
        default_max_parallel_requests=default_max_parallel_requests,
    )

    assert calculated == expected


@pytest.mark.asyncio
async def test_max_parallel_requests_rpm_rate_limiting() -> None:
    router: Final = _rpm_limited_router()

    with patch(_ROUTER_CLOCK, return_value=_FIRST_MINUTE):
        first_response: Final = await router.acompletion(**_RPM_LIMITED_REQUEST)

        assert first_response.choices[0].message.content == "response"
        with pytest.raises(litellm.RateLimitError):
            await router.acompletion(**_RPM_LIMITED_REQUEST)


@pytest.mark.asyncio
async def test_rpm_limit_resets_in_the_next_minute() -> None:
    router: Final = _rpm_limited_router()

    with patch(_ROUTER_CLOCK, return_value=_FIRST_MINUTE):
        await router.acompletion(**_RPM_LIMITED_REQUEST)
    with patch(_ROUTER_CLOCK, return_value=_NEXT_MINUTE):
        next_minute_response: Final = await router.acompletion(**_RPM_LIMITED_REQUEST)

    assert next_minute_response.choices[0].message.content == "response"


@pytest.mark.asyncio
async def test_max_parallel_requests_tpm_rate_limiting_base_case() -> None:
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "limited",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "test-key",
                    "tpm": 1,
                },
            }
        ],
        routing_strategy="usage-based-routing-v2",
        enable_pre_call_checks=True,
        num_retries=0,
    )
    request: Final = {
        "model": "limited",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
        "mock_response": "ok",
    }

    with pytest.raises(litellm.RateLimitError):
        await router.acompletion(**request)


class _OneChunkAudioStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b"audio"

    async def aclose(self) -> None:
        return None


def _speech_router() -> litellm.Router:
    return litellm.Router(
        model_list=[
            {
                "model_name": "tts",
                "litellm_params": {
                    "model": "elevenlabs/eleven_multilingual_v2",
                    "api_key": "fake-key",
                    "max_parallel_requests": 1,
                },
            }
        ]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False], ids=["streamed", "buffered"])
async def test_aspeech_holds_its_deployment_slot_until_the_audio_is_released(
    monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    """A streamed synthesis occupies its deployment's only slot for as long as audio is
    still arriving, so a second caller is turned away rather than doubling up on it."""
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    router: Final = _speech_router()

    def _client() -> litellm.AsyncHTTPHandler:
        return litellm.AsyncHTTPHandler(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, headers={"content-type": "audio/mpeg"}, stream=_OneChunkAudioStream())
            )
        )

    response: Final = await router.aspeech(
        model="tts", input="hello", voice="voice-id", client=_client(), stream=stream
    )

    async def _second_caller() -> str:
        try:
            await router.aspeech(model="tts", input="hello", voice="voice-id", client=_client(), stream=stream)
        except litellm.RateLimitError:
            return "refused"
        return "admitted"

    while_audio_open: Final = await _second_caller()
    await response.aclose()
    await asyncio.sleep(0)

    assert (while_audio_open, await _second_caller()) == ("refused" if stream else "admitted", "admitted")


class _AudioCutOffMidClip(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b"audio"
        raise httpx.RemoteProtocolError("peer closed connection without sending complete message body")

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_aspeech_releases_its_deployment_slot_when_the_audio_read_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provider cutting a streamed clip off mid-read frees the slot even when the caller never closes the response."""
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    router: Final = _speech_router()

    def _client() -> litellm.AsyncHTTPHandler:
        return litellm.AsyncHTTPHandler(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, headers={"content-type": "audio/mpeg"}, stream=_AudioCutOffMidClip())
            )
        )

    response: Final = await router.aspeech(model="tts", input="hello", voice="voice-id", client=_client(), stream=True)
    with pytest.raises(litellm.APIConnectionError):
        await response.aread()

    await router.aspeech(model="tts", input="hello", voice="voice-id", client=_client(), stream=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "releases_immediately"),
    [
        pytest.param(HttpxBinaryResponseContent(httpx.Response(200, content=b"audio")), True, id="buffered-body"),
        pytest.param("not-a-binary-response", True, id="non-binary-response"),
        pytest.param(
            HttpxBinaryResponseContent(httpx.Response(200, stream=_OneChunkAudioStream())), False, id="unread-body"
        ),
    ],
)
async def test_hold_slot_until_closed_keys_off_whether_the_body_is_still_arriving(
    response: object, releases_immediately: bool
) -> None:
    """Only a response whose body has yet to be read keeps its slot past the attempt, everything else frees it."""
    released: Final[list[str]] = []

    with contextlib.ExitStack() as attempt_slot:
        attempt_slot.callback(released.append, "released")
        hold_slot_until_closed(response, attempt_slot, lambda _end: None)

    assert bool(released) is releases_immediately
    if not releases_immediately:
        await response.aclose()
        assert released == ["released"]


class _AudioStalledMidClip(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b"audio"
        raise httpx.ReadTimeout("the provider stopped sending audio")

    async def aclose(self) -> None:
        return None


def _tts_deployment(model_name: str, api_base: str, **params: object) -> dict[str, object]:
    return {
        "model_name": model_name,
        "litellm_params": {
            "model": "elevenlabs/eleven_multilingual_v2",
            "api_key": "fake-key",
            "api_base": api_base,
            "max_parallel_requests": 1,
            **params,
        },
        "model_info": {"id": api_base},
    }


def _client_answering(answer: Callable[[httpx.Request], httpx.Response]) -> litellm.AsyncHTTPHandler:
    return litellm.AsyncHTTPHandler(transport=httpx.MockTransport(answer))


def _audio(stream: httpx.AsyncByteStream) -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "audio/mpeg"}, stream=stream)


@pytest.mark.asyncio
async def test_aspeech_falls_back_on_a_failure_before_the_headers_and_frees_the_failed_slot() -> None:
    """A provider refusing a streamed clip before any audio fails the attempt, so the fallback serves it and the
    failed deployment's only slot is free for the next caller."""
    router: Final = litellm.Router(
        model_list=[
            _tts_deployment("tts", "http://refusing.test"),
            _tts_deployment("tts-backup", "http://backup.test"),
        ],
        fallbacks=[{"tts": ["tts-backup"]}],
        num_retries=0,
        disable_cooldowns=True,
    )
    hosts: Final[list[str]] = []

    def _answer(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        if request.url.host == "refusing.test":
            return httpx.Response(500, json={"detail": "synthesis failed"})
        return _audio(_OneChunkAudioStream())

    async def _clip() -> bytes:
        response: Final = await router.aspeech(
            model="tts", input="hello", voice="voice-id", client=_client_answering(_answer), stream=True
        )
        return await response.aread()

    assert (await _clip(), await _clip()) == (b"audio", b"audio")
    assert hosts == ["refusing.test", "backup.test", "refusing.test", "backup.test"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("audio", "read_it", "counted"),
    [
        pytest.param(_OneChunkAudioStream, True, (1, 0), id="read-to-the-end"),
        pytest.param(_AudioCutOffMidClip, True, (0, 1), id="cut-off-mid-read"),
        pytest.param(_AudioCutOffMidClip, False, (1, 0), id="caller-hung-up"),
    ],
)
async def test_aspeech_counts_a_streamed_clip_against_its_deployment_when_the_audio_ends(
    audio: type[httpx.AsyncByteStream], read_it: bool, counted: tuple[int, int]
) -> None:
    """The attempt returns at the headers, so only the audio's end says how the clip went: a provider cutting it
    off is a failure, and a caller walking away says nothing bad about the deployment."""
    router: Final = litellm.Router(model_list=[_tts_deployment("tts", "http://tts.test")], num_retries=0)

    def _outcome() -> tuple[int, int]:
        return (
            get_deployment_successes_for_current_minute(router, "http://tts.test"),
            get_deployment_failures_for_current_minute(router, "http://tts.test"),
        )

    clip: Final = await router.aspeech(
        model="tts",
        input="hello",
        voice="voice-id",
        client=_client_answering(lambda _: _audio(audio())),
        stream=True,
    )
    at_the_headers: Final = _outcome()
    if read_it:
        with contextlib.suppress(litellm.APIConnectionError):
            await clip.aread()
    else:
        await clip.aclose()
    await asyncio.gather(*_RECORDINGS)

    assert (at_the_headers, _outcome()) == ((0, 0), counted)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("read_it", "next_host"),
    [
        pytest.param(True, "backup.test", id="provider-stalled-mid-clip"),
        pytest.param(False, "first.test", id="caller-hung-up"),
    ],
)
async def test_aspeech_cools_a_deployment_down_for_a_clip_that_stalled_after_the_headers(
    read_it: bool, next_host: str
) -> None:
    """A stalled clip is a 200 to the attempt, so the failure the audio raises is what takes the deployment out of
    rotation for the next caller."""
    router: Final = litellm.Router(
        model_list=[
            _tts_deployment("tts", "http://first.test", order=1),
            _tts_deployment("tts", "http://backup.test", order=2),
        ],
        num_retries=0,
        allowed_fails=0,
    )
    hosts: Final[list[str]] = []

    def _answer(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        return _audio(_AudioStalledMidClip())

    async def _clip() -> HttpxBinaryResponseContent:
        return await router.aspeech(
            model="tts", input="hello", voice="voice-id", client=_client_answering(_answer), stream=True
        )

    first_clip: Final = await _clip()
    if read_it:
        with pytest.raises(litellm.Timeout):
            await first_clip.aread()
    else:
        await first_clip.aclose()
    await asyncio.gather(*_RECORDINGS)
    await (await _clip()).aclose()

    assert hosts == ["first.test", next_host]
