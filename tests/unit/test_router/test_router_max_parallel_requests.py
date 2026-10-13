from __future__ import annotations

import asyncio

from typing import Final

import httpx
import litellm
import pytest
from litellm.router_utils.client_initalization_utils import MaxParallelRequestsLimit
from litellm.utils import calculate_max_parallel_requests


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
def test_setting_mpr_limits_per_model(
    max_parallel_requests: int | None,
    tpm: int | None,
    rpm: int | None,
    default_max_parallel_requests: int | None,
    expected: int | None,
) -> None:
    deployment: Final = {
        "model_name": "limited",
        "litellm_params": {
            "model": "openai/gpt-4o-mini",
            "api_key": "test-key",
            "max_parallel_requests": max_parallel_requests,
            "tpm": tpm,
            "rpm": rpm,
        },
        "model_info": {"id": "limited-deployment"},
    }
    router: Final = litellm.Router(
        model_list=[deployment],
        default_max_parallel_requests=default_max_parallel_requests,
    )
    limit: Final = router._get_client(
        deployment=deployment,
        kwargs={},
        client_type="max_parallel_requests",
    )

    if expected is None:
        assert limit is None
        return

    assert isinstance(limit, MaxParallelRequestsLimit)
    assert limit.max_parallel_requests == expected


@pytest.mark.asyncio
async def test_max_parallel_requests_rpm_rate_limiting() -> None:
    router: Final = litellm.Router(
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
    request: Final = {
        "model": "limited",
        "messages": [{"role": "user", "content": "rate limit"}],
        "mock_response": "response",
    }

    first_response: Final = await router.acompletion(**request)

    assert first_response.choices[0].message.content == "response"
    with pytest.raises(litellm.RateLimitError):
        await router.acompletion(**request)


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
