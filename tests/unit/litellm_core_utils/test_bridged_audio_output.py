import json
from collections.abc import Awaitable, Callable, Iterator
from typing import Final, TypeAlias

import httpx
import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache

GEMINI_API: Final = "https://generativelanguage.googleapis.com/"
VOICE_API: Final = "https://voice.test/v1"
TTS_MODEL: Final = "gemini/gemini-3.8-flash-tts"
SINGER_MODEL: Final = "gemini/litellm-test-singer"
VOICE_MODEL: Final = "custom_openai/litellm-test-voice"
UNLISTED_MODEL: Final = "custom_openai/litellm-test-unlisted"
SPOKEN_REPLY: Final = {
    "candidates": [
        {
            "content": {
                "role": "model",
                "parts": [{"inlineData": {"mimeType": "audio/L16;codec=pcm;rate=24000", "data": "AAAAAA=="}}],
            },
            "finishReason": "STOP",
            "index": 0,
        }
    ],
    "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 3, "totalTokenCount": 5},
}
SUNG_REPLY: Final = {
    "candidates": [
        {
            "content": {
                "role": "model",
                "parts": [
                    {"text": "La la la"},
                    {"inlineData": {"mimeType": "audio/mpeg", "data": "AAAAAA=="}},
                ],
            },
            "finishReason": "STOP",
            "index": 0,
        }
    ],
    "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 3, "totalTokenCount": 5},
}
TEXT_REPLY: Final = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1,
    "model": "litellm-test-voice",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
}
AUDIO: Final = {"voice": "alloy", "format": "wav"}

Bridge: TypeAlias = Callable[..., Awaitable[object]]


async def _responses(model: str, **params: object) -> object:
    return await litellm.aresponses(model=model, input="Say hello", api_key="test-key", **params)


async def _messages(model: str, **params: object) -> object:
    return await litellm.anthropic_messages(
        model=model,
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=16,
        api_key="test-key",
        **params,
    )


async def _text_completion(model: str, **params: object) -> object:
    return await litellm.atext_completion(model=model, prompt="Say hello", api_key="test-key", **params)


BRIDGES: Final = (
    pytest.param(_responses, "/v1/responses", id="responses"),
    pytest.param(_messages, "/v1/messages", id="messages"),
    pytest.param(_text_completion, "/v1/completions", id="completions"),
)
SURFACES: Final = (
    pytest.param(_responses, id="responses"),
    pytest.param(_messages, id="messages"),
    pytest.param(_text_completion, id="completions"),
)


@pytest.fixture(autouse=True)
def _providers_over_httpx(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    client_cache: Final = LLMClientCache()
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", client_cache)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "force_ipv4", False)
    litellm.register_model(
        {
            SINGER_MODEL: {"litellm_provider": "gemini", "mode": "chat", "supported_output_modalities": ["audio"]},
            VOICE_MODEL: {
                "litellm_provider": "custom_openai",
                "mode": "chat",
                "supported_output_modalities": ["text", "audio"],
            },
        },
        persist_across_reloads=False,
        add_to_provider_model_sets=False,
    )
    yield
    client_cache.flush_cache()


def _upstream(respx_mock: respx.MockRouter, api: str, reply: dict[str, object]) -> respx.Route:
    return respx_mock.post(url__startswith=api).mock(return_value=httpx.Response(200, json=reply))


def _sent_body(route: respx.Route) -> dict[str, object]:
    return json.loads(route.calls.last.request.content)


@pytest.mark.respx(assert_all_called=False)
@pytest.mark.parametrize(("bridge", "endpoint"), BRIDGES)
async def test_tts_model_is_refused_before_any_upstream_call(
    respx_mock: respx.MockRouter, bridge: Bridge, endpoint: str
) -> None:
    upstream: Final = _upstream(respx_mock, GEMINI_API, SPOKEN_REPLY)

    with pytest.raises(litellm.UnsupportedParamsError) as refusal:
        await bridge(TTS_MODEL, drop_params=True)

    assert refusal.value.status_code == 400
    assert f"{TTS_MODEL} returns audio, which {endpoint} cannot carry. Call /v1/chat/completions" in str(refusal.value)
    assert upstream.call_count == 0


@pytest.mark.respx(assert_all_called=True)
@pytest.mark.parametrize("bridge", SURFACES)
async def test_model_listed_with_only_audio_output_still_reaches_upstream(
    respx_mock: respx.MockRouter, bridge: Bridge
) -> None:
    upstream: Final = _upstream(respx_mock, GEMINI_API, SUNG_REPLY)

    await bridge(SINGER_MODEL)

    assert upstream.call_count == 1


@pytest.mark.respx(assert_all_called=False)
@pytest.mark.parametrize(("bridge", "endpoint"), BRIDGES)
@pytest.mark.parametrize(
    ("audio_request", "named"),
    [
        pytest.param({"audio": AUDIO}, "audio", id="audio"),
        pytest.param({"modalities": ["text", "audio"]}, "modalities", id="modalities"),
        pytest.param(
            {"audio": AUDIO, "modalities": ["audio"], "additional_drop_params": ["audio"]},
            "modalities",
            id="only-modalities-left-after-additional-drop",
        ),
    ],
)
async def test_audio_request_is_refused_without_drop_params(
    respx_mock: respx.MockRouter, bridge: Bridge, endpoint: str, audio_request: dict[str, object], named: str
) -> None:
    upstream: Final = _upstream(respx_mock, VOICE_API, TEXT_REPLY)

    with pytest.raises(litellm.UnsupportedParamsError) as refusal:
        await bridge(VOICE_MODEL, api_base=VOICE_API, **audio_request)

    assert refusal.value.status_code == 400
    assert (
        f"asks {VOICE_MODEL} for audio through {named}, which {endpoint} cannot carry. "
        f"Leave out {named}, set drop_params: true, or call /v1/chat/completions"
    ) in str(refusal.value)
    assert upstream.call_count == 0


@pytest.mark.respx(assert_all_called=True)
@pytest.mark.parametrize("bridge", SURFACES)
@pytest.mark.parametrize(
    "drop",
    [
        pytest.param({"drop_params": True}, id="drop_params"),
        pytest.param({"additional_drop_params": ["audio", "modalities"]}, id="additional_drop_params"),
    ],
)
async def test_dropped_audio_request_reaches_upstream_without_audio(
    respx_mock: respx.MockRouter, bridge: Bridge, drop: dict[str, object]
) -> None:
    upstream: Final = _upstream(respx_mock, VOICE_API, TEXT_REPLY)

    await bridge(VOICE_MODEL, api_base=VOICE_API, audio=AUDIO, modalities=["audio"], **drop)

    sent: Final = _sent_body(upstream)
    assert "audio" not in sent
    assert "modalities" not in sent


@pytest.mark.respx(assert_all_called=True)
@pytest.mark.parametrize("bridge", SURFACES)
@pytest.mark.parametrize(
    "audio_request",
    [
        pytest.param({"modalities": ["text", "audio"]}, id="audio-modality"),
        pytest.param({"audio": AUDIO, "modalities": ["text"]}, id="audio-config"),
    ],
)
async def test_dropping_audio_keeps_the_text_modality(
    respx_mock: respx.MockRouter, bridge: Bridge, audio_request: dict[str, object]
) -> None:
    upstream: Final = _upstream(respx_mock, VOICE_API, TEXT_REPLY)

    await bridge(VOICE_MODEL, api_base=VOICE_API, drop_params=True, **audio_request)

    sent: Final = _sent_body(upstream)
    assert sent["modalities"] == ["text"]
    assert "audio" not in sent


@pytest.mark.respx(assert_all_called=True)
@pytest.mark.parametrize("bridge", SURFACES)
@pytest.mark.parametrize("model", [VOICE_MODEL, UNLISTED_MODEL])
async def test_text_request_passes_through(respx_mock: respx.MockRouter, bridge: Bridge, model: str) -> None:
    upstream: Final = _upstream(respx_mock, VOICE_API, TEXT_REPLY)

    await bridge(model, api_base=VOICE_API, modalities=["text"])

    assert _sent_body(upstream)["modalities"] == ["text"]


@pytest.mark.respx(assert_all_called=True)
async def test_chat_completions_still_returns_audio_from_tts_model(respx_mock: respx.MockRouter) -> None:
    upstream: Final = _upstream(respx_mock, GEMINI_API, SPOKEN_REPLY)

    response: Final = await litellm.acompletion(
        model=TTS_MODEL, messages=[{"role": "user", "content": "Say hello"}], api_key="test-key"
    )

    assert upstream.call_count == 1
    assert response.choices[0].message.audio is not None
