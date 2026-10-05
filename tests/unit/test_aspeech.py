import asyncio

import httpx
import pytest

import litellm
from litellm.main import aspeech
from litellm.types.llms.openai import HttpxBinaryResponseContent


def _binary_response(content: bytes) -> HttpxBinaryResponseContent:
    return HttpxBinaryResponseContent(
        response=httpx.Response(
            200,
            content=content,
            request=httpx.Request("POST", "https://example.test"),
        )
    )


@pytest.mark.asyncio
async def test_aspeech_sync_result_is_used_without_second_call(monkeypatch):
    """
    #44546: when the provider's speech path returns synchronously (e.g. Gemini
    TTS through the speech-to-completion bridge), aspeech must use the result
    of the single run_in_executor call instead of invoking the provider a
    second time - otherwise the provider generates and bills the audio twice.
    """
    calls = []
    result = _binary_response(b"RIFF....WAVE-audio")

    async def fake_speech(*args, **kwargs):
        calls.append(kwargs.get("model"))
        return result

    monkeypatch.setattr(litellm.main, "speech", fake_speech)
    monkeypatch.setattr(
        litellm.main,
        "get_llm_provider",
        lambda model, api_base=None: ("gemini/gemini-3.8-flash-tts", "gemini", None, None),
    )

    response = await aspeech(model="gemini/gemini-3.8-flash-tts", input="Hello.", voice="Kore")

    assert len(calls) == 1, "sync provider must be called exactly once"
    assert response is result
    assert response.content == b"RIFF....WAVE-audio"


@pytest.mark.asyncio
async def test_aspeech_awaits_coroutine_result(monkeypatch):
    """
    Async speech providers (the awaitable speech() path) must still be awaited
    exactly once and return the awaited result.
    """
    calls = []
    result = _binary_response(b"async-audio")

    async def fake_speech(*args, **kwargs):
        calls.append(kwargs.get("model"))
        return result

    monkeypatch.setattr(litellm.main, "speech", fake_speech)
    monkeypatch.setattr(
        litellm.main,
        "get_llm_provider",
        lambda model, api_base=None: ("openai/tts-1", "openai", None, None),
    )

    response = await aspeech(model="openai/tts-1", input="Hello.", voice="alloy")

    assert len(calls) == 1
    assert response is result
    assert response.content == b"async-audio"
