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
async def test_aspeech_sync_provider_called_once_with_marker(monkeypatch):
    """
    #44546: a real synchronous provider (plain def speech() returning the
    binary response, not a coroutine) must be invoked exactly once. The
    returned audio must come from that single call (unique marker), and the
    call must produce exactly one usage/cost record - otherwise a successful
    aspeech() call triggers two upstream syntheses while logging one, making
    user charges and provider cost reconciliation misleading.
    """
    calls = []
    marker = b"RIFF....WAVE-marker-44546"

    def fake_speech(*args, **kwargs):  # synchronous, like the Gemini TTS bridge
        calls.append(kwargs.get("model"))
        return _binary_response(marker)

    monkeypatch.setattr(litellm.main, "speech", fake_speech)
    monkeypatch.setattr(
        litellm.main,
        "get_llm_provider",
        lambda model, api_base=None: ("gemini/gemini-3.8-flash-tts", "gemini", None, None),
    )

    cost_records = []
    monkeypatch.setattr(
        litellm, "success_callback", [lambda *a, **kw: cost_records.append(kw)]
    )

    response = await aspeech(model="gemini/gemini-3.8-flash-tts", input="Hello.", voice="Kore")

    assert len(calls) == 1, "sync provider must be called exactly once"
    assert response.content == marker, "returned audio must come from that single call"
    assert len(cost_records) == 1, "exactly one usage/cost record per call"


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
