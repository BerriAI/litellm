"""
Test RunwayML text-to-speech transformation
"""

from collections.abc import Iterator
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.llms.runwayml.text_to_speech.transformation import (
    RunwayMLTextToSpeechConfig,
)


@pytest.fixture
def httpx_transport(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    yield
    litellm.in_memory_llm_clients_cache.flush_cache()


@pytest.mark.asyncio
async def test_async_speech_polls_the_runwayml_task_with_the_caller_key_and_returns_its_audio(
    respx_mock: respx.MockRouter, httpx_transport: None
) -> None:
    respx_mock.post("https://api.dev.runwayml.com/v1/text_to_speech").mock(
        return_value=httpx.Response(200, json={"id": "task_1", "status": "PENDING"})
    )
    poll: Final = respx_mock.get("https://api.dev.runwayml.com/v1/tasks/task_1").mock(
        return_value=httpx.Response(
            200, json={"id": "task_1", "status": "SUCCEEDED", "output": ["https://storage.example.com/a1.mp3"]}
        )
    )
    respx_mock.get("https://storage.example.com/a1.mp3").mock(
        return_value=httpx.Response(200, content=b"ID3runway-audio")
    )

    speech: Final = await litellm.aspeech(
        model="runwayml/eleven_multilingual_v2", input="Hello.", voice="alloy", api_key="rw-key"
    )

    assert speech.content == b"ID3runway-audio"
    assert poll.calls.last.request.headers["authorization"] == "Bearer rw-key"


def test_openai_voice_mapping_to_runwayml():
    """
    Test that OpenAI voice names are correctly mapped to RunwayML preset IDs
    """
    config = RunwayMLTextToSpeechConfig()

    # Test OpenAI voice mappings
    openai_to_runway = {
        "alloy": "Maya",
        "echo": "James",
        "fable": "Bernard",
        "onyx": "Vincent",
        "nova": "Serene",
        "shimmer": "Ella",
    }

    for openai_voice, expected_runway_voice in openai_to_runway.items():
        mapped_voice, mapped_params = config.map_openai_params(
            model="eleven_multilingual_v2",
            optional_params={},
            voice=openai_voice,
            drop_params=False,
            kwargs={},
        )

        assert mapped_voice is None
        assert "runwayml_voice" in mapped_params
        assert mapped_params["runwayml_voice"]["type"] == "runway-preset"
        assert mapped_params["runwayml_voice"]["presetId"] == expected_runway_voice


def test_runwayml_native_voice_passthrough():
    """
    Test that RunwayML native voice names are passed through correctly as-is
    """
    config = RunwayMLTextToSpeechConfig()

    # Test various RunwayML native voices
    runway_voices = ["Bernard", "Maya", "Arjun", "Serene", "Chad"]

    for runway_voice in runway_voices:
        mapped_voice, mapped_params = config.map_openai_params(
            model="eleven_multilingual_v2",
            optional_params={},
            voice=runway_voice,
            drop_params=False,
            kwargs={},
        )

        assert mapped_voice is None
        assert "runwayml_voice" in mapped_params
        assert mapped_params["runwayml_voice"]["type"] == "runway-preset"
        assert mapped_params["runwayml_voice"]["presetId"] == runway_voice
