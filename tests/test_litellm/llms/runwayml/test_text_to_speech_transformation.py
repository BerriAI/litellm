"""
Test RunwayML text-to-speech transformation
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.llms.runwayml.text_to_speech.transformation import (
    RunwayMLTextToSpeechConfig,
)


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

def test_transform_text_to_speech_response():
    """Test TTS audio download with SSRF-protected fetch."""
    from unittest.mock import Mock, patch

    import httpx

    import litellm
    from litellm.types.llms.openai import HttpxBinaryResponseContent

    config = RunwayMLTextToSpeechConfig()

    # Mock the initial response (task created)
    mock_response = Mock(spec=httpx.Response)
    mock_response.json.return_value = {"id": "task-123", "status": "PENDING"}
    mock_response.request.headers = {"Authorization": "Bearer test"}

    # Mock the polled response (task completed)
    mock_polled = Mock(spec=httpx.Response)
    mock_polled.json.return_value = {
        "id": "task-123",
        "status": "SUCCEEDED",
        "output": ["https://example.com/audio.mp3"],
    }

    # Mock the audio download response
    mock_audio_response = Mock(spec=httpx.Response)
    mock_audio_response.raise_for_status = Mock()

    mock_client = Mock()
    mock_client.get.return_value = mock_audio_response

    with patch.object(litellm, "user_url_validation", False):
        with patch.object(config, "_poll_task_sync", return_value=mock_polled):
            with patch("litellm.llms.custom_httpx.http_handler._get_httpx_client", return_value=mock_client):
                result = config.transform_text_to_speech_response(
                    model="eleven_multilingual_v2",
                    raw_response=mock_response,
                    logging_obj=Mock(),
                )

    assert isinstance(result, HttpxBinaryResponseContent)
    mock_client.get.assert_called_once()


def test_async_transform_text_to_speech_response():
    """Test async TTS audio download with SSRF-protected fetch."""
    import asyncio
    from unittest.mock import AsyncMock, Mock, patch

    import httpx

    import litellm
    from litellm.types.llms.openai import HttpxBinaryResponseContent

    config = RunwayMLTextToSpeechConfig()

    # Mock the initial response (task created)
    mock_response = Mock(spec=httpx.Response)
    mock_response.json.return_value = {"id": "task-123", "status": "PENDING"}
    mock_response.request.headers = {"Authorization": "Bearer test"}

    # Mock the polled response (task completed)
    mock_polled = Mock(spec=httpx.Response)
    mock_polled.json.return_value = {
        "id": "task-123",
        "status": "SUCCEEDED",
        "output": ["https://example.com/audio.mp3"],
    }

    # Mock the audio download response
    mock_audio_response = Mock(spec=httpx.Response)
    mock_audio_response.raise_for_status = Mock()

    mock_client = Mock()
    mock_client.get = AsyncMock(return_value=mock_audio_response)

    async def run_test():
        with patch.object(litellm, "user_url_validation", False):
            with patch.object(config, "_poll_task_async", new_callable=AsyncMock, return_value=mock_polled):
                with patch("litellm.llms.custom_httpx.http_handler.get_async_httpx_client", return_value=mock_client):
                    result = await config.async_transform_text_to_speech_response(
                        model="eleven_multilingual_v2",
                        raw_response=mock_response,
                        logging_obj=Mock(),
                    )
        return result

    result = asyncio.run(run_test())
    assert isinstance(result, HttpxBinaryResponseContent)
    mock_client.get.assert_called_once()

