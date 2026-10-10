# What is this?
## unit tests for openai tts endpoint

import os

from dotenv import load_dotenv

load_dotenv()

from pathlib import Path

import pytest

import litellm


@pytest.mark.asyncio
@pytest.mark.flaky(retries=3, delay=1)
async def test_azure_ava_tts_async():
    """
    Test Azure AVA (Cognitive Services) Text-to-Speech with real API request.
    """
    litellm.turn_on_debug()
    api_key = os.getenv("AZURE_TTS_API_KEY")
    api_base = os.getenv("AZURE_TTS_API_BASE")

    speech_file_path = Path(__file__).parent / "azure_speech.mp3"

    try:
        response = await litellm.aspeech(
            model="azure/tts",
            voice="alloy",
            input="Hello, this is a test of Azure text to speech",
            api_base=api_base,
            api_key=api_key,
            response_format="mp3",
            speed=1.0,
        )

        # Assert the response is HttpxBinaryResponseContent
        from litellm.types.llms.openai import HttpxBinaryResponseContent

        assert isinstance(response, HttpxBinaryResponseContent)

        # Get the binary content
        binary_content = response.content
        assert len(binary_content) > 0

        # MP3 files start with these magic bytes
        # ID3 tag or MPEG sync word
        assert (
            binary_content[:3] == b"ID3"
            or binary_content[:2] == b"\xff\xfb"
            or binary_content[:2] == b"\xff\xf3"
        )

        # Write to file
        response.stream_to_file(speech_file_path)

        # Verify file was created and has content
        assert speech_file_path.exists()
        assert speech_file_path.stat().st_size > 0

        print(f"Azure TTS audio saved to: {speech_file_path}")

    except Exception as e:
        pytest.fail(f"Test failed with exception: {str(e)}")
