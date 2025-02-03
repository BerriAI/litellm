"""
This test ensures that the proxy can passthrough anthropic requests
"""

import pytest
import assemblyai
import aiohttp
import asyncio


def test_assemblyai_basic_transcribe():
    print("making basic transcribe request to assemblyai passthrough")
    import assemblyai as aai

    # Replace with your API key
    aai.settings.api_key = "Bearer sk-1234"
    aai.settings.base_url = "http://0.0.0.0:4000/assemblyai"

    # URL of the file to transcribe
    FILE_URL = "https://assembly.ai/wildfires.mp3"

    # You can also transcribe a local file by passing in a file path
    # FILE_URL = './path/to/file.mp3'

    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(FILE_URL)

    if transcript.status == aai.TranscriptStatus.error:
        print(transcript.error)
        pytest.fail(f"Failed to transcribe file error: {transcript.error}")
    else:
        print(transcript.text)
