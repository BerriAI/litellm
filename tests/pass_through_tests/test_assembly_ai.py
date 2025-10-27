"""
This test ensures that the proxy can passthrough requests to assemblyai
"""

import pytest
import assemblyai as aai
import aiohttp
import asyncio
import time

TEST_MASTER_KEY = "sk-1234"
TEST_BASE_URL = "http://0.0.0.0:4000/assemblyai"


def test_assemblyai_basic_transcribe():
    print("making basic transcribe request to assemblyai passthrough")

    # Replace with your API key
    aai.settings.api_key = f"Bearer {TEST_MASTER_KEY}"
    aai.settings.base_url = TEST_BASE_URL

    # URL of the file to transcribe
    FILE_URL = "https://assembly.ai/wildfires.mp3"

    # You can also transcribe a local file by passing in a file path
    # FILE_URL = './path/to/file.mp3'

    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(FILE_URL)
    print(transcript)
    print(transcript.id)
    if transcript.id:
        transcript.delete_by_id(transcript.id)
    else:
        pytest.fail("Failed to get transcript id")

    if transcript.status == aai.TranscriptStatus.error:
        print(transcript.error)
        pytest.fail(f"Failed to transcribe file error: {transcript.error}")
    else:
        print(transcript.text)


async def generate_key(calling_key: str) -> str:
    """Helper function to generate a new API key"""
    url = "http://0.0.0.0:4000/key/generate"
    headers = {
        "Authorization": f"Bearer {calling_key}",
        "Content-Type": "application/json",
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json={}) as response:
            if response.status == 200:
                data = await response.json()
                return data.get("key")
            raise Exception(f"Failed to generate key: {response.status}")


@pytest.mark.asyncio
async def test_assemblyai_transcribe_with_non_admin_key():
    # Generate a non-admin key using the helper
    non_admin_key = await generate_key(TEST_MASTER_KEY)
    print(f"Generated non-admin key: {non_admin_key}")

    # Use the non-admin key to transcribe
    # Replace with your API key
    aai.settings.api_key = f"Bearer {non_admin_key}"
    aai.settings.base_url = TEST_BASE_URL

    # URL of the file to transcribe
    FILE_URL = "https://assembly.ai/wildfires.mp3"

    # You can also transcribe a local file by passing in a file path
    # FILE_URL = './path/to/file.mp3'

    request_start_time = time.time()

    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(FILE_URL)
    print(transcript)
    print(transcript.id)
    if transcript.id:
        transcript.delete_by_id(transcript.id)
    else:
        pytest.fail("Failed to get transcript id")

    if transcript.status == aai.TranscriptStatus.error:
        print(transcript.error)
        pytest.fail(f"Failed to transcribe file error: {transcript.error}")
    else:
        print(transcript.text)

    request_end_time = time.time()
    print(f"Request took {request_end_time - request_start_time} seconds")
