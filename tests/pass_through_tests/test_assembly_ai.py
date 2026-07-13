"""
This test ensures that the proxy can passthrough requests to assemblyai
"""

import time

import pytest
import httpx
import aiohttp
import asyncio

TEST_MASTER_KEY = "sk-1234"
TEST_BASE_URL = "http://0.0.0.0:4000/assemblyai"


def _transcribe_and_verify(virtual_key: str, base_url: str):
    file_url = "https://assembly.ai/wildfires.mp3"
    headers = {
        "Authorization": f"Bearer {virtual_key}",
        "Content-Type": "application/json",
    }
    create_payload = {
        "audio_url": file_url,
        "speech_models": ["universal-2"],
    }

    create_response = httpx.post(
        url=f"{base_url}/v2/transcript",
        headers=headers,
        json=create_payload,
        timeout=60.0,
    )
    if create_response.status_code != 200:
        pytest.fail(
            "Failed to create transcript request: "
            f"status={create_response.status_code}, body={create_response.text}"
        )

    transcript = create_response.json()
    transcript_id = transcript.get("id")
    if not transcript_id:
        pytest.fail("Failed to get transcript id")

    for _ in range(60):
        poll_response = httpx.get(
            url=f"{base_url}/v2/transcript/{transcript_id}",
            headers=headers,
            timeout=30.0,
        )
        if poll_response.status_code != 200:
            pytest.fail(
                "Failed to poll transcript status: "
                f"status={poll_response.status_code}, body={poll_response.text}"
            )
        transcript = poll_response.json()
        if transcript.get("status") in ("completed", "error"):
            break
        time.sleep(1)

    httpx.delete(
        url=f"{base_url}/v2/transcript/{transcript_id}",
        headers=headers,
        timeout=30.0,
    )

    if transcript.get("status") == "error":
        pytest.fail(f"Failed to transcribe file error: {transcript.get('error')}")

    print(transcript.get("text"))


def test_assemblyai_basic_transcribe():
    print("making basic transcribe request to assemblyai passthrough")
    _transcribe_and_verify(TEST_MASTER_KEY, TEST_BASE_URL)


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
    non_admin_key = await generate_key(TEST_MASTER_KEY)
    print(f"Generated non-admin key: {non_admin_key}")

    request_start_time = time.time()
    _transcribe_and_verify(non_admin_key, TEST_BASE_URL)
    request_end_time = time.time()
    print(f"Request took {request_end_time - request_start_time} seconds")
