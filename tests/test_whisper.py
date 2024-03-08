# What is this?
## Tests `litellm.transcription` endpoint
import pytest
import asyncio, time
import aiohttp
from openai import AsyncOpenAI
import sys, os, dotenv
from typing import Optional
from dotenv import load_dotenv

audio_file = open("./gettysburg.wav", "rb")

load_dotenv()

sys.path.insert(
    0, os.path.abspath("../")
)  # Adds the parent directory to the system path
import litellm


def test_transcription():
    transcript = litellm.transcription(
        model="whisper-1",
        file=audio_file,
    )
    print(f"transcript: {transcript}")


# test_transcription()


def test_transcription_azure():
    transcript = litellm.transcription(
        model="azure/azure-whisper",
        file=audio_file,
        api_base=os.getenv("AZURE_EUROPE_API_BASE"),
        api_key=os.getenv("AZURE_EUROPE_API_KEY"),
        api_version=os.getenv("2024-02-15-preview"),
    )

    assert transcript.text is not None
    assert isinstance(transcript.text, str)


# test_transcription_azure()


@pytest.mark.asyncio
async def test_transcription_async_azure():
    transcript = await litellm.atranscription(
        model="azure/azure-whisper",
        file=audio_file,
        api_base=os.getenv("AZURE_EUROPE_API_BASE"),
        api_key=os.getenv("AZURE_EUROPE_API_KEY"),
        api_version=os.getenv("2024-02-15-preview"),
    )

    assert transcript.text is not None
    assert isinstance(transcript.text, str)


# asyncio.run(test_transcription_async_azure())


@pytest.mark.asyncio
async def test_transcription_async_openai():
    transcript = await litellm.atranscription(
        model="whisper-1",
        file=audio_file,
    )

    assert transcript.text is not None
    assert isinstance(transcript.text, str)
