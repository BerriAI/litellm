# What is this?
## Tests `litellm.transcription` endpoint. Outside litellm module b/c of audio file used in testing (it's ~700kb).

import pytest
import asyncio, time
import aiohttp, traceback
from openai import AsyncOpenAI
import sys, os, dotenv
from typing import Optional
from dotenv import load_dotenv

# Get the current directory of the file being run
pwd = os.path.dirname(os.path.realpath(__file__))
print(pwd)

file_path = os.path.join(pwd, "gettysburg.wav")

audio_file = open(file_path, "rb")

load_dotenv()

sys.path.insert(
    0, os.path.abspath("../")
)  # Adds the parent directory to the system path
import litellm
from litellm import Router


def test_transcription():
    transcript = litellm.transcription(
        model="whisper-1",
        file=audio_file,
    )
    print(f"transcript: {transcript.model_dump()}")
    print(f"transcript: {transcript._hidden_params}")


# test_transcription()


def test_transcription_azure():
    litellm.set_verbose = True
    transcript = litellm.transcription(
        model="azure/azure-whisper",
        file=audio_file,
        api_base="https://my-endpoint-europe-berri-992.openai.azure.com/",
        api_key=os.getenv("AZURE_EUROPE_API_KEY"),
        api_version="2024-02-15-preview",
    )

    print(f"transcript: {transcript}")
    assert transcript.text is not None
    assert isinstance(transcript.text, str)


# test_transcription_azure()


@pytest.mark.asyncio
async def test_transcription_async_azure():
    transcript = await litellm.atranscription(
        model="azure/azure-whisper",
        file=audio_file,
        api_base="https://my-endpoint-europe-berri-992.openai.azure.com/",
        api_key=os.getenv("AZURE_EUROPE_API_KEY"),
        api_version="2024-02-15-preview",
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


@pytest.mark.asyncio
async def test_transcription_on_router():
    litellm.set_verbose = True
    print("\n Testing async transcription on router\n")
    try:
        model_list = [
            {
                "model_name": "whisper",
                "litellm_params": {
                    "model": "whisper-1",
                },
            },
            {
                "model_name": "whisper",
                "litellm_params": {
                    "model": "azure/azure-whisper",
                    "api_base": "https://my-endpoint-europe-berri-992.openai.azure.com/",
                    "api_key": os.getenv("AZURE_EUROPE_API_KEY"),
                    "api_version": "2024-02-15-preview",
                },
            },
        ]

        router = Router(model_list=model_list)
        response = await router.atranscription(
            model="whisper",
            file=audio_file,
        )
        print(response)
    except Exception as e:
        traceback.print_exc()
        pytest.fail(f"Error occurred: {e}")
