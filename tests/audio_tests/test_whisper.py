# What is this?
## Tests `litellm.transcription` endpoint. Outside litellm module b/c of audio file used in testing (it's ~700kb).

import asyncio
import logging
import os
import time
import traceback
from typing import Optional

import aiohttp
import dotenv
import pytest
from dotenv import load_dotenv
from openai import AsyncOpenAI

import litellm

# Get the current directory of the file being run
pwd = os.path.dirname(os.path.realpath(__file__))
print(pwd)

file_path = os.path.join(pwd, "gettysburg.wav")

with open(file_path, "rb") as _f:
    _GETTYSBURG_BYTES = _f.read()


def _audio_file():
    return ("gettysburg.wav", _GETTYSBURG_BYTES, "audio/wav")


load_dotenv()

from litellm import Router


async def _run_transcription(
    model, api_key, api_base, response_format, timestamp_granularities
):
    transcript = await litellm.atranscription(
        model=model,
        file=_audio_file(),
        api_key=api_key,
        api_base=api_base,
        response_format=response_format,
        timestamp_granularities=timestamp_granularities,
        drop_params=True,
    )
    print(f"transcript: {transcript.model_dump()}")
    print(f"transcript hidden params: {transcript._hidden_params}")

    assert transcript.text is not None


@pytest.mark.parametrize(
    "response_format, timestamp_granularities",
    [("json", None), ("vtt", None), ("verbose_json", ["word"])],
)
@pytest.mark.asyncio
@pytest.mark.flaky(retries=3, delay=1)
async def test_transcription_azure_whisper(response_format, timestamp_granularities):
    await _run_transcription(
        model="azure/whisper",
        api_key=os.getenv("AZURE_WHISPER_API_KEY"),
        api_base=os.getenv("AZURE_WHISPER_API_BASE"),
        response_format=response_format,
        timestamp_granularities=timestamp_granularities,
    )
