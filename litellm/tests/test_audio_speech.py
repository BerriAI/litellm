# What is this?
## unit tests for openai tts endpoint

import sys, os, asyncio, time, random, uuid
import traceback
from dotenv import load_dotenv

load_dotenv()
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest
import litellm, openai
from pathlib import Path


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_audio_speech_openai(sync_mode):

    speech_file_path = Path(__file__).parent / "speech.mp3"
    openai_chat_completions = litellm.OpenAIChatCompletion()
    if sync_mode:
        with openai_chat_completions.audio_speech(
            model="tts-1",
            voice="alloy",
            input="the quick brown fox jumped over the lazy dogs",
            api_base=None,
            api_key=None,
            organization=None,
            project=None,
            max_retries=1,
            timeout=600,
            client=None,
            optional_params={},
        ) as response:
            response.stream_to_file(speech_file_path)
    else:
        async with openai_chat_completions.async_audio_speech(
            model="tts-1",
            voice="alloy",
            input="the quick brown fox jumped over the lazy dogs",
            api_base=None,
            api_key=None,
            organization=None,
            project=None,
            max_retries=1,
            timeout=600,
            client=None,
            optional_params={},
        ) as response:
            speech = await response.parse()


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_audio_speech_litellm(sync_mode):
    speech_file_path = Path(__file__).parent / "speech.mp3"

    if sync_mode:
        with litellm.speech(
            model="openai/tts-1",
            voice="alloy",
            input="the quick brown fox jumped over the lazy dogs",
            api_base=None,
            api_key=None,
            organization=None,
            project=None,
            max_retries=1,
            timeout=600,
            client=None,
            optional_params={},
        ) as response:
            response.stream_to_file(speech_file_path)
    else:
        async with litellm.aspeech(
            model="openai/tts-1",
            voice="alloy",
            input="the quick brown fox jumped over the lazy dogs",
            api_base=None,
            api_key=None,
            organization=None,
            project=None,
            max_retries=1,
            timeout=600,
            client=None,
            optional_params={},
        ) as response:
            await response.stream_to_file(speech_file_path)
