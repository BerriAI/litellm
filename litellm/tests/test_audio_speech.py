# What is this?
## unit tests for openai tts endpoint

import asyncio
import os
import random
import sys
import time
import traceback
import uuid

from dotenv import load_dotenv

load_dotenv()
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from pathlib import Path

import openai
import pytest

import litellm


@pytest.mark.parametrize(
    "sync_mode",
    [True, False],
)
@pytest.mark.parametrize(
    "model, api_key, api_base",
    [
        (
            "azure/azure-tts",
            os.getenv("AZURE_SWEDEN_API_KEY"),
            os.getenv("AZURE_SWEDEN_API_BASE"),
        ),
        ("openai/tts-1", os.getenv("OPENAI_API_KEY"), None),
    ],
)  # ,
@pytest.mark.asyncio
async def test_audio_speech_litellm(sync_mode, model, api_base, api_key):
    speech_file_path = Path(__file__).parent / "speech.mp3"

    if sync_mode:
        response = litellm.speech(
            model=model,
            voice="alloy",
            input="the quick brown fox jumped over the lazy dogs",
            api_base=api_base,
            api_key=api_key,
            organization=None,
            project=None,
            max_retries=1,
            timeout=600,
            client=None,
            optional_params={},
        )

        from litellm.llms.openai import HttpxBinaryResponseContent

        assert isinstance(response, HttpxBinaryResponseContent)
    else:
        response = await litellm.aspeech(
            model=model,
            voice="alloy",
            input="the quick brown fox jumped over the lazy dogs",
            api_base=api_base,
            api_key=api_key,
            organization=None,
            project=None,
            max_retries=1,
            timeout=600,
            client=None,
            optional_params={},
        )

        from litellm.llms.openai import HttpxBinaryResponseContent

        assert isinstance(response, HttpxBinaryResponseContent)


@pytest.mark.parametrize("mode", ["iterator"])  # "file",
@pytest.mark.asyncio
async def test_audio_speech_router(mode):
    speech_file_path = Path(__file__).parent / "speech.mp3"

    from litellm import Router

    client = Router(
        model_list=[
            {
                "model_name": "tts",
                "litellm_params": {
                    "model": "openai/tts-1",
                },
            },
        ]
    )

    response = await client.aspeech(
        model="tts",
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
    )

    from litellm.llms.openai import HttpxBinaryResponseContent

    assert isinstance(response, HttpxBinaryResponseContent)
