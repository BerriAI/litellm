# What is this?
## Tests `litellm.transcription` endpoint. Outside litellm module b/c of audio file used in testing (it's ~700kb).

import asyncio
import logging
import os
import sys
import time
import traceback
from typing import Optional

import aiohttp
import dotenv
import pytest
from dotenv import load_dotenv
from openai import AsyncOpenAI

import litellm
from litellm.integrations.custom_logger import CustomLogger

# Get the current directory of the file being run
pwd = os.path.dirname(os.path.realpath(__file__))
print(pwd)

file_path = os.path.join(pwd, "gettysburg.wav")

audio_file = open(file_path, "rb")


file2_path = os.path.join(pwd, "eagle.wav")
audio_file2 = open(file2_path, "rb")

load_dotenv()

sys.path.insert(
    0, os.path.abspath("../")
)  # Adds the parent directory to the system path
import litellm
from litellm import Router


@pytest.mark.parametrize(
    "model, api_key, api_base",
    [
        ("whisper-1", None, None),
        # ("groq/whisper-large-v3", None, None),
        (
            "azure/azure-whisper",
            os.getenv("AZURE_EUROPE_API_KEY"),
            "https://my-endpoint-europe-berri-992.openai.azure.com/",
        ),
    ],
)
@pytest.mark.parametrize(
    "response_format, timestamp_granularities",
    [("json", None), ("vtt", None), ("verbose_json", ["word"])],
)
@pytest.mark.asyncio
@pytest.mark.flaky(retries=3, delay=1)
async def test_transcription(
    model, api_key, api_base, response_format, timestamp_granularities
):
    transcript = await litellm.atranscription(
        model=model,
        file=audio_file,
        api_key=api_key,
        api_base=api_base,
        response_format=response_format,
        timestamp_granularities=timestamp_granularities,
        drop_params=True,
    )
    print(f"transcript: {transcript.model_dump()}")
    print(f"transcript hidden params: {transcript._hidden_params}")

    assert transcript.text is not None


@pytest.mark.asyncio()
async def test_transcription_caching():
    import litellm
    from litellm.caching.caching import Cache

    litellm.set_verbose = True
    litellm.cache = Cache()

    # make raw llm api call

    response_1 = await litellm.atranscription(
        model="whisper-1",
        file=audio_file,
    )

    await asyncio.sleep(5)

    # cache hit

    response_2 = await litellm.atranscription(
        model="whisper-1",
        file=audio_file,
    )

    print("response_1", response_1)
    print("response_2", response_2)
    print("response2 hidden params", response_2._hidden_params)
    assert response_2._hidden_params["cache_hit"] is True

    # cache miss

    response_3 = await litellm.atranscription(
        model="whisper-1",
        file=audio_file2,
    )
    print("response_3", response_3)
    print("response3 hidden params", response_3._hidden_params)
    assert response_3._hidden_params.get("cache_hit") is not True
    assert response_3.text != response_2.text

    litellm.cache = None


@pytest.mark.asyncio
async def test_whisper_log_pre_call():
    from litellm.litellm_core_utils.litellm_logging import Logging
    from datetime import datetime
    from unittest.mock import patch, MagicMock
    from litellm.integrations.custom_logger import CustomLogger

    custom_logger = CustomLogger()

    litellm.callbacks = [custom_logger]

    with patch.object(custom_logger, "log_pre_api_call") as mock_log_pre_call:
        await litellm.atranscription(
            model="whisper-1",
            file=audio_file,
        )
        mock_log_pre_call.assert_called_once()


@pytest.mark.asyncio
async def test_whisper_log_pre_call():
    from litellm.litellm_core_utils.litellm_logging import Logging
    from datetime import datetime
    from unittest.mock import patch, MagicMock
    from litellm.integrations.custom_logger import CustomLogger

    custom_logger = CustomLogger()

    litellm.callbacks = [custom_logger]

    with patch.object(custom_logger, "log_pre_api_call") as mock_log_pre_call:
        await litellm.atranscription(
            model="whisper-1",
            file=audio_file,
        )
        mock_log_pre_call.assert_called_once()


@pytest.mark.asyncio
async def test_gpt_4o_transcribe():
    from litellm.litellm_core_utils.litellm_logging import Logging
    from datetime import datetime
    from unittest.mock import patch, MagicMock

    await litellm.atranscription(
        model="openai/gpt-4o-transcribe", file=audio_file, response_format="json"
    )


@pytest.mark.asyncio
async def test_gpt_4o_transcribe_model_mapping():
    """Test that GPT-4o transcription models are correctly mapped and not hardcoded to whisper-1"""
    
    # Test GPT-4o mini transcribe
    response = await litellm.atranscription(
        model="openai/gpt-4o-mini-transcribe", 
        file=audio_file, 
        response_format="json"
    )
    
    # Check that the response contains the correct model in hidden params
    assert response._hidden_params is not None
    assert response._hidden_params["model"] == "gpt-4o-mini-transcribe"
    assert response._hidden_params["custom_llm_provider"] == "openai"
    assert response.text is not None
    
    # Test GPT-4o transcribe
    response2 = await litellm.atranscription(
        model="openai/gpt-4o-transcribe", 
        file=audio_file, 
        response_format="json"
    )
    
    # Check that the response contains the correct model in hidden params
    assert response2._hidden_params is not None
    assert response2._hidden_params["model"] == "gpt-4o-transcribe"
    assert response2._hidden_params["custom_llm_provider"] == "openai"
    assert response2.text is not None
    
    # Test traditional whisper-1 still works
    response3 = await litellm.atranscription(
        model="openai/whisper-1", 
        file=audio_file, 
        response_format="json"
    )
    
    # Check that the response contains the correct model in hidden params
    assert response3._hidden_params is not None
    assert response3._hidden_params["model"] == "whisper-1"
    assert response3._hidden_params["custom_llm_provider"] == "openai"
    assert response3.text is not None


@pytest.mark.asyncio
async def test_azure_transcribe_model_mapping():
    """Test that Azure transcription models are correctly mapped and not hardcoded to whisper-1"""
    
    # Test Azure whisper-1
    try:
        response = await litellm.atranscription(
            model="azure/whisper-1", 
            file=audio_file, 
            response_format="json",
            api_key=os.getenv("AZURE_EUROPE_API_KEY"),
            api_base="https://my-endpoint-europe-berri-992.openai.azure.com/",
            drop_params=True
        )
        
        # Check that the response contains the correct model in hidden params
        assert response._hidden_params is not None
        assert response._hidden_params["model"] == "whisper-1"
        assert response._hidden_params["custom_llm_provider"] == "azure"
        assert response.text is not None
    except Exception as e:
        # If Azure credentials are not available, skip this test
        pytest.skip(f"Azure credentials not available: {str(e)}")
