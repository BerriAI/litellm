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
from unittest.mock import AsyncMock, MagicMock, patch

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


@pytest.mark.parametrize(
    "sync_mode",
    [False, True],
)
@pytest.mark.skip(reason="local only test - we run testing using MockRequests below")
@pytest.mark.asyncio
async def test_audio_speech_litellm_vertex(sync_mode):
    litellm.set_verbose = True
    speech_file_path = Path(__file__).parent / "speech_vertex.mp3"
    model = "vertex_ai/test"
    if sync_mode:
        response = litellm.speech(
            model="vertex_ai/test",
            input="hello what llm guardrail do you have",
        )

        response.stream_to_file(speech_file_path)

    else:
        response = await litellm.aspeech(
            model="vertex_ai/",
            input="async hello what llm guardrail do you have",
        )

        from types import SimpleNamespace

        from litellm.llms.openai import HttpxBinaryResponseContent

        response.stream_to_file(speech_file_path)


@pytest.mark.asyncio
async def test_speech_litellm_vertex_async():
    # Mock the response
    mock_response = AsyncMock()

    def return_val():
        return {
            "audioContent": "dGVzdCByZXNwb25zZQ==",
        }

    mock_response.json = return_val
    mock_response.status_code = 200

    # Set up the mock for asynchronous calls
    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_async_post:
        mock_async_post.return_value = mock_response
        model = "vertex_ai/test"

        response = await litellm.aspeech(
            model=model,
            input="async hello what llm guardrail do you have",
        )

        # Assert asynchronous call
        mock_async_post.assert_called_once()
        _, kwargs = mock_async_post.call_args
        print("call args", kwargs)

        assert kwargs["url"] == "https://texttospeech.googleapis.com/v1/text:synthesize"

        assert "x-goog-user-project" in kwargs["headers"]
        assert kwargs["headers"]["Authorization"] is not None

        assert kwargs["json"] == {
            "input": {"text": "async hello what llm guardrail do you have"},
            "voice": {"languageCode": "en-US", "name": "en-US-Studio-O"},
            "audioConfig": {"audioEncoding": "LINEAR16", "speakingRate": "1"},
        }


@pytest.mark.asyncio
async def test_speech_litellm_vertex_async_with_voice():
    # Mock the response
    mock_response = AsyncMock()

    def return_val():
        return {
            "audioContent": "dGVzdCByZXNwb25zZQ==",
        }

    mock_response.json = return_val
    mock_response.status_code = 200

    # Set up the mock for asynchronous calls
    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_async_post:
        mock_async_post.return_value = mock_response
        model = "vertex_ai/test"

        response = await litellm.aspeech(
            model=model,
            input="async hello what llm guardrail do you have",
            voice={
                "languageCode": "en-UK",
                "name": "en-UK-Studio-O",
            },
            audioConfig={
                "audioEncoding": "LINEAR22",
                "speakingRate": "10",
            },
        )

        # Assert asynchronous call
        mock_async_post.assert_called_once()
        _, kwargs = mock_async_post.call_args
        print("call args", kwargs)

        assert kwargs["url"] == "https://texttospeech.googleapis.com/v1/text:synthesize"

        assert "x-goog-user-project" in kwargs["headers"]
        assert kwargs["headers"]["Authorization"] is not None

        assert kwargs["json"] == {
            "input": {"text": "async hello what llm guardrail do you have"},
            "voice": {"languageCode": "en-UK", "name": "en-UK-Studio-O"},
            "audioConfig": {"audioEncoding": "LINEAR22", "speakingRate": "10"},
        }
