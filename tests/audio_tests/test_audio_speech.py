# What is this?
## unit tests for openai tts endpoint

import asyncio
import os
import random
import sys
import time
import traceback
from litellm._uuid import uuid

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
@pytest.mark.flaky(retries=3, delay=1)
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

        from litellm.types.llms.openai import HttpxBinaryResponseContent

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

        from litellm.llms.openai.openai import HttpxBinaryResponseContent

        assert isinstance(response, HttpxBinaryResponseContent)


@pytest.mark.parametrize(
    "sync_mode",
    [False, True],
)
@pytest.mark.skip(reason="local only test - we run testing using MockRequests below")
@pytest.mark.asyncio
@pytest.mark.flaky(retries=3, delay=1)
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

        from litellm.llms.openai.openai import HttpxBinaryResponseContent

        response.stream_to_file(speech_file_path)


@pytest.mark.flaky(retries=6, delay=2)
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

        try:
            response = await litellm.aspeech(
                model=model,
                input="async hello what llm guardrail do you have",
            )
        except litellm.APIConnectionError as e:
            if "Your default credentials were not found" in str(e):
                pytest.skip("skipping test, credentials not found")

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

        try:
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
        except litellm.APIConnectionError as e:
            if "Your default credentials were not found" in str(e):
                pytest.skip("skipping test, credentials not found")

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


@pytest.mark.asyncio
async def test_speech_litellm_vertex_async_with_voice_ssml():
    # Mock the response
    mock_response = AsyncMock()

    def return_val():
        return {
            "audioContent": "dGVzdCByZXNwb25zZQ==",
        }

    mock_response.json = return_val
    mock_response.status_code = 200

    ssml = """
    <speak>
        <p>Hello, world!</p>
        <p>This is a test of the <break strength="medium" /> text-to-speech API.</p>
    </speak>
    """

    # Set up the mock for asynchronous calls
    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_async_post:
        mock_async_post.return_value = mock_response
        model = "vertex_ai/test"

        try:
            response = await litellm.aspeech(
                input=ssml,
                model=model,
                voice={
                    "languageCode": "en-UK",
                    "name": "en-UK-Studio-O",
                },
                audioConfig={
                    "audioEncoding": "LINEAR22",
                    "speakingRate": "10",
                },
            )
        except litellm.APIConnectionError as e:
            if "Your default credentials were not found" in str(e):
                pytest.skip("skipping test, credentials not found")

        # Assert asynchronous call
        mock_async_post.assert_called_once()
        _, kwargs = mock_async_post.call_args
        print("call args", kwargs)

        assert kwargs["url"] == "https://texttospeech.googleapis.com/v1/text:synthesize"

        assert "x-goog-user-project" in kwargs["headers"]
        assert kwargs["headers"]["Authorization"] is not None

        assert kwargs["json"] == {
            "input": {"ssml": ssml},
            "voice": {"languageCode": "en-UK", "name": "en-UK-Studio-O"},
            "audioConfig": {"audioEncoding": "LINEAR22", "speakingRate": "10"},
        }


@pytest.mark.skip(reason="causes openai rate limit errors")
def test_audio_speech_cost_calc():
    from litellm.integrations.custom_logger import CustomLogger

    model = "azure/azure-tts"
    api_base = os.getenv("AZURE_SWEDEN_API_BASE")
    api_key = os.getenv("AZURE_SWEDEN_API_KEY")

    custom_logger = CustomLogger()
    litellm.set_verbose = True

    with patch.object(custom_logger, "log_success_event") as mock_cost_calc:
        litellm.callbacks = [custom_logger]
        litellm.speech(
            model=model,
            voice="alloy",
            input="the quick brown fox jumped over the lazy dogs",
            api_base=api_base,
            api_key=api_key,
            base_model="azure/tts-1",
        )

        time.sleep(1)

        mock_cost_calc.assert_called_once()

        print(
            f"mock_cost_calc.call_args: {mock_cost_calc.call_args.kwargs['kwargs'].keys()}"
        )
        standard_logging_payload = mock_cost_calc.call_args.kwargs["kwargs"][
            "standard_logging_object"
        ]
        print(f"standard_logging_payload: {standard_logging_payload}")
        assert standard_logging_payload["response_cost"] > 0


def test_audio_speech_gemini():
    result = litellm.speech(
        model="gemini/gemini-2.5-flash-preview-tts",
        input="the quick brown fox jumped over the lazy dogs",
        api_key=os.getenv("GEMINI_API_KEY"),
    )

    print(result)


@pytest.mark.asyncio
@pytest.mark.flaky(retries=3, delay=1)
async def test_azure_ava_tts_async():
    """
    Test Azure AVA (Cognitive Services) Text-to-Speech with real API request.
    """
    litellm._turn_on_debug()
    api_key = os.getenv("AZURE_TTS_API_KEY")
    api_base = "https://eastus.tts.speech.microsoft.com"
    

    speech_file_path = Path(__file__).parent / "azure_speech.mp3"
    
    try:
        response = await litellm.aspeech(
            model="azure/speech/azure-tts",
            voice="alloy",
            input="Hello, this is a test of Azure text to speech",
            api_base=api_base,
            api_key=api_key,
            response_format="mp3",
            speed=1.0,
        )

        # Assert the response is HttpxBinaryResponseContent
        from litellm.types.llms.openai import HttpxBinaryResponseContent
        
        assert isinstance(response, HttpxBinaryResponseContent)
        
        # Get the binary content
        binary_content = response.content
        assert len(binary_content) > 0
        
        # MP3 files start with these magic bytes
        # ID3 tag or MPEG sync word
        assert binary_content[:3] == b"ID3" or binary_content[:2] == b"\xff\xfb" or binary_content[:2] == b"\xff\xf3"
        
        # Write to file
        response.stream_to_file(speech_file_path)
        
        # Verify file was created and has content
        assert speech_file_path.exists()
        assert speech_file_path.stat().st_size > 0
        
        print(f"Azure TTS audio saved to: {speech_file_path}")

        # assert response cost is greater than 0
        print("Response cost: ", response._hidden_params["response_cost"])
        assert response._hidden_params["response_cost"] > 0

    except Exception as e:
        pytest.fail(f"Test failed with exception: {str(e)}")
