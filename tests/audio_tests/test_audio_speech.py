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
            "azure/tts",
            os.getenv("AZURE_SWEDEN_API_KEY"),
            os.getenv("AZURE_SWEDEN_API_BASE"),
        ),
        ("openai/tts-1", os.getenv("OPENAI_API_KEY"), None),
    ],
)  # ,
@pytest.mark.asyncio
@pytest.mark.flaky(retries=3, delay=1)
async def test_audio_speech_litellm(sync_mode, model, api_base, api_key):
    litellm._turn_on_debug()
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
    api_base = os.getenv("AZURE_TTS_API_BASE")
    

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


@pytest.mark.asyncio
@pytest.mark.flaky(retries=3, delay=1)
@pytest.mark.skip(reason="RunwayML TTS API only tested locally")
async def test_runwayml_tts_async():
    """
    Test RunwayML Text-to-Speech with real API request.
    """
    litellm._turn_on_debug()
    api_key = os.getenv("RUNWAYML_API_KEY")
    api_base = os.getenv("RUNWAYML_API_BASE")
    

    speech_file_path = Path(__file__).parent / "runwayml_speech.mp3"
    
    try:
        response = await litellm.aspeech(
            model="runwayml/eleven_multilingual_v2",
            voice="Rachel",
            input="Yuneng is gone, we miss him so much I hope he has a good coffee",
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
        
        print(f"RunwayML TTS audio saved to: {speech_file_path}")

        # assert response cost is greater than 0
        print("Response cost: ", response._hidden_params["response_cost"])
        assert response._hidden_params["response_cost"] > 0
    
    except Exception as e:
        pytest.fail(f"Test failed with exception: {str(e)}")


@pytest.mark.asyncio
async def test_azure_ava_tts_with_custom_voice():
    """
    Test that when using a custom Azure voice (en-US-AndrewNeural),
    the SSML request body contains the selected voice.
    """
    from unittest.mock import AsyncMock, MagicMock, patch
    import httpx
    
    # Mock response
    mock_response_content = b"fake_audio_data"
    mock_httpx_response = MagicMock(spec=httpx.Response)
    mock_httpx_response.content = mock_response_content
    mock_httpx_response.status_code = 200
    mock_httpx_response.headers = {"content-type": "audio/mpeg"}
    
    with patch("litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post") as mock_post:
        mock_post.return_value = mock_httpx_response
        
        response = await litellm.aspeech(
            model="azure/speech/azure-tts",
            voice="en-US-AndrewNeural",
            input="Hello, this is a test",
            api_base="https://eastus.tts.speech.microsoft.com",
            api_key="fake-key",
            response_format="mp3",
        )
        
        # Verify the mock was called
        assert mock_post.called
        
        # Get the call arguments
        call_args = mock_post.call_args
        ssml_body = call_args.kwargs.get("data")
        
        # Verify the SSML contains the custom voice
        assert ssml_body is not None
        assert "en-US-AndrewNeural" in ssml_body
        assert "Hello, this is a test" in ssml_body
        assert "<speak" in ssml_body
        assert "<voice" in ssml_body


@pytest.mark.asyncio
async def test_azure_ava_tts_fable_voice_mapping():
    """
    Test that when using OpenAI voice 'fable',
    it gets mapped to Azure voice 'en-GB-RyanNeural' in the SSML.
    """
    from unittest.mock import AsyncMock, MagicMock, patch
    import httpx
    
    # Mock response
    mock_response_content = b"fake_audio_data"
    mock_httpx_response = MagicMock(spec=httpx.Response)
    mock_httpx_response.content = mock_response_content
    mock_httpx_response.status_code = 200
    mock_httpx_response.headers = {"content-type": "audio/mpeg"}
    
    with patch("litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post") as mock_post:
        mock_post.return_value = mock_httpx_response
        
        response = await litellm.aspeech(
            model="azure/speech/azure-tts",
            voice="fable",
            input="Testing voice mapping",
            api_base="https://eastus.tts.speech.microsoft.com",
            api_key="fake-key",
            response_format="mp3",
        )
        
        # Verify the mock was called
        assert mock_post.called
        
        # Get the call arguments
        call_args = mock_post.call_args
        ssml_body = call_args.kwargs.get("data")
        
        # Verify the SSML contains the mapped voice (en-GB-RyanNeural, not 'fable')
        assert ssml_body is not None
        assert "en-GB-RyanNeural" in ssml_body
        assert "fable" not in ssml_body.lower()
        assert "Testing voice mapping" in ssml_body
        assert "<speak" in ssml_body
        assert "<voice" in ssml_body


@pytest.mark.asyncio
async def test_aws_polly_tts_with_native_voice():
    """
    Test AWS Polly TTS with a native Polly voice (Joanna).
    Verifies the request is formatted correctly for the Polly API.
    """
    import json
    from unittest.mock import MagicMock, patch
    import httpx

    # Mock response - Polly returns audio bytes directly
    mock_response_content = b"fake_audio_data"
    mock_httpx_response = MagicMock(spec=httpx.Response)
    mock_httpx_response.content = mock_response_content
    mock_httpx_response.status_code = 200
    mock_httpx_response.headers = {"content-type": "audio/mpeg"}

    with patch("litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post") as mock_post:
        mock_post.return_value = mock_httpx_response

        response = await litellm.aspeech(
            model="aws_polly/neural",
            voice="Joanna",
            input="Hello, this is a test of AWS Polly",
            aws_region_name="us-east-1",
        )

        # Verify the mock was called
        assert mock_post.called

        # Get the call arguments - AWS Polly uses data= with JSON string (for SigV4 signing)
        call_args = mock_post.call_args
        request_data = call_args.kwargs.get("data")

        # Parse the JSON body
        assert request_data is not None
        request_body = json.loads(request_data)

        # Verify the request body is formatted correctly for Polly
        assert request_body["VoiceId"] == "Joanna"
        assert request_body["Text"] == "Hello, this is a test of AWS Polly"
        assert request_body["OutputFormat"] == "mp3"
        assert request_body["Engine"] == "neural"
        assert request_body.get("TextType", "text") == "text"


@pytest.mark.asyncio
async def test_aws_polly_tts_with_openai_voice_mapping():
    """
    Test AWS Polly TTS with OpenAI voice mapping (alloy -> Joanna).
    Verifies that OpenAI voices are correctly mapped to Polly voices.
    """
    import json
    from unittest.mock import MagicMock, patch
    import httpx

    mock_response_content = b"fake_audio_data"
    mock_httpx_response = MagicMock(spec=httpx.Response)
    mock_httpx_response.content = mock_response_content
    mock_httpx_response.status_code = 200
    mock_httpx_response.headers = {"content-type": "audio/mpeg"}

    with patch("litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post") as mock_post:
        mock_post.return_value = mock_httpx_response

        response = await litellm.aspeech(
            model="aws_polly/neural",
            voice="alloy",
            input="Testing OpenAI voice mapping",
            aws_region_name="us-east-1",
        )

        assert mock_post.called

        call_args = mock_post.call_args
        request_data = call_args.kwargs.get("data")

        # Parse the JSON body
        assert request_data is not None
        request_body = json.loads(request_data)

        # Verify alloy was mapped to Joanna
        assert request_body["VoiceId"] == "Joanna"
        assert request_body["Text"] == "Testing OpenAI voice mapping"


@pytest.mark.asyncio
async def test_aws_polly_tts_with_ssml():
    """
    Test AWS Polly TTS with SSML input.
    Verifies that SSML is detected and TextType is set correctly.
    """
    import json
    from unittest.mock import MagicMock, patch
    import httpx

    mock_response_content = b"fake_audio_data"
    mock_httpx_response = MagicMock(spec=httpx.Response)
    mock_httpx_response.content = mock_response_content
    mock_httpx_response.status_code = 200
    mock_httpx_response.headers = {"content-type": "audio/mpeg"}

    ssml_input = '<speak>Hello, <break time="500ms"/> this is SSML.</speak>'

    with patch("litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post") as mock_post:
        mock_post.return_value = mock_httpx_response

        response = await litellm.aspeech(
            model="aws_polly/neural",
            voice="Joanna",
            input=ssml_input,
            aws_region_name="us-east-1",
        )

        assert mock_post.called

        call_args = mock_post.call_args
        request_data = call_args.kwargs.get("data")

        # Parse the JSON body
        assert request_data is not None
        request_body = json.loads(request_data)

        # Verify SSML is detected and TextType is set to ssml
        assert request_body["Text"] == ssml_input
        assert request_body["TextType"] == "ssml"
        assert request_body["VoiceId"] == "Joanna"


@pytest.mark.asyncio
async def test_aws_polly_tts_real_api():
    """
    Test AWS Polly TTS with real API request.
    Requires AWS credentials to be configured.
    """
    speech_file_path = Path(__file__).parent / "aws_polly_speech_generative.mp3"

    response = await litellm.aspeech(
        model="aws_polly/generative",
        voice="Joanna",
        input="Hello, this is a test of AWS Polly text to speech integration with LiteLLM.",
        aws_region_name="us-east-1",
    )

    from litellm.types.llms.openai import HttpxBinaryResponseContent

    assert isinstance(response, HttpxBinaryResponseContent)

    binary_content = response.content
    assert len(binary_content) > 0

    # MP3 files start with ID3 tag or MPEG sync word
    assert binary_content[:3] == b"ID3" or binary_content[:2] == b"\xff\xfb" or binary_content[:2] == b"\xff\xf3"

    response.stream_to_file(speech_file_path)

    assert speech_file_path.exists()
    assert speech_file_path.stat().st_size > 0

    print(f"AWS Polly TTS audio saved to: {speech_file_path}")
