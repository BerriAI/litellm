import litellm
import pytest
from unittest.mock import MagicMock


@pytest.mark.asyncio
async def test_azure_ava_tts_with_custom_voice():
    """
    Test that when using a custom Azure voice (en-US-AndrewNeural),
    the SSML request body contains the selected voice.
    """
    from unittest.mock import patch

    import httpx

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

        assert mock_post.called

        call_args = mock_post.call_args
        ssml_body = call_args.kwargs.get("data")

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
    from unittest.mock import patch

    import httpx

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

        assert mock_post.called

        call_args = mock_post.call_args
        ssml_body = call_args.kwargs.get("data")

        assert ssml_body is not None
        assert "en-GB-RyanNeural" in ssml_body
        assert "fable" not in ssml_body.lower()
        assert "Testing voice mapping" in ssml_body
        assert "<speak" in ssml_body
        assert "<voice" in ssml_body


@pytest.mark.asyncio
@pytest.mark.usefixtures("fake_provider_credentials")
async def test_aws_polly_tts_with_native_voice():
    """
    Test AWS Polly TTS with a native Polly voice (Joanna).
    Verifies the request is formatted correctly for the Polly API.
    """
    import json
    from unittest.mock import patch

    import httpx

    # Mock response - Polly returns audio bytes directly
    mock_response_content = b"fake_audio_data"
    mock_httpx_response = MagicMock(spec=httpx.Response)
    mock_httpx_response.content = mock_response_content
    mock_httpx_response.status_code = 200
    mock_httpx_response.headers = {"content-type": "audio/mpeg"}

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post"
    ) as mock_post:
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
@pytest.mark.usefixtures("fake_provider_credentials")
async def test_aws_polly_tts_with_openai_voice_mapping():
    """
    Test AWS Polly TTS with OpenAI voice mapping (alloy -> Joanna).
    Verifies that OpenAI voices are correctly mapped to Polly voices.
    """
    import json
    from unittest.mock import patch

    import httpx

    mock_response_content = b"fake_audio_data"
    mock_httpx_response = MagicMock(spec=httpx.Response)
    mock_httpx_response.content = mock_response_content
    mock_httpx_response.status_code = 200
    mock_httpx_response.headers = {"content-type": "audio/mpeg"}

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post"
    ) as mock_post:
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
@pytest.mark.usefixtures("fake_provider_credentials")
async def test_aws_polly_tts_with_ssml():
    """
    Test AWS Polly TTS with SSML input.
    Verifies that SSML is detected and TextType is set correctly.
    """
    import json
    from unittest.mock import patch

    import httpx

    mock_response_content = b"fake_audio_data"
    mock_httpx_response = MagicMock(spec=httpx.Response)
    mock_httpx_response.content = mock_response_content
    mock_httpx_response.status_code = 200
    mock_httpx_response.headers = {"content-type": "audio/mpeg"}

    ssml_input = '<speak>Hello, <break time="500ms"/> this is SSML.</speak>'

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post"
    ) as mock_post:
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
