from unittest.mock import Mock

import httpx
import pytest

from litellm.llms.azure.text_to_speech.transformation import AzureAVATextToSpeechConfig


@pytest.fixture
def azure_tts_config() -> AzureAVATextToSpeechConfig:
    """
    Fixture for AzureAVATextToSpeechConfig instance
    """
    return AzureAVATextToSpeechConfig()


# Tests for map_openai_params
def test_map_openai_params_voice_mapping(azure_tts_config: AzureAVATextToSpeechConfig):
    """
    Test mapping OpenAI voice to Azure AVA voice
    """
    optional_params = {"voice": "alloy"}
    
    mapped = azure_tts_config.map_openai_params(
        model="azure-tts",
        optional_params=optional_params,
        drop_params=False
    )
    
    assert mapped["voice"] == "en-US-JennyNeural"


def test_map_openai_params_custom_azure_voice(azure_tts_config: AzureAVATextToSpeechConfig):
    """
    Test using custom Azure voice directly
    """
    optional_params = {"voice": "en-GB-RyanNeural"}
    
    mapped = azure_tts_config.map_openai_params(
        model="azure-tts",
        optional_params=optional_params,
        drop_params=False
    )
    
    assert mapped["voice"] == "en-GB-RyanNeural"


def test_map_openai_params_response_format(azure_tts_config: AzureAVATextToSpeechConfig):
    """
    Test mapping OpenAI response format to Azure output format
    """
    optional_params = {"response_format": "mp3"}
    
    mapped = azure_tts_config.map_openai_params(
        model="azure-tts",
        optional_params=optional_params,
        drop_params=False
    )
    
    assert mapped["output_format"] == "audio-24khz-48kbitrate-mono-mp3"


def test_map_openai_params_default_format(azure_tts_config: AzureAVATextToSpeechConfig):
    """
    Test default output format when none specified
    """
    optional_params = {}
    
    mapped = azure_tts_config.map_openai_params(
        model="azure-tts",
        optional_params=optional_params,
        drop_params=False
    )
    
    assert mapped["output_format"] == "audio-24khz-48kbitrate-mono-mp3"


def test_map_openai_params_speed(azure_tts_config: AzureAVATextToSpeechConfig):
    """
    Test mapping OpenAI speed to Azure rate
    """
    optional_params = {"speed": 1.5}
    
    mapped = azure_tts_config.map_openai_params(
        model="azure-tts",
        optional_params=optional_params,
        drop_params=False
    )
    
    # Speed 1.5 should map to +50%
    assert mapped["rate"] == "+50%"


def test_map_openai_params_slow_speed(azure_tts_config: AzureAVATextToSpeechConfig):
    """
    Test mapping slow speed to Azure rate
    """
    optional_params = {"speed": 0.5}
    
    mapped = azure_tts_config.map_openai_params(
        model="azure-tts",
        optional_params=optional_params,
        drop_params=False
    )
    
    # Speed 0.5 should map to -50%
    assert mapped["rate"] == "-50%"


# Tests for get_complete_url
def test_get_complete_url_cognitive_services(azure_tts_config: AzureAVATextToSpeechConfig):
    """
    Test converting Cognitive Services endpoint to TTS endpoint
    """
    api_base = "https://eastus.api.cognitive.microsoft.com"
    
    url = azure_tts_config.get_complete_url(
        model="azure-tts",
        api_base=api_base,
        litellm_params={}
    )
    
    assert url == "https://eastus.tts.speech.microsoft.com/cognitiveservices/v1"


def test_get_complete_url_tts_endpoint(azure_tts_config: AzureAVATextToSpeechConfig):
    """
    Test using TTS endpoint directly
    """
    api_base = "https://westus.tts.speech.microsoft.com"
    
    url = azure_tts_config.get_complete_url(
        model="azure-tts",
        api_base=api_base,
        litellm_params={}
    )
    
    assert url == "https://westus.tts.speech.microsoft.com/cognitiveservices/v1"


def test_get_complete_url_tts_endpoint_with_path(azure_tts_config: AzureAVATextToSpeechConfig):
    """
    Test TTS endpoint that already has the path
    """
    api_base = "https://westus.tts.speech.microsoft.com/cognitiveservices/v1"
    
    url = azure_tts_config.get_complete_url(
        model="azure-tts",
        api_base=api_base,
        litellm_params={}
    )
    
    assert url == "https://westus.tts.speech.microsoft.com/cognitiveservices/v1"


def test_get_complete_url_custom_endpoint(azure_tts_config: AzureAVATextToSpeechConfig):
    """
    Test custom endpoint URL
    """
    api_base = "https://custom.domain.com"
    
    url = azure_tts_config.get_complete_url(
        model="azure-tts",
        api_base=api_base,
        litellm_params={}
    )
    
    assert url == "https://custom.domain.com/cognitiveservices/v1"


def test_get_complete_url_missing_api_base(azure_tts_config: AzureAVATextToSpeechConfig):
    """
    Test error when api_base is missing
    """
    with pytest.raises(ValueError, match="api_base is required"):
        azure_tts_config.get_complete_url(
            model="azure-tts",
            api_base=None,
            litellm_params={}
        )


# Tests for transform_text_to_speech_request
def test_transform_text_to_speech_request_basic(azure_tts_config: AzureAVATextToSpeechConfig):
    """
    Test basic TTS request transformation
    """
    result = azure_tts_config.transform_text_to_speech_request(
        model="azure-tts",
        input="Hello world",
        voice="en-US-AriaNeural",
        optional_params={"voice": "en-US-AriaNeural"},
        litellm_params={},
        headers={}
    )
    
    assert "ssml_body" in result
    assert "Hello world" in result["ssml_body"]
    assert "en-US-AriaNeural" in result["ssml_body"]
    assert "<speak" in result["ssml_body"]
    assert "<voice" in result["ssml_body"]
    assert "<prosody" in result["ssml_body"]


def test_transform_text_to_speech_request_with_rate(azure_tts_config: AzureAVATextToSpeechConfig):
    """
    Test TTS request with custom rate
    """
    result = azure_tts_config.transform_text_to_speech_request(
        model="azure-tts",
        input="Test message",
        voice="en-US-AriaNeural",
        optional_params={"voice": "en-US-AriaNeural", "rate": "+50%"},
        litellm_params={},
        headers={}
    )
    
    assert "+50%" in result["ssml_body"]


def test_transform_text_to_speech_request_xml_escaping(azure_tts_config: AzureAVATextToSpeechConfig):
    """
    Test XML special characters are properly escaped
    """
    input_text = "Test <tag> & 'quotes' \"double\""
    
    result = azure_tts_config.transform_text_to_speech_request(
        model="azure-tts",
        input=input_text,
        voice="en-US-AriaNeural",
        optional_params={"voice": "en-US-AriaNeural"},
        litellm_params={},
        headers={}
    )
    
    ssml = result["ssml_body"]
    assert "&lt;tag&gt;" in ssml
    assert "&amp;" in ssml
    assert "&apos;" in ssml
    assert "&quot;" in ssml


def test_transform_text_to_speech_request_headers(azure_tts_config: AzureAVATextToSpeechConfig):
    """
    Test that output format is added to headers
    """
    result = azure_tts_config.transform_text_to_speech_request(
        model="azure-tts",
        input="Test",
        voice="en-US-AriaNeural",
        optional_params={
            "voice": "en-US-AriaNeural",
            "output_format": "audio-16khz-32kbitrate-mono-mp3"
        },
        litellm_params={},
        headers={}
    )
    
    assert result["headers"]["X-Microsoft-OutputFormat"] == "audio-16khz-32kbitrate-mono-mp3"


# Tests for transform_text_to_speech_response
def test_transform_text_to_speech_response(azure_tts_config: AzureAVATextToSpeechConfig):
    """
    Test TTS response transformation
    """
    # Create a mock response
    mock_response = Mock(spec=httpx.Response)
    mock_response.content = b"fake_audio_data"
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "audio/mpeg"}
    
    mock_logging = Mock()
    
    result = azure_tts_config.transform_text_to_speech_response(
        model="azure-tts",
        raw_response=mock_response,
        logging_obj=mock_logging
    )
    
    # Should return HttpxBinaryResponseContent wrapper
    from litellm.types.llms.openai import HttpxBinaryResponseContent
    assert isinstance(result, HttpxBinaryResponseContent)

