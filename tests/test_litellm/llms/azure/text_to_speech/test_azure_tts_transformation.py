import json
from unittest.mock import Mock, patch

import httpx
import pytest

import litellm
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
    optional_params = {}
    
    mapped_voice, mapped_params = azure_tts_config.map_openai_params(
        model="azure-tts",
        optional_params=optional_params,
        voice="alloy",
        drop_params=False
    )
    
    assert mapped_voice == "en-US-JennyNeural"


def test_map_openai_params_custom_azure_voice(azure_tts_config: AzureAVATextToSpeechConfig):
    """
    Test using custom Azure voice directly
    """
    optional_params = {}
    
    mapped_voice, mapped_params = azure_tts_config.map_openai_params(
        model="azure-tts",
        optional_params=optional_params,
        voice="en-GB-RyanNeural",
        drop_params=False
    )
    
    assert mapped_voice == "en-GB-RyanNeural"


def test_map_openai_params_response_format(azure_tts_config: AzureAVATextToSpeechConfig):
    """
    Test mapping OpenAI response format to Azure output format
    """
    optional_params = {"response_format": "mp3"}
    
    mapped_voice, mapped_params = azure_tts_config.map_openai_params(
        model="azure-tts",
        optional_params=optional_params,
        drop_params=False
    )
    
    assert mapped_params["output_format"] == "audio-24khz-48kbitrate-mono-mp3"


def test_map_openai_params_default_format(azure_tts_config: AzureAVATextToSpeechConfig):
    """
    Test default output format when none specified
    """
    optional_params = {}
    
    mapped_voice, mapped_params = azure_tts_config.map_openai_params(
        model="azure-tts",
        optional_params=optional_params,
        drop_params=False
    )
    
    assert mapped_params["output_format"] == "audio-24khz-48kbitrate-mono-mp3"


def test_map_openai_params_speed(azure_tts_config: AzureAVATextToSpeechConfig):
    """
    Test mapping OpenAI speed to Azure rate
    """
    optional_params = {"speed": 1.5}
    
    mapped_voice, mapped_params = azure_tts_config.map_openai_params(
        model="azure-tts",
        optional_params=optional_params,
        drop_params=False
    )
    
    # Speed 1.5 should map to +50%
    assert mapped_params["rate"] == "+50%"


def test_map_openai_params_slow_speed(azure_tts_config: AzureAVATextToSpeechConfig):
    """
    Test mapping slow speed to Azure rate
    """
    optional_params = {"speed": 0.5}
    
    mapped_voice, mapped_params = azure_tts_config.map_openai_params(
        model="azure-tts",
        optional_params=optional_params,
        drop_params=False
    )
    
    # Speed 0.5 should map to -50%
    assert mapped_params["rate"] == "-50%"


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


# Tests for helper methods
def test_build_express_as_element_with_style(azure_tts_config: AzureAVATextToSpeechConfig):
    """
    Test _build_express_as_element helper with style only
    """
    result = azure_tts_config._build_express_as_element(
        content="<prosody rate='+0%'>Test</prosody>",
        style="cheerful"
    )
    
    assert result == "<mstts:express-as style='cheerful'><prosody rate='+0%'>Test</prosody></mstts:express-as>"


def test_build_express_as_element_with_all_attrs(azure_tts_config: AzureAVATextToSpeechConfig):
    """
    Test _build_express_as_element helper with all attributes
    """
    result = azure_tts_config._build_express_as_element(
        content="<prosody rate='+0%'>Test</prosody>",
        style="cheerful",
        styledegree="2",
        role="SeniorFemale"
    )
    
    assert "<mstts:express-as" in result
    assert "style='cheerful'" in result
    assert "styledegree='2'" in result
    assert "role='SeniorFemale'" in result
    assert "<prosody rate='+0%'>Test</prosody>" in result
    assert "</mstts:express-as>" in result


def test_build_express_as_element_no_attrs(azure_tts_config: AzureAVATextToSpeechConfig):
    """
    Test _build_express_as_element helper returns content unchanged when no attrs
    """
    content = "<prosody rate='+0%'>Test</prosody>"
    result = azure_tts_config._build_express_as_element(content=content)
    
    assert result == content
    assert "<mstts:express-as" not in result


def test_get_voice_language_with_explicit_lang(azure_tts_config: AzureAVATextToSpeechConfig):
    """
    Test _get_voice_language returns explicit language when provided
    """
    result = azure_tts_config._get_voice_language(
        voice_name="en-US-AvaMultilingualNeural",
        explicit_lang="es-ES"
    )
    
    assert result == "es-ES"


def test_get_voice_language_without_explicit_lang(azure_tts_config: AzureAVATextToSpeechConfig):
    """
    Test _get_voice_language returns None when no explicit language provided
    """
    result = azure_tts_config._get_voice_language(
        voice_name="en-US-AriaNeural",
        explicit_lang=None
    )
    
    assert result is None


def test_get_voice_language_explicit_takes_precedence(azure_tts_config: AzureAVATextToSpeechConfig):
    """
    Test that explicit language takes precedence over voice name
    """
    result = azure_tts_config._get_voice_language(
        voice_name="en-US-AvaMultilingualNeural",
        explicit_lang="fr-FR"
    )
    
    assert result == "fr-FR"


# Tests for Azure-specific SSML features
def test_map_openai_params_style(azure_tts_config: AzureAVATextToSpeechConfig):
    """
    Test passing through Azure style parameter
    """
    optional_params = {}
    
    mapped_voice, mapped_params = azure_tts_config.map_openai_params(
        model="azure-tts",
        optional_params=optional_params,
        drop_params=False,
        kwargs={"style": "cheerful"}
    )
    
    assert mapped_params["style"] == "cheerful"


def test_map_openai_params_style_and_role(azure_tts_config: AzureAVATextToSpeechConfig):
    """
    Test passing through Azure style, styledegree, and role parameters
    """
    optional_params = {}
    
    mapped_voice, mapped_params = azure_tts_config.map_openai_params(
        model="azure-tts",
        optional_params=optional_params,
        drop_params=False,
        kwargs={
            "style": "cheerful",
            "styledegree": "2",
            "role": "SeniorFemale"
        }
    )
    
    assert mapped_params["style"] == "cheerful"
    assert mapped_params["styledegree"] == "2"
    assert mapped_params["role"] == "SeniorFemale"


def test_map_openai_params_lang(azure_tts_config: AzureAVATextToSpeechConfig):
    """
    Test passing through Azure lang parameter for multilingual voices
    """
    optional_params = {}
    
    mapped_voice, mapped_params = azure_tts_config.map_openai_params(
        model="azure-tts",
        optional_params=optional_params,
        drop_params=False,
        kwargs={"lang": "es-ES"}
    )
    
    assert mapped_params["lang"] == "es-ES"


def test_transform_text_to_speech_request_with_style(azure_tts_config: AzureAVATextToSpeechConfig):
    """
    Test SSML generation with style parameter includes mstts:express-as element
    """
    result = azure_tts_config.transform_text_to_speech_request(
        model="azure-tts",
        input="Hello world",
        voice="en-US-AriaNeural",
        optional_params={
            "voice": "en-US-AriaNeural",
            "style": "cheerful"
        },
        litellm_params={},
        headers={}
    )
    
    ssml = result["ssml_body"]
    
    # Should include mstts namespace
    assert "xmlns:mstts='https://www.w3.org/2001/mstts'" in ssml
    
    # Should include mstts:express-as with style
    assert "<mstts:express-as style='cheerful'>" in ssml
    assert "</mstts:express-as>" in ssml
    
    # Should still include the content
    assert "Hello world" in ssml
    assert "en-US-AriaNeural" in ssml


def test_transform_text_to_speech_request_with_style_degree_role(azure_tts_config: AzureAVATextToSpeechConfig):
    """
    Test SSML generation with style, styledegree, and role parameters
    """
    result = azure_tts_config.transform_text_to_speech_request(
        model="azure-tts",
        input="Test message",
        voice="en-US-AriaNeural",
        optional_params={
            "voice": "en-US-AriaNeural",
            "style": "cheerful",
            "styledegree": "2",
            "role": "SeniorFemale"
        },
        litellm_params={},
        headers={}
    )
    
    ssml = result["ssml_body"]
    
    # Should include mstts namespace
    assert "xmlns:mstts='https://www.w3.org/2001/mstts'" in ssml
    
    # Should include mstts:express-as with all attributes
    assert "<mstts:express-as" in ssml
    assert "style='cheerful'" in ssml
    assert "styledegree='2'" in ssml
    assert "role='SeniorFemale'" in ssml
    assert "</mstts:express-as>" in ssml


def test_transform_text_to_speech_request_without_style(azure_tts_config: AzureAVATextToSpeechConfig):
    """
    Test that SSML without style does not include mstts namespace or express-as
    """
    result = azure_tts_config.transform_text_to_speech_request(
        model="azure-tts",
        input="Hello world",
        voice="en-US-AriaNeural",
        optional_params={"voice": "en-US-AriaNeural"},
        litellm_params={},
        headers={}
    )
    
    ssml = result["ssml_body"]
    
    # Should NOT include mstts namespace
    assert "xmlns:mstts" not in ssml
    
    # Should NOT include mstts:express-as
    assert "<mstts:express-as" not in ssml
    
    # Should still include basic SSML structure
    assert "<speak" in ssml
    assert "<voice" in ssml
    assert "en-US-AriaNeural" in ssml
    assert "Hello world" in ssml


def test_transform_text_to_speech_request_with_lang(azure_tts_config: AzureAVATextToSpeechConfig):
    """
    Test SSML generation with lang parameter for multilingual voices
    """
    result = azure_tts_config.transform_text_to_speech_request(
        model="azure-tts",
        input="Hola mundo",
        voice="en-US-AvaMultilingualNeural",
        optional_params={
            "voice": "en-US-AvaMultilingualNeural",
            "lang": "es-ES"
        },
        litellm_params={},
        headers={}
    )
    
    ssml = result["ssml_body"]
    
    # Should include xml:lang on voice element
    assert "xml:lang='es-ES'" in ssml
    
    # Should still include the content and voice
    assert "Hola mundo" in ssml
    assert "en-US-AvaMultilingualNeural" in ssml


def test_transform_text_to_speech_request_without_lang(azure_tts_config: AzureAVATextToSpeechConfig):
    """
    Test that SSML without lang parameter does not include xml:lang on voice element
    """
    result = azure_tts_config.transform_text_to_speech_request(
        model="azure-tts",
        input="Hello world",
        voice="en-US-AriaNeural",
        optional_params={"voice": "en-US-AriaNeural"},
        litellm_params={},
        headers={}
    )
    
    ssml = result["ssml_body"]
    
    # Voice element should not have xml:lang attribute (only the speak element should)
    # Check that voice element doesn't have xml:lang by ensuring the pattern doesn't exist
    assert "<voice name='en-US-AriaNeural' xml:lang=" not in ssml
    assert "<voice name='en-US-AriaNeural'>" in ssml


def test_transform_text_to_speech_request_with_raw_ssml(azure_tts_config: AzureAVATextToSpeechConfig):
    """
    Test that raw SSML input is auto-detected and passed through without transformation
    """
    raw_ssml = """<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'>
    <voice name='en-US-JennyNeural'>
        <prosody rate='fast' pitch='high'>
            This is custom SSML with specific settings!
        </prosody>
    </voice>
</speak>"""
    
    result = azure_tts_config.transform_text_to_speech_request(
        model="azure-tts",
        input=raw_ssml,
        voice="en-US-AriaNeural",
        optional_params={"voice": "en-US-AriaNeural"},
        litellm_params={},
        headers={}
    )
    
    ssml = result["ssml_body"]
    
    # The SSML should be passed through as-is
    assert ssml == raw_ssml
    assert "en-US-JennyNeural" in ssml
    assert "fast" in ssml
    assert "high" in ssml
    assert "This is custom SSML with specific settings!" in ssml
    
    # Should NOT have been wrapped or transformed
    assert ssml.count("<speak") == 1
    assert ssml.count("</speak>") == 1


def test_transform_text_to_speech_request_with_raw_ssml_header(azure_tts_config: AzureAVATextToSpeechConfig):
    """
    Test that raw SSML preserves output format headers
    """
    raw_ssml = """<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'>
    <voice name='en-US-GuyNeural'>
        Hello from raw SSML
    </voice>
</speak>"""
    
    result = azure_tts_config.transform_text_to_speech_request(
        model="azure-tts",
        input=raw_ssml,
        voice="en-US-AriaNeural",
        optional_params={
            "voice": "en-US-AriaNeural",
            "output_format": "audio-16khz-32kbitrate-mono-mp3"
        },
        litellm_params={},
        headers={}
    )
    
    # SSML should be passed through
    assert result["ssml_body"] == raw_ssml
    
    # Headers should still be set correctly
    assert result["headers"]["X-Microsoft-OutputFormat"] == "audio-16khz-32kbitrate-mono-mp3"


def test_transform_text_to_speech_request_ssml_with_mstts_namespace(azure_tts_config: AzureAVATextToSpeechConfig):
    """
    Test that raw SSML with Azure-specific mstts namespace is passed through
    """
    raw_ssml = """<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xmlns:mstts='https://www.w3.org/2001/mstts' xml:lang='en-US'>
    <voice name='en-US-AriaNeural'>
        <mstts:express-as style='cheerful' styledegree='2'>
            <prosody rate='+20%'>
                This is custom SSML with Azure-specific features!
            </prosody>
        </mstts:express-as>
    </voice>
</speak>"""
    
    result = azure_tts_config.transform_text_to_speech_request(
        model="azure-tts",
        input=raw_ssml,
        voice="en-US-AriaNeural",
        optional_params={"voice": "en-US-AriaNeural"},
        litellm_params={},
        headers={}
    )
    
    ssml = result["ssml_body"]
    
    # The SSML should be passed through as-is with all Azure-specific features
    assert ssml == raw_ssml
    assert "mstts:express-as" in ssml
    assert "style='cheerful'" in ssml
    assert "styledegree='2'" in ssml
    assert "rate='+20%'" in ssml


@patch("litellm.llms.custom_httpx.llm_http_handler.HTTPHandler.post")
def test_litellm_speech_with_ssml_passthrough(mock_post):
    """
    Test that litellm.speech passes SSML through to Azure AVA without transformation
    """
    raw_ssml = """<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'>
    <voice name='en-US-JennyNeural'>
        <prosody rate='fast' pitch='high'>
            Custom SSML content!
        </prosody>
    </voice>
</speak>"""
    
    mock_response = Mock(spec=httpx.Response)
    mock_response.content = b"fake_audio_data"
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "audio/mpeg"}
    mock_post.return_value = mock_response
    
    litellm.speech(
        model="azure/speech/tts",
        input=raw_ssml,
        voice="en-US-AriaNeural",
        api_key="test-key",
        api_base="https://eastus.api.cognitive.microsoft.com"
    )
    
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args.kwargs
    
    # Verify the SSML was sent in the request body
    assert "data" in call_kwargs
    assert call_kwargs["data"] == raw_ssml
    print("REQUEST BODY: ", json.dumps(call_kwargs["data"], indent=4))
    
    # Verify the SSML contains the original content
    assert "en-US-JennyNeural" in call_kwargs["data"]
    assert "fast" in call_kwargs["data"]
    assert "high" in call_kwargs["data"]
    assert "Custom SSML content!" in call_kwargs["data"]

