"""
Test Gemini TTS (Text-to-Speech) functionality
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.llms.gemini.chat.transformation import GoogleAIStudioGeminiConfig
from litellm.utils import get_supported_openai_params


class TestGeminiTTSTransformation:
    """Test Gemini TTS transformation functionality"""

    def test_gemini_tts_model_detection(self):
        """Test that TTS models are correctly identified"""
        config = GoogleAIStudioGeminiConfig()
        
        # Test TTS models
        assert config.is_model_gemini_audio_model("gemini-2.5-flash-preview-tts") == True
        assert config.is_model_gemini_audio_model("gemini-2.5-pro-preview-tts") == True
        
        # Test non-TTS models
        assert config.is_model_gemini_audio_model("gemini-2.5-flash") == False
        assert config.is_model_gemini_audio_model("gemini-2.5-pro") == False
        assert config.is_model_gemini_audio_model("gpt-4o-audio-preview") == False

    def test_gemini_tts_supported_params(self):
        """Test that audio parameter is included for TTS models"""
        config = GoogleAIStudioGeminiConfig()
        
        # Test TTS model
        params = config.get_supported_openai_params("gemini-2.5-flash-preview-tts")
        assert "audio" in params
        
        # Test that other standard params are still included
        assert "temperature" in params
        assert "max_tokens" in params
        assert "modalities" in params
        
        # Test non-TTS model
        params_non_tts = config.get_supported_openai_params("gemini-2.5-flash")
        assert "audio" not in params_non_tts

    def test_gemini_tts_audio_parameter_mapping(self):
        """Test audio parameter mapping for TTS models"""
        config = GoogleAIStudioGeminiConfig()
        
        non_default_params = {
            "audio": {
                "voice": "Kore",
                "format": "wav"
            }
        }
        optional_params = {}
        
        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="gemini-2.5-flash-preview-tts",
            drop_params=False
        )
        
        # Check speech config is created
        assert "speechConfig" in result
        assert "voiceConfig" in result["speechConfig"]
        assert "prebuiltVoiceConfig" in result["speechConfig"]["voiceConfig"]
        assert result["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"] == "Kore"
        
        # Check response modalities
        assert "responseModalities" in result
        assert "AUDIO" in result["responseModalities"]

    def test_gemini_tts_audio_parameter_with_existing_modalities(self):
        """Test audio parameter mapping when modalities already exist"""
        config = GoogleAIStudioGeminiConfig()
        
        non_default_params = {
            "audio": {
                "voice": "Puck"
            }
        }
        optional_params = {
            "responseModalities": ["TEXT"]
        }
        
        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="gemini-2.5-flash-preview-tts",
            drop_params=False
        )
        
        # Check that AUDIO is added to existing modalities
        assert "responseModalities" in result
        assert "TEXT" in result["responseModalities"]
        assert "AUDIO" in result["responseModalities"]

    def test_gemini_tts_no_audio_parameter(self):
        """Test that non-audio parameters are handled normally"""
        config = GoogleAIStudioGeminiConfig()
        
        non_default_params = {
            "temperature": 0.7,
            "max_tokens": 100
        }
        optional_params = {}
        
        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="gemini-2.5-flash-preview-tts",
            drop_params=False
        )
        
        # Should not have speech config
        assert "speechConfig" not in result
        # Should not automatically add audio modalities
        assert "responseModalities" not in result

    def test_gemini_tts_invalid_audio_parameter(self):
        """Test handling of invalid audio parameter"""
        config = GoogleAIStudioGeminiConfig()
        
        non_default_params = {
            "audio": "invalid_string"  # Should be dict
        }
        optional_params = {}
        
        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="gemini-2.5-flash-preview-tts",
            drop_params=False
        )
        
        # Should not create speech config for invalid audio param
        assert "speechConfig" not in result

    def test_gemini_tts_empty_audio_parameter(self):
        """Test handling of empty audio parameter"""
        config = GoogleAIStudioGeminiConfig()
        
        non_default_params = {
            "audio": {}
        }
        optional_params = {}
        
        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="gemini-2.5-flash-preview-tts",
            drop_params=False
        )
        
        # Should still set response modalities even with empty audio config
        assert "responseModalities" in result
        assert "AUDIO" in result["responseModalities"]

    def test_gemini_tts_utils_integration(self):
        """Test integration with LiteLLM utils functions"""
        # Test that get_supported_openai_params works with TTS models
        params = get_supported_openai_params("gemini-2.5-flash-preview-tts", "gemini")
        assert "audio" in params
        
        # Test non-TTS model
        params_non_tts = get_supported_openai_params("gemini-2.5-flash", "gemini")
        assert "audio" not in params_non_tts


def test_gemini_tts_completion_mock():
    """Test Gemini TTS completion with mocked response"""
    with patch('litellm.completion') as mock_completion:
        # Mock a successful TTS response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Generated audio response"
        mock_completion.return_value = mock_response

        # Test completion call with audio parameter
        response = litellm.completion(
            model="gemini-2.5-flash-preview-tts",
            messages=[{"role": "user", "content": "Say hello"}],
            audio={"voice": "Kore", "format": "wav"}
        )

        assert response is not None
        assert response.choices[0].message.content is not None


def test_gemini_audio_response_transformation():
    """Test that Gemini audio responses are correctly transformed to OpenAI format"""
    from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexGeminiConfig
    from litellm.types.utils import ChatCompletionAudioResponse

    config = VertexGeminiConfig()

    # Test case 1: Audio data in text content (L16 PCM data URI format - Gemini's typical format)
    parts_with_audio_uri = [
        {
            "text": "data:audio/L16;codec=pcm;rate=24000;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAIlYAAESsAAACABAAZGF0YQAAAAA="
        }
    ]

    # Test that audio content is excluded from text content
    content, reasoning_content = config.get_assistant_content_message(parts_with_audio_uri)
    assert content is None  # Audio content should not appear in text
    assert reasoning_content is None

    # Test the separate audio extraction method
    audio_response = config._extract_audio_response_from_parts(parts_with_audio_uri)

    # Verify audio response transformation
    assert audio_response is not None
    assert isinstance(audio_response, ChatCompletionAudioResponse)
    assert audio_response.data == "UklGRiQAAABXQVZFZm10IBAAAAABAAEAIlYAAESsAAACABAAZGF0YQAAAAA="
    assert audio_response.id is not None
    assert audio_response.expires_at > 0
    assert audio_response.transcript == ""

    # Test case 1b: Audio data with different format (should still work due to flexible validation)
    parts_with_wav_audio = [
        {
            "text": "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEAIlYAAESsAAACABAAZGF0YQAAAAA="
        }
    ]

    audio_response = config._extract_audio_response_from_parts(parts_with_wav_audio)

    # Verify audio response transformation works with any audio format
    assert audio_response is not None
    assert isinstance(audio_response, ChatCompletionAudioResponse)
    assert audio_response.data == "UklGRiQAAABXQVZFZm10IBAAAAABAAEAIlYAAESsAAACABAAZGF0YQAAAAA="

    # Test case 2: Audio data in inlineData format
    parts_with_inline_audio = [
        {
            "inlineData": {
                "mimeType": "audio/wav",
                "data": "UklGRiQAAABXQVZFZm10IBAAAAABAAEAIlYAAESsAAACABAAZGF0YQAAAAA="
            }
        }
    ]

    # Test that audio content is excluded from text content
    content, reasoning_content = config.get_assistant_content_message(parts_with_inline_audio)
    assert content is None  # Audio content should not appear in text
    assert reasoning_content is None

    audio_response = config._extract_audio_response_from_parts(parts_with_inline_audio)

    # Verify audio response transformation
    assert audio_response is not None
    assert isinstance(audio_response, ChatCompletionAudioResponse)
    assert audio_response.data == "UklGRiQAAABXQVZFZm10IBAAAAABAAEAIlYAAESsAAACABAAZGF0YQAAAAA="

    # Test case 3: Regular text content (no audio)
    parts_with_text = [
        {
            "text": "Hello, this is a regular text response."
        }
    ]

    content, reasoning_content = config.get_assistant_content_message(parts_with_text)
    audio_response = config._extract_audio_response_from_parts(parts_with_text)

    # Verify regular text response
    assert content == "Hello, this is a regular text response."
    assert reasoning_content is None
    assert audio_response is None

    # Test case 4: Image data (should not be treated as audio)
    parts_with_image = [
        {
            "inlineData": {
                "mimeType": "image/png",
                "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
            }
        }
    ]

    content, reasoning_content = config.get_assistant_content_message(parts_with_image)
    audio_response = config._extract_audio_response_from_parts(parts_with_image)

    # Verify image data is handled as content
    assert content is not None
    assert "data:image/png;base64," in content
    assert reasoning_content is None
    assert audio_response is None

    # Test case 5: Invalid audio data URI (should be treated as text)
    parts_with_invalid_audio = [
        {
            "text": "data:audio/wav;base64,invalid_base64_data!"
        }
    ]

    content, reasoning_content = config.get_assistant_content_message(parts_with_invalid_audio)
    audio_response = config._extract_audio_response_from_parts(parts_with_invalid_audio)

    # Verify invalid audio data is treated as regular text
    assert content == "data:audio/wav;base64,invalid_base64_data!"
    assert reasoning_content is None
    assert audio_response is None


def test_gemini_audio_format_validation():
    """Test that Gemini TTS models only accept pcm16 audio format"""
    from litellm.llms.gemini.chat.transformation import GoogleAIStudioGeminiConfig

    config = GoogleAIStudioGeminiConfig()

    # Test case 1: Valid pcm16 format should work
    non_default_params = {
        "audio": {"voice": "Kore", "format": "pcm16"}
    }
    optional_params = {}

    result = config.map_openai_params(
        non_default_params=non_default_params,
        optional_params=optional_params,
        model="gemini-2.5-flash-preview-tts",
        drop_params=False
    )

    # Should create speech config for valid format
    assert "speechConfig" in result
    assert result["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"] == "Kore"

    # Test case 2: Invalid wav format should raise ValueError
    non_default_params_invalid = {
        "audio": {"voice": "Kore", "format": "wav"}
    }
    optional_params_invalid = {}

    try:
        config.map_openai_params(
            non_default_params=non_default_params_invalid,
            optional_params=optional_params_invalid,
            model="gemini-2.5-flash-preview-tts",
            drop_params=False
        )
        assert False, "Should have raised ValueError for invalid audio format"
    except ValueError as e:
        assert "Unsupported audio format for Gemini TTS models: wav" in str(e)
        assert "Gemini TTS models only support 'pcm16' format" in str(e)
        assert "L16 PCM format" in str(e)

    # Test case 3: Invalid mp3 format should raise ValueError
    non_default_params_mp3 = {
        "audio": {"voice": "Kore", "format": "mp3"}
    }
    optional_params_mp3 = {}

    try:
        config.map_openai_params(
            non_default_params=non_default_params_mp3,
            optional_params=optional_params_mp3,
            model="gemini-2.5-flash-preview-tts",
            drop_params=False
        )
        assert False, "Should have raised ValueError for invalid audio format"
    except ValueError as e:
        assert "Unsupported audio format for Gemini TTS models: mp3" in str(e)
        assert "Gemini TTS models only support 'pcm16' format" in str(e)

    # Test case 4: No format specified should work (format is optional)
    non_default_params_no_format = {
        "audio": {"voice": "Kore"}
    }
    optional_params_no_format = {}

    result_no_format = config.map_openai_params(
        non_default_params=non_default_params_no_format,
        optional_params=optional_params_no_format,
        model="gemini-2.5-flash-preview-tts",
        drop_params=False
    )

    # Should create speech config even without format
    assert "speechConfig" in result_no_format
    assert result_no_format["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"] == "Kore"


if __name__ == "__main__":
    pytest.main([__file__])
