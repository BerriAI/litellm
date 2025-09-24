"""
Test Gemini TTS (Text-to-Speech) functionality
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(
    0, os.path.abspath("../../../..")
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
                "format": "pcm16"
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
                "voice": "Puck",
                "format": "pcm16"
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

    def test_gemini_tts_audio_format_validation(self):
        """Test audio format validation for TTS models"""
        config = GoogleAIStudioGeminiConfig()
        
        # Test invalid format
        non_default_params = {
            "audio": {
                "voice": "Kore",
                "format": "wav"  # Invalid format
            }
        }
        optional_params = {}
        
        with pytest.raises(ValueError, match="Unsupported audio format for Gemini TTS models"):
            config.map_openai_params(
                non_default_params=non_default_params,
                optional_params=optional_params,
                model="gemini-2.5-flash-preview-tts",
                drop_params=False
            )

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
            audio={"voice": "Kore", "format": "pcm16"}
        )
        
        assert response is not None
        assert response.choices[0].message.content is not None


if __name__ == "__main__":
    pytest.main([__file__])
