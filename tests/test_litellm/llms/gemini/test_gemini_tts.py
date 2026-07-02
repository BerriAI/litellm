"""
Test Gemini TTS (Text-to-Speech) functionality
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath("../../../.."))  # Adds the parent directory to the system path

import litellm
from litellm.endpoints.speech.speech_to_completion_bridge.transformation import (
    SpeechToCompletionBridgeTransformationHandler,
)
from litellm.llms.gemini.chat.transformation import GoogleAIStudioGeminiConfig
from litellm.utils import get_supported_openai_params


GEMINI_3_1_FLASH_TTS_MODEL = "gemini-3.1-flash-tts-preview"

MULTI_SPEAKER_SPEECH_CONFIG = {
    "multi_speaker_voice_config": {
        "speaker_voice_configs": [
            {
                "speaker": "Ryan",
                "voice_config": {
                    "prebuilt_voice_config": {
                        "voice_name": "Umbriel",
                    },
                },
            },
            {
                "speaker": "Katie",
                "voice_config": {
                    "prebuilt_voice_config": {
                        "voice_name": "Leda",
                    },
                },
            },
        ],
    },
    "language_code": "en-US",
}

NORMALIZED_MULTI_SPEAKER_SPEECH_CONFIG = {
    "multiSpeakerVoiceConfig": {
        "speakerVoiceConfigs": [
            {
                "speaker": "Ryan",
                "voiceConfig": {
                    "prebuiltVoiceConfig": {
                        "voiceName": "Umbriel",
                    },
                },
            },
            {
                "speaker": "Katie",
                "voiceConfig": {
                    "prebuiltVoiceConfig": {
                        "voiceName": "Leda",
                    },
                },
            },
        ],
    },
    "languageCode": "en-US",
}


class TestGeminiTTSTransformation:
    """Test Gemini TTS transformation functionality"""

    def test_gemini_tts_model_detection(self):
        """Test that TTS models are correctly identified"""
        config = GoogleAIStudioGeminiConfig()

        # Test TTS models (both preview and non-preview versions)
        assert config.is_model_gemini_audio_model("gemini-2.5-flash-preview-tts")
        assert config.is_model_gemini_audio_model("gemini-2.5-pro-preview-tts")
        assert config.is_model_gemini_audio_model("gemini-2.5-flash-tts")
        assert config.is_model_gemini_audio_model("gemini-2.5-pro-tts")
        assert config.is_model_gemini_audio_model(GEMINI_3_1_FLASH_TTS_MODEL)

        # Test non-TTS models
        assert not config.is_model_gemini_audio_model("gemini-2.5-flash")
        assert not config.is_model_gemini_audio_model("gemini-2.5-pro")
        assert not config.is_model_gemini_audio_model("gpt-4o-audio-preview")

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

        non_default_params = {"audio": {"voice": "Kore", "format": "pcm16"}}
        optional_params = {}

        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="gemini-2.5-flash-preview-tts",
            drop_params=False,
        )

        # Check speech config is created
        assert "speechConfig" in result
        assert "voiceConfig" in result["speechConfig"]
        assert "prebuiltVoiceConfig" in result["speechConfig"]["voiceConfig"]
        assert result["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"] == "Kore"

        # Check response modalities
        assert "responseModalities" in result
        assert "AUDIO" in result["responseModalities"]

    def test_gemini_tts_audio_parameter_mapping_with_language_code(self):
        config = GoogleAIStudioGeminiConfig()

        non_default_params = {"audio": {"voice": "Kore", "format": "pcm16", "language_code": "en-US"}}
        optional_params = {}

        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="gemini-2.5-flash-preview-tts",
            drop_params=False,
        )

        assert "speechConfig" in result
        assert result["speechConfig"]["languageCode"] == "en-US"
        assert result["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"] == "Kore"

    def test_map_audio_params_language_code(self):
        config = GoogleAIStudioGeminiConfig()

        result = config._map_audio_params({"voice": "Kore", "format": "pcm16", "language_code": "de-DE"})

        assert result["languageCode"] == "de-DE"
        assert result["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"] == "Kore"

    def test_map_audio_params_no_language_code(self):
        config = GoogleAIStudioGeminiConfig()

        result = config._map_audio_params({"voice": "Kore", "format": "pcm16"})

        assert "languageCode" not in result
        assert result["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"] == "Kore"

    def test_gemini_tts_multi_speaker_audio_parameter_mapping(self):
        """Test multi-speaker audio parameter mapping for Gemini 3.1 TTS models"""
        config = GoogleAIStudioGeminiConfig()

        non_default_params = {
            "audio": {
                "speech_config": MULTI_SPEAKER_SPEECH_CONFIG,
                "format": "pcm16",
            }
        }
        optional_params = {}

        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=GEMINI_3_1_FLASH_TTS_MODEL,
            drop_params=False,
        )

        assert result["speechConfig"] == NORMALIZED_MULTI_SPEAKER_SPEECH_CONFIG
        assert result["responseModalities"] == ["AUDIO"]
        assert "temperature" not in result

    def test_gemini_tts_audio_parameter_with_existing_modalities(self):
        """Test audio parameter mapping when modalities already exist"""
        config = GoogleAIStudioGeminiConfig()

        non_default_params = {"audio": {"voice": "Puck", "format": "pcm16"}}
        optional_params = {"responseModalities": ["TEXT"]}

        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="gemini-2.5-flash-preview-tts",
            drop_params=False,
        )

        # Check that AUDIO is added to existing modalities
        assert "responseModalities" in result
        assert "TEXT" in result["responseModalities"]
        assert "AUDIO" in result["responseModalities"]

    def test_gemini_tts_no_audio_parameter(self):
        """Test that non-audio parameters are handled normally"""
        config = GoogleAIStudioGeminiConfig()

        non_default_params = {"temperature": 0.7, "max_tokens": 100}
        optional_params = {}

        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="gemini-2.5-flash-preview-tts",
            drop_params=False,
        )

        # Should not have speech config
        assert "speechConfig" not in result
        # Should not automatically add audio modalities
        assert "responseModalities" not in result

    def test_gemini_tts_invalid_audio_parameter(self):
        """Test handling of invalid audio parameter"""
        config = GoogleAIStudioGeminiConfig()

        non_default_params = {"audio": "invalid_string"}  # Should be dict
        optional_params = {}

        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="gemini-2.5-flash-preview-tts",
            drop_params=False,
        )

        # Should not create speech config for invalid audio param
        assert "speechConfig" not in result

    def test_gemini_tts_empty_audio_parameter(self):
        """Test handling of empty audio parameter"""
        config = GoogleAIStudioGeminiConfig()

        non_default_params = {"audio": {}}
        optional_params = {}

        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model="gemini-2.5-flash-preview-tts",
            drop_params=False,
        )

        # Should still set response modalities even with empty audio config
        assert "responseModalities" in result
        assert "AUDIO" in result["responseModalities"]

    def test_gemini_tts_audio_format_validation(self):
        """Test audio format validation for TTS models"""
        config = GoogleAIStudioGeminiConfig()

        # Test invalid format
        non_default_params = {
            "audio": {"voice": "Kore", "format": "wav"}  # Invalid format
        }
        optional_params = {}

        with pytest.raises(ValueError, match="Unsupported audio format for Gemini TTS models"):
            config.map_openai_params(
                non_default_params=non_default_params,
                optional_params=optional_params,
                model="gemini-2.5-flash-preview-tts",
                drop_params=False,
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
    with patch("litellm.completion") as mock_completion:
        # Mock a successful TTS response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Generated audio response"
        mock_completion.return_value = mock_response

        # Test completion call with audio parameter
        response = litellm.completion(
            model="gemini-2.5-flash-preview-tts",
            messages=[{"role": "user", "content": "Say hello"}],
            audio={"voice": "Kore", "format": "pcm16"},
        )

        assert response is not None
        assert response.choices[0].message.content is not None


def test_gemini_tts_speech_bridge_accepts_multi_speaker_voice_dict():
    handler = SpeechToCompletionBridgeTransformationHandler()

    result = handler.transform_request(
        model=f"vertex_ai/{GEMINI_3_1_FLASH_TTS_MODEL}",
        input="Ryan: How are you doing today Katie?\nKatie: Not too bad.",
        voice=MULTI_SPEAKER_SPEECH_CONFIG,
        optional_params={"response_format": "mp3"},
        litellm_params={},
        headers={},
        litellm_logging_obj=MagicMock(),
        custom_llm_provider="vertex_ai",
    )

    assert result["modalities"] == ["audio"]
    assert result["audio"] == {
        "speech_config": NORMALIZED_MULTI_SPEAKER_SPEECH_CONFIG,
        "format": "pcm16",
    }
    assert "response_format" not in result


class TestGeminiTTSSpeechConfigInRequestBody:
    """Test that speechConfig is properly included in the final request body.

    This tests the full transformation pipeline, not just map_openai_params().
    Previously, speechConfig was created but filtered out because it was missing
    from the GenerationConfig TypedDict.
    """

    @pytest.mark.parametrize(
        "model,custom_llm_provider",
        [
            ("gemini-2.5-flash-tts", "vertex_ai"),
            ("gemini-2.5-flash-tts", "gemini"),
            ("gemini-2.5-flash-preview-tts", "vertex_ai"),
            ("gemini-2.5-flash-preview-tts", "gemini"),
            ("gemini-2.5-pro-tts", "vertex_ai"),
            (GEMINI_3_1_FLASH_TTS_MODEL, "vertex_ai"),
            (GEMINI_3_1_FLASH_TTS_MODEL, "gemini"),
        ],
    )
    def test_speechconfig_in_generation_config_transform_request_body(self, model, custom_llm_provider):
        """Test that speechConfig is included in generationConfig after _transform_request_body()"""
        from litellm.llms.vertex_ai.gemini.transformation import (
            _transform_request_body,
        )

        # Simulate optional_params after map_openai_params() has run
        optional_params = {
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}}},
            "responseModalities": ["AUDIO"],
        }

        messages = [{"role": "user", "content": "Say hello"}]

        # Call _transform_request_body which applies the filtering
        request_body = _transform_request_body(
            messages=messages,
            model=model,
            optional_params=optional_params,
            custom_llm_provider=custom_llm_provider,
            litellm_params={},
            cached_content=None,
        )

        # Verify speechConfig is in generationConfig (not filtered out)
        assert "generationConfig" in request_body
        generation_config = request_body["generationConfig"]
        assert "speechConfig" in generation_config, (
            f"speechConfig was filtered out of generationConfig for model={model}, provider={custom_llm_provider}. "
            "Ensure speechConfig is in the GenerationConfig TypedDict."
        )
        assert generation_config["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"] == "Kore"

    @pytest.mark.parametrize(
        "model,custom_llm_provider",
        [
            ("gemini-2.5-flash-tts", "vertex_ai"),
            ("gemini-2.5-flash-tts", "gemini"),
            ("gemini-2.5-flash-preview-tts", "vertex_ai"),
            (GEMINI_3_1_FLASH_TTS_MODEL, "vertex_ai"),
            (GEMINI_3_1_FLASH_TTS_MODEL, "gemini"),
        ],
    )
    def test_speechconfig_end_to_end_mapping(self, model, custom_llm_provider):
        """Test full pipeline: audio param -> map_openai_params -> _transform_request_body"""
        from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
            VertexGeminiConfig,
        )
        from litellm.llms.vertex_ai.gemini.transformation import (
            _transform_request_body,
        )

        config = VertexGeminiConfig()

        # Step 1: Map OpenAI audio param to speechConfig
        non_default_params = {"audio": {"voice": "Puck", "format": "pcm16"}}
        optional_params = {}

        mapped_params = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=False,
        )

        # Verify map_openai_params creates speechConfig
        assert "speechConfig" in mapped_params

        messages = [{"role": "user", "content": "Hello world"}]

        # Step 2: Transform to request body (this is where the bug was)
        request_body = _transform_request_body(
            messages=messages,
            model=model,
            optional_params=mapped_params,
            custom_llm_provider=custom_llm_provider,
            litellm_params={},
            cached_content=None,
        )

        # Verify speechConfig survives the transformation
        assert "generationConfig" in request_body
        generation_config = request_body["generationConfig"]
        assert "speechConfig" in generation_config, (
            f"speechConfig was filtered out during _transform_request_body() for model={model}, provider={custom_llm_provider}. "
            "This breaks Gemini TTS - speechConfig must be in GenerationConfig TypedDict."
        )
        assert generation_config["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"] == "Puck"

        # Also verify responseModalities is present
        assert "responseModalities" in generation_config
        assert "AUDIO" in generation_config["responseModalities"]

    @pytest.mark.parametrize(
        "custom_llm_provider",
        [
            "vertex_ai",
            "gemini",
        ],
    )
    def test_multi_speaker_speechconfig_end_to_end_mapping(self, custom_llm_provider):
        """Test full pipeline for Gemini 3.1 multi-speaker TTS speechConfig"""
        from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
            VertexGeminiConfig,
        )
        from litellm.llms.vertex_ai.gemini.transformation import (
            _transform_request_body,
        )

        config = VertexGeminiConfig()

        mapped_params = config.map_openai_params(
            non_default_params={
                "audio": {
                    "speech_config": MULTI_SPEAKER_SPEECH_CONFIG,
                    "format": "pcm16",
                }
            },
            optional_params={},
            model=GEMINI_3_1_FLASH_TTS_MODEL,
            drop_params=False,
        )

        request_body = _transform_request_body(
            messages=[{"role": "user", "content": "Ryan: Hi.\nKatie: Hello."}],
            model=GEMINI_3_1_FLASH_TTS_MODEL,
            optional_params=mapped_params,
            custom_llm_provider=custom_llm_provider,
            litellm_params={},
            cached_content=None,
        )

        generation_config = request_body["generationConfig"]
        assert generation_config["speechConfig"] == NORMALIZED_MULTI_SPEAKER_SPEECH_CONFIG
        assert generation_config["responseModalities"] == ["AUDIO"]
        assert "temperature" not in generation_config

    @pytest.mark.parametrize(
        "model,custom_llm_provider",
        [
            ("gemini-2.5-flash-tts", "vertex_ai"),
            ("gemini-2.5-flash-tts", "gemini"),
            ("gemini-2.5-flash-preview-tts", "vertex_ai"),
        ],
    )
    def test_language_code_end_to_end_mapping(self, model, custom_llm_provider):
        from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
            VertexGeminiConfig,
        )
        from litellm.llms.vertex_ai.gemini.transformation import (
            _transform_request_body,
        )

        config = VertexGeminiConfig()

        non_default_params = {"audio": {"voice": "Puck", "format": "pcm16", "language_code": "pt-BR"}}
        optional_params = {}

        mapped_params = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=False,
        )

        assert mapped_params["speechConfig"]["languageCode"] == "pt-BR"

        request_body = _transform_request_body(
            messages=[{"role": "user", "content": "Hello world"}],
            model=model,
            optional_params=mapped_params,
            custom_llm_provider=custom_llm_provider,
            litellm_params={},
            cached_content=None,
        )

        generation_config = request_body["generationConfig"]
        assert generation_config["speechConfig"]["languageCode"] == "pt-BR"
        assert generation_config["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"] == "Puck"
        assert "AUDIO" in generation_config["responseModalities"]


if __name__ == "__main__":
    pytest.main([__file__])
