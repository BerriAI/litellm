import base64
import io
import math
import struct
import sys
import wave
from pathlib import Path
from typing import Final
import json
from collections.abc import Iterator, Mapping
from unittest.mock import MagicMock, Mock, patch

import httpx
import pytest
import respx
import responses
from pydantic import ValidationError

import litellm
from litellm.litellm_core_utils.exception_mapping_utils import exception_type
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import OPENAI_TO_GEMINI_TTS_VOICES
from litellm.llms.vertex_ai.text_to_speech.transformation import (
    VertexAILyriaTextToSpeechConfig,
    VertexAITextToSpeechConfig,
    _fallback_gemini_tts_audio_duration,
)
from litellm.cost_calculator import response_cost_calculator
from litellm.litellm_core_utils.audio_utils.utils import calculate_request_duration
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

pytestmark = pytest.mark.usefixtures("local_model_cost_map")


@pytest.mark.parametrize(
    ("voice", "multi_speaker"),
    (
        ("aLlOy", False),
        ({"name": "aLlOy"}, False),
        ({"voice_config": {"prebuilt_voice_config": {"voice_name": "aLlOy"}}}, False),
        (
            {
                "multi_speaker_voice_config": {
                    "speaker_voice_configs": [
                        {"speaker": "reader", "voice_config": {"prebuilt_voice_config": {"voice_name": "aLlOy"}}}
                    ]
                }
            },
            True,
        ),
    ),
)
def test_gemini_cloud_tts_maps_openai_voice_names_in_all_voice_forms(
    voice: str | dict[str, object], multi_speaker: bool
) -> None:
    config: Final = VertexAITextToSpeechConfig()
    model: Final = "gemini-3.1-flash-tts-preview"
    _, params = config.map_openai_params(model=model, voice=voice, optional_params={"response_format": "mp3"})
    mapped_voice: Final = OPENAI_TO_GEMINI_TTS_VOICES["alloy"]
    expected_voice: Final = {
        "languageCode": config.DEFAULT_LANGUAGE_CODE,
        "modelName": model,
        **(
            {
                "multiSpeakerVoiceConfig": {
                    "speakerVoiceConfigs": [{"speakerAlias": "reader", "speakerId": mapped_voice}]
                }
            }
            if multi_speaker
            else {"name": mapped_voice}
        ),
    }

    assert params["vertex_voice_dict"] == expected_voice


class TestVertexAITextToSpeechConfig:
    """Tests for VertexAITextToSpeechConfig transformation"""

    def test_get_complete_url(self):
        """Test that get_complete_url returns the correct Google Cloud TTS API URL"""
        config = VertexAITextToSpeechConfig()

        url = config.get_complete_url(
            model="vertex_ai/chirp",
            api_base=None,
            litellm_params={},
        )

        assert url == "https://texttospeech.googleapis.com/v1/text:synthesize"

    def test_get_complete_url_with_custom_api_base(self):
        """Test that get_complete_url uses custom api_base when provided"""
        config = VertexAITextToSpeechConfig()

        custom_url = "https://custom-tts-endpoint.example.com/v1/synthesize"
        url = config.get_complete_url(
            model="vertex_ai/chirp",
            api_base=custom_url,
            litellm_params={},
        )

        assert url == custom_url

    @patch.object(  # test-quality-ok: isolates provider credentials while testing request serialization
        VertexAITextToSpeechConfig, "_ensure_access_token"
    )
    @patch.object(  # test-quality-ok: fixes the provider URL at the authentication boundary
        VertexAITextToSpeechConfig, "_get_token_and_url"
    )
    def test_transform_text_to_speech_request_body(self, mock_get_token, mock_ensure_token):
        """Test that transform_text_to_speech_request generates correct request body"""
        # Mock authentication
        mock_ensure_token.return_value = ("mock-token", "test-project")
        mock_get_token.return_value = ("mock-token", "mock-url")

        config = VertexAITextToSpeechConfig()

        # Test with voice dict in litellm_params (as set by dispatch)
        result = config.transform_text_to_speech_request(
            model="vertex_ai/chirp",
            input="Hello, this is a test",
            voice=None,
            optional_params={
                "vertex_voice_dict": {
                    "languageCode": "en-US",
                    "name": "en-US-Chirp3-HD-Charon",
                }
            },
            litellm_params={
                "vertex_credentials": None,
                "vertex_project": "test-project",
                "vertex_location": "us-central1",
            },
            headers={},
        )

        # Verify request body structure
        assert "dict_body" in result
        request_body = result["dict_body"]

        assert "input" in request_body
        assert request_body["input"] == {"text": "Hello, this is a test"}

        assert "voice" in request_body
        assert request_body["voice"]["languageCode"] == "en-US"
        assert request_body["voice"]["name"] == "en-US-Chirp3-HD-Charon"

        assert "audioConfig" in request_body

        # Verify headers contain auth
        assert "headers" in result
        assert "Authorization" in result["headers"]

        raw_voice_request: Final = config.transform_text_to_speech_request(
            model="vertex_ai/chirp",
            input="Hello",
            voice="en-US-Chirp3-HD-Charon",
            optional_params={},
            litellm_params={
                "vertex_credentials": None,
                "vertex_project": "test-project",
                "vertex_location": "us-central1",
            },
            headers={},
        )
        assert raw_voice_request["dict_body"]["voice"]["name"] == "en-US-Chirp3-HD-Charon"

    def test_voice_mapping_openai_to_vertex(self):
        """Test that OpenAI voice names are correctly mapped to Vertex AI voices"""
        config = VertexAITextToSpeechConfig()

        # Test the _map_voice_to_vertex_format helper
        voice_str, voice_dict = config._map_voice_to_vertex_format("alloy")

        assert voice_str == "alloy"
        assert voice_dict is not None
        assert voice_dict["name"] == "en-US-Studio-O"
        assert voice_dict["languageCode"] == "en-US"

    def test_voice_mapping_vertex_voice_passthrough(self):
        """Test that Vertex AI voice names are passed through directly"""
        config = VertexAITextToSpeechConfig()

        # Test with a Chirp3 HD voice
        voice_str, voice_dict = config._map_voice_to_vertex_format("en-US-Chirp3-HD-Charon")

        assert voice_str == "en-US-Chirp3-HD-Charon"
        assert voice_dict is not None
        assert voice_dict["name"] == "en-US-Chirp3-HD-Charon"
        assert voice_dict["languageCode"] == "en-US"

    def test_voice_mapping_dict_passthrough(self):
        """Test that voice dict is passed through unchanged"""
        config = VertexAITextToSpeechConfig()

        voice_input = {
            "languageCode": "de-DE",
            "name": "de-DE-Chirp3-HD-Charon",
        }
        voice_str, voice_dict = config._map_voice_to_vertex_format(voice_input)

        assert voice_str is None
        assert voice_dict == voice_input

    def test_gemini_tts_multi_speaker_voice_mapping(self):
        config = VertexAITextToSpeechConfig()

        voice = {
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
        }

        expected_voice = {
            "languageCode": "en-US",
            "modelName": "gemini-3.1-flash-tts-preview",
            "multiSpeakerVoiceConfig": {
                "speakerVoiceConfigs": [
                    {
                        "speakerAlias": "Ryan",
                        "speakerId": "Umbriel",
                    },
                    {
                        "speakerAlias": "Katie",
                        "speakerId": "Leda",
                    },
                ],
            },
        }

        voice_str, optional_params = config.map_openai_params(
            model="gemini-3.1-flash-tts-preview",
            optional_params={"response_format": "mp3"},
            voice=voice,
        )
        assert voice_str is None
        assert optional_params["audioEncoding"] == "MP3"
        assert optional_params["vertex_voice_dict"] == expected_voice

        voice_str, optional_params = config.map_openai_params(
            model="gemini-3.1-flash-tts-preview",
            optional_params={"response_format": "opus"},
            voice=voice,
        )
        assert voice_str is None
        assert optional_params["audioEncoding"] == "OGG_OPUS"
        assert optional_params["vertex_voice_dict"] == expected_voice

    def test_gemini_tts_nested_language_and_voice_alias_reach_cloud_tts(self):
        config: Final = VertexAITextToSpeechConfig()
        voice: Final = {"speech_config": {"language_code": "fr-FR"}, "voice": "Kore"}

        voice_name, optional_params = config.map_openai_params(
            model="gemini-3.1-flash-tts-preview",
            optional_params={"response_format": "mp3"},
            voice=voice,
        )

        assert voice_name is None
        assert optional_params["vertex_voice_dict"] == {
            "languageCode": "fr-FR",
            "modelName": "gemini-3.1-flash-tts-preview",
            "name": "Kore",
        }

    @pytest.mark.parametrize(
        ("voice", "expected_voice"),
        [
            (
                {
                    "multi_speaker_voice_config": {
                        "speaker_voice_configs": [
                            {"speaker": "Ryan", "voice_config": {"prebuilt_voice_config": {"voice_name": "Umbriel"}}},
                            {"speaker": "Katie", "voice_config": {"prebuilt_voice_config": {"voice_name": "Leda"}}},
                        ],
                    },
                    "speech_config": {"language_code": "fr-FR"},
                },
                {
                    "languageCode": "fr-FR",
                    "modelName": "gemini-3.1-flash-tts-preview",
                    "multiSpeakerVoiceConfig": {
                        "speakerVoiceConfigs": [
                            {"speakerAlias": "Ryan", "speakerId": "Umbriel"},
                            {"speakerAlias": "Katie", "speakerId": "Leda"},
                        ],
                    },
                },
            ),
            (
                {
                    "voice_config": {"prebuilt_voice_config": {"voice_name": "Puck"}},
                    "speechConfig": {"languageCode": "en-GB"},
                },
                {"languageCode": "en-GB", "modelName": "gemini-3.1-flash-tts-preview", "name": "Puck"},
            ),
        ],
    )
    def test_gemini_tts_voice_beside_speech_config_wrapper_reaches_cloud_tts(
        self, voice: dict[str, object], expected_voice: dict[str, object]
    ):
        config: Final = VertexAITextToSpeechConfig()

        voice_name, optional_params = config.map_openai_params(
            model="gemini-3.1-flash-tts-preview",
            optional_params={"response_format": "mp3"},
            voice=voice,
        )

        assert voice_name is None
        assert optional_params["vertex_voice_dict"] == expected_voice

    @pytest.mark.parametrize(
        ("voice", "expected_language", "expected_voices"),
        [
            (
                {
                    "multiSpeakerVoiceConfig": {
                        "speakerVoiceConfigs": [
                            {"speaker": "Ryan", "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Umbriel"}}},
                            {"speaker": "Katie", "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Leda"}}},
                        ],
                    },
                    "speech_config": {
                        "multi_speaker_voice_config": {
                            "speaker_voice_configs": [
                                {"speaker": "Ana", "voice_config": {"prebuilt_voice_config": {"voice_name": "Kore"}}},
                                {"speaker": "Ben", "voice_config": {"prebuilt_voice_config": {"voice_name": "Puck"}}},
                            ],
                        },
                    },
                },
                None,
                (("Ana", "Kore"), ("Ben", "Puck")),
            ),
            (
                {"name": "Puck", "languageCode": "fr-FR", "speech_config": {"language_code": "de-DE"}},
                "de-DE",
                ((None, "Puck"),),
            ),
            (
                {"name": "Puck", "voice_config": {"prebuilt_voice_config": {"voice_name": "Charon"}}},
                None,
                ((None, "Charon"),),
            ),
        ],
    )
    def test_gemini_tts_cloud_tts_speaks_the_voice_wav_sends_to_gemini(
        self,
        voice: dict[str, object],
        expected_language: str | None,
        expected_voices: tuple[tuple[str | None, str], ...],
    ):
        from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import normalize_gemini_speech_config

        _, optional_params = VertexAITextToSpeechConfig().map_openai_params(
            model="gemini-3.1-flash-tts-preview",
            optional_params={"response_format": "mp3"},
            voice=voice,
        )
        cloud_tts_voice: Final = optional_params["vertex_voice_dict"]
        cloud_tts_speakers: Final = cloud_tts_voice.get("multiSpeakerVoiceConfig", {}).get("speakerVoiceConfigs")
        cloud_tts_voices: Final = (
            tuple((speaker["speakerAlias"], speaker["speakerId"]) for speaker in cloud_tts_speakers)
            if cloud_tts_speakers is not None
            else ((None, cloud_tts_voice["name"]),)
        )
        wav_config: Final = normalize_gemini_speech_config(voice, drop_unknown_fields=True)
        wav_speakers: Final = wav_config.get("multiSpeakerVoiceConfig", {}).get("speakerVoiceConfigs")
        wav_voices: Final = (
            tuple(
                (speaker["speaker"], speaker["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"])
                for speaker in wav_speakers
            )
            if wav_speakers is not None
            else ((None, wav_config["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"]),)
        )

        assert cloud_tts_voices == wav_voices == expected_voices
        assert cloud_tts_voice["languageCode"] == (expected_language or "en-US")
        assert wav_config.get("languageCode") == expected_language

    def test_gemini_tts_string_voice_reaches_cloud_tts(self):
        config: Final = VertexAITextToSpeechConfig()

        voice_name, optional_params = config.map_openai_params(
            model="gemini-3.1-flash-tts-preview",
            optional_params={"response_format": "mp3"},
            voice="Kore",
        )

        assert voice_name == "Kore"
        assert optional_params["vertex_voice_dict"] == {
            "languageCode": "en-US",
            "modelName": "gemini-3.1-flash-tts-preview",
            "name": "Kore",
        }

    @pytest.mark.parametrize(
        "speaker_voice_configs",
        (
            [None, {"speakerAlias": "Katie", "speakerId": "Leda"}],
            [{"speakerAlias": "Ryan"}, {"speakerAlias": "Katie", "speakerId": "Leda"}],
            [
                {"speaker": "Ryan", "voice_config": {"prebuilt_voice_config": {"name": "Umbriel"}}},
                {"speakerAlias": "Katie", "speakerId": "Leda"},
            ],
            "invalid",
        ),
    )
    def test_gemini_tts_malformed_speaker_entry_is_a_bad_request(self, speaker_voice_configs: object):
        config: Final = VertexAITextToSpeechConfig()

        with pytest.raises(litellm.UnsupportedParamsError, match="speaker_voice_configs"):
            config.map_openai_params(
                model="gemini-3.1-flash-tts-preview",
                optional_params={"response_format": "mp3"},
                voice={"multiSpeakerVoiceConfig": {"speakerVoiceConfigs": speaker_voice_configs}},
            )

    def test_gemini_tts_malformed_speaker_entry_is_dropped_with_drop_params(self):
        config: Final = VertexAITextToSpeechConfig()

        _, optional_params = config.map_openai_params(
            model="gemini-3.1-flash-tts-preview",
            optional_params={"response_format": "mp3"},
            voice={
                "multiSpeakerVoiceConfig": {
                    "speakerVoiceConfigs": [
                        {"speakerAlias": "Ryan", "speakerId": "Umbriel"},
                        {"speakerAlias": "Katie"},
                    ]
                }
            },
            drop_params=True,
        )

        assert optional_params["vertex_voice_dict"]["multiSpeakerVoiceConfig"] == {
            "speakerVoiceConfigs": [{"speakerAlias": "Ryan", "speakerId": "Umbriel"}]
        }

    @pytest.mark.parametrize(
        "voice",
        [
            {"speechConfig": {"voice": "Kore"}},
            {"speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}}}},
            {"speechConfig": {"voiceConfig": {}}},
            {"speech_config": {}},
            {"multi_speaker_voice_config": {"speaker_voice_configs": []}},
            {"languageCode": "en-US"},
        ],
    )
    def test_gemini_tts_nested_voice_mapping_sends_only_cloud_tts_voice_fields(self, voice: dict[str, object]):
        config: Final = VertexAITextToSpeechConfig()

        _, optional_params = config.map_openai_params(
            model="gemini-3.1-flash-tts-preview",
            optional_params={"response_format": "mp3"},
            voice=voice,
        )

        assert optional_params["vertex_voice_dict"] == {
            "languageCode": "en-US",
            "modelName": "gemini-3.1-flash-tts-preview",
            "name": "Kore",
        }

    @pytest.mark.parametrize(
        ("voice", "unsendable_field"),
        [
            (
                {"voice_config": {"replicated_voice_config": {"mime_type": "audio/wav", "voice_sample_audio": "UklG"}}},
                "voiceConfig.replicatedVoiceConfig",
            ),
            (
                {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Puck", "style": "calm"}}},
                "voiceConfig.prebuiltVoiceConfig.style",
            ),
            ({"speechConfig": {"voiceConfig": {"voice": "Puck"}}}, "voiceConfig.voice"),
            (
                {
                    "multiSpeakerVoiceConfig": {
                        "speakerVoiceConfigs": [
                            {"speaker": "Ryan", "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Puck"}}},
                            {"speaker": "Katie", "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Leda"}}, "tone": "x"},
                        ]
                    }
                },
                "multiSpeakerVoiceConfig.speakerVoiceConfigs.tone",
            ),
            (
                {
                    "multiSpeakerVoiceConfig": {
                        "speakerVoiceConfigs": [
                            {"speaker": "Ryan", "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Puck"}}, "tone": "x"},
                            {"speaker": "Katie", "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Leda"}}, "tone": "y"},
                        ]
                    }
                },
                "multiSpeakerVoiceConfig.speakerVoiceConfigs.tone",
            ),
            (
                {
                    "multiSpeakerVoiceConfig": {
                        "speakerVoiceConfigs": [
                            {
                                "speakerAlias": "Ryan",
                                "speakerId": "Puck",
                                "voiceConfig": {
                                    "replicatedVoiceConfig": {"mimeType": "audio/wav", "voiceSampleAudio": "AAAA"}
                                },
                            },
                            {"speakerAlias": "Katie", "speakerId": "Kore"},
                        ]
                    }
                },
                "multiSpeakerVoiceConfig.speakerVoiceConfigs.voiceConfig.replicatedVoiceConfig",
            ),
            ({"languageCode": ["en-US"], "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Puck"}}}, "languageCode"),
            ({"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": 5}}}, "voiceConfig.prebuiltVoiceConfig.voiceName"),
            ({"voiceConfig": "Puck"}, "voiceConfig"),
            ({"multiSpeakerVoiceConfig": [{"speaker": "Ryan"}]}, "multiSpeakerVoiceConfig"),
            ({"name": 5}, "name"),
            ({"speechConfig": {"voice": ["Puck"]}}, "speechConfig.voice"),
            ({"speechConfig": "Puck"}, "speechConfig"),
            ({"speech_config": [{"voice_config": {"prebuilt_voice_config": {"voice_name": "Puck"}}}]}, "speech_config"),
            (
                {
                    "multiSpeakerVoiceConfig": {
                        "speakerVoiceConfigs": [
                            {
                                "speakerAlias": "Ryan",
                                "speakerId": "Umbriel",
                                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Puck"}},
                            },
                            {"speakerAlias": "Katie", "speakerId": "Kore"},
                        ]
                    }
                },
                "multiSpeakerVoiceConfig.speakerVoiceConfigs.voiceConfig.prebuiltVoiceConfig",
            ),
        ],
    )
    def test_gemini_tts_cloud_tts_rejects_voice_fields_it_cannot_send(
        self, voice: dict[str, object], unsendable_field: str
    ):
        config: Final = VertexAITextToSpeechConfig()

        with pytest.raises(litellm.UnsupportedParamsError) as exc_info:
            config.map_openai_params(
                model="gemini-3.1-flash-tts-preview",
                optional_params={"response_format": "mp3"},
                voice=voice,
            )

        assert f"voice fields {unsendable_field} as mp3" in str(exc_info.value)

    @pytest.mark.parametrize(
        ("voice", "voice_name"),
        [
            ({"voiceConfig": {"replicatedVoiceConfig": {"mimeType": "audio/wav", "voiceSampleAudio": "UklG"}}}, "Kore"),
            ({"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Puck", "style": "calm"}}}, "Puck"),
            ({"voiceConfig": "Puck"}, "Kore"),
            ({"speechConfig": "Puck"}, "Kore"),
        ],
    )
    def test_gemini_tts_cloud_tts_drops_voice_fields_it_cannot_send_with_drop_params(
        self, voice: dict[str, object], voice_name: str
    ):
        config: Final = VertexAITextToSpeechConfig()

        _, optional_params = config.map_openai_params(
            model="gemini-3.1-flash-tts-preview",
            optional_params={"response_format": "mp3"},
            voice=voice,
            drop_params=True,
        )

        assert optional_params["vertex_voice_dict"] == {
            "languageCode": "en-US",
            "modelName": "gemini-3.1-flash-tts-preview",
            "name": voice_name,
        }

    @pytest.mark.parametrize(
        "voice",
        [
            {"name": "Umbriel", "modelName": "gemini-2.5-flash-tts"},
            {"name": "Umbriel", "model_name": "gemini-2.5-flash-tts"},
            {"modelName": "gemini-2.5-flash-tts", "model_name": "chirp-3"},
            {
                "modelName": "gemini-2.5-flash-tts",
                "model_name": "chirp-3",
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
                    ],
                },
            },
        ],
    )
    def test_gemini_tts_ignores_voice_model_name_override(self, voice):
        config = VertexAITextToSpeechConfig()
        routed_model = "gemini-3.1-flash-tts-preview"

        _, optional_params = config.map_openai_params(
            model=routed_model,
            optional_params={"response_format": "mp3"},
            voice=voice,
        )
        assert optional_params["vertex_voice_dict"]["modelName"] == routed_model

    def test_dispatch_maps_gemini_cloud_tts_params_when_provider_config_skipped(self):
        config = VertexAITextToSpeechConfig()
        handler = MagicMock()
        handler.text_to_speech_handler.return_value = "ok"

        result = config.dispatch_text_to_speech(
            model="gemini-3.1-flash-tts-preview",
            input="Hi",
            voice={
                "name": "Umbriel",
                "modelName": "gemini-2.5-flash-tts",
            },
            optional_params={"response_format": "mp3"},
            litellm_params_dict={},
            logging_obj=MagicMock(),
            timeout=10,
            extra_headers=None,
            base_llm_http_handler=handler,
            aspeech=False,
            api_base=None,
            api_key=None,
        )

        assert result == "ok"
        mapped_params = handler.text_to_speech_handler.call_args.kwargs["text_to_speech_optional_params"]
        assert mapped_params["audioEncoding"] == "MP3"
        assert mapped_params["vertex_voice_dict"]["name"] == "Umbriel"
        assert mapped_params["vertex_voice_dict"]["modelName"] == "gemini-3.1-flash-tts-preview"

    @pytest.mark.parametrize(
        ("voice", "expected_name"),
        [
            ("en-US-Chirp3-HD-Charon", "en-US-Chirp3-HD-Charon"),
            ({"name": "en-US-Chirp3-HD-Charon"}, "en-US-Chirp3-HD-Charon"),
            (None, None),
        ],
    )
    def test_dispatch_keeps_pre_mapped_cloud_tts_params(
        self, voice: str | dict[str, str] | None, expected_name: str | None
    ):
        config = VertexAITextToSpeechConfig()
        handler = MagicMock()
        handler.text_to_speech_handler.return_value = "ok"
        optional_params = {
            "audioEncoding": "OGG_OPUS",
            "vertex_voice_dict": {"languageCode": "en-US", "name": "en-US-Chirp3-HD-Charon"},
        }

        config.dispatch_text_to_speech(
            model="chirp",
            input="Hi",
            voice=voice,
            optional_params=optional_params,
            litellm_params_dict={},
            logging_obj=MagicMock(),
            timeout=10,
            extra_headers=None,
            base_llm_http_handler=handler,
            aspeech=False,
            api_base=None,
            api_key=None,
        )

        call_kwargs = handler.text_to_speech_handler.call_args.kwargs
        assert call_kwargs["voice"] == expected_name
        assert call_kwargs["text_to_speech_optional_params"] is optional_params
        assert call_kwargs["text_to_speech_optional_params"]["audioEncoding"] == "OGG_OPUS"


@pytest.mark.parametrize(
    ("audio", "expected_content_type"),
    [
        (b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00", "audio/wav"),
        (b"\xff\xfb\x90\x64\x00\x00\x00\x00", "audio/mpeg"),
        (b"OggS" + b"\x00" * 24 + b"OpusHead", "audio/opus"),
        (b"fLaC\x00\x00\x00\x22", "audio/flac"),
    ],
)
def test_transform_text_to_speech_response_labels_content_type(audio, expected_content_type):
    raw_response = httpx.Response(
        status_code=200,
        json={"audioContent": base64.b64encode(audio).decode()},
    )

    result = VertexAITextToSpeechConfig().transform_text_to_speech_response(
        model="vertex_ai/chirp",
        raw_response=raw_response,
        logging_obj=MagicMock(),
    )

    assert result.response.headers["content-type"] == expected_content_type
    assert result.response.content == audio


def test_transform_text_to_speech_response_leaves_unknown_bytes_unlabeled():
    raw_pcm = b"\x00\x01\x02\x03\x04\x05\x06\x07"
    raw_response = httpx.Response(
        status_code=200,
        json={"audioContent": base64.b64encode(raw_pcm).decode()},
    )

    result = VertexAITextToSpeechConfig().transform_text_to_speech_response(
        model="vertex_ai/chirp",
        raw_response=raw_response,
        logging_obj=MagicMock(),
    )

    assert "content-type" not in result.response.headers
    assert result.response.content == raw_pcm


@pytest.mark.parametrize("encoding", ["LINEAR16", "PCM", "MP3", "OGG_OPUS", "ALAW", "MULAW"])
def test_gemini_cloud_tts_response_bills_text_and_audio(encoding: str, monkeypatch: pytest.MonkeyPatch) -> None:
    model: Final = "gemini-3.1-flash-tts-preview"
    input_text: Final = "Hello from Gemini text to speech"
    pcm_audio: Final = b"\x00\x00" * 24000
    wav_buffer: Final = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(pcm_audio)
    g711_audio: Final = (
        struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF",
            36 + 24000,
            b"WAVE",
            b"fmt ",
            16,
            6 if encoding == "ALAW" else 7,
            1,
            24000,
            24000,
            1,
            8,
            b"data",
            24000,
        )
        + b"\x00" * 24000
    )
    fixture_dir: Final = Path(__file__).resolve().parents[4] / "audio_tests"
    audio_by_encoding: Final = {
        "PCM": pcm_audio,
        "LINEAR16": wav_buffer.getvalue(),
        "MP3": (fixture_dir / "gemini_tts_speech.mp3").read_bytes(),
        "OGG_OPUS": (fixture_dir / "gemini_tts_speech.ogg").read_bytes(),
        "ALAW": g711_audio,
        "MULAW": g711_audio,
    }
    audio_bytes: Final = audio_by_encoding[encoding]
    logger: Final = MagicMock()
    sent_request: Final = httpx.Request(
        "POST",
        _SYNTHESIZE_URL,
        json={
            "input": {"text": input_text},
            "audioConfig": {"audioEncoding": encoding, "sampleRateHertz": 24000},
        },
    )
    raw_response: Final = httpx.Response(
        200, json={"audioContent": base64.b64encode(audio_bytes).decode()}, request=sent_request
    )

    result: Final = VertexAITextToSpeechConfig().transform_text_to_speech_response(model, raw_response, logger)
    usage: Final = result.usage
    assert usage is not None
    duration: Final = len(audio_bytes) / 48000 if encoding == "PCM" else calculate_request_duration(audio_bytes)
    assert duration is not None
    assert _fallback_gemini_tts_audio_duration(audio_bytes, encoding, 24000) == pytest.approx(duration), (
        "Google Cloud AudioEncoding and RFC 7845, checked 2026-09-23: "
        "https://cloud.google.com/text-to-speech/docs/reference/rest/v1/AudioEncoding "
        "https://www.rfc-editor.org/rfc/rfc7845.html"
    )
    assert usage.completion_tokens == math.ceil(duration * 25), (
        "Google Cloud TTS pricing, 2026-09-23: https://cloud.google.com/text-to-speech/pricing"
    )
    assert usage.prompt_tokens == litellm.token_counter(model=model, text=input_text)
    assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens
    assert usage.completion_tokens_details is not None
    assert usage.completion_tokens_details.audio_tokens == usage.completion_tokens
    with monkeypatch.context() as dependencies:
        dependencies.setitem(sys.modules, "soundfile", None)
        sdk_result: Final = VertexAITextToSpeechConfig().transform_text_to_speech_response(model, raw_response, logger)
    assert sdk_result.usage == usage

    model_info: Final = litellm.get_model_info(model, custom_llm_provider="vertex_ai")
    cost: Final = response_cost_calculator(
        response_object=result,
        model=model,
        custom_llm_provider="vertex_ai",
        call_type="speech",
        optional_params={},
        prompt=input_text,
    )
    expected_cost: Final = (
        usage.prompt_tokens * model_info["input_cost_per_token"]
        + usage.completion_tokens * model_info["output_cost_per_audio_token"]
    )
    assert cost == pytest.approx(expected_cost)
    assert cost > usage.prompt_tokens * model_info["input_cost_per_token"]


@pytest.mark.parametrize("encoding", ["ALAW", "MULAW"])
@pytest.mark.parametrize("audio", [b"\x12" * 24000, b"RIFF" + b"\x12" * 23996])
def test_gemini_cloud_tts_returns_raw_g711_audio_when_duration_decoder_is_unavailable(encoding: str, audio: bytes):
    logger: Final = MagicMock()
    sent_request: Final = httpx.Request(
        "POST",
        _SYNTHESIZE_URL,
        json={
            "input": {"text": "Hello"},
            "audioConfig": {"audioEncoding": encoding, "sampleRateHertz": 24000},
        },
    )
    raw_response: Final = httpx.Response(
        200, json={"audioContent": base64.b64encode(audio).decode()}, request=sent_request
    )

    with patch("litellm.llms.vertex_ai.text_to_speech.transformation.calculate_request_duration", return_value=None):
        result: Final = VertexAITextToSpeechConfig().transform_text_to_speech_response(
            "gemini-3.1-flash-tts-preview", raw_response, logger
        )

    assert result.response.content == audio
    assert result.usage is not None
    assert result.usage.prompt_tokens > 0
    assert result.usage.completion_tokens == 25, (
        "Google Cloud TTS pricing, 2026-09-23: https://cloud.google.com/text-to-speech/pricing"
    )


def test_gemini_cloud_tts_returns_audio_unbilled_when_duration_is_unmeasurable():
    logger: Final = MagicMock()
    sent_request: Final = httpx.Request(
        "POST",
        _SYNTHESIZE_URL,
        json={
            "input": {"text": "Hello"},
            "audioConfig": {"audioEncoding": "MP3", "sampleRateHertz": 24000},
        },
    )
    raw_response: Final = httpx.Response(
        200, json={"audioContent": base64.b64encode(b"invalid").decode()}, request=sent_request
    )

    with patch("litellm.llms.vertex_ai.text_to_speech.transformation.calculate_request_duration", return_value=None):
        result: Final = VertexAITextToSpeechConfig().transform_text_to_speech_response(
            "gemini-3.1-flash-tts-preview", raw_response, logger
        )

    assert result.response.content == b"invalid"
    assert result.usage is None


class TestVertexAILyriaTextToSpeechConfig:
    @pytest.mark.parametrize(
        "model",
        ["lyria-002", "vertex_ai/lyria-3-clip-preview", "lyria-3-pro-preview"],
    )
    def test_provider_config_manager_selects_lyria_config(self, model):
        config = ProviderConfigManager.get_provider_text_to_speech_config(
            model=model,
            provider=LlmProviders.VERTEX_AI,
        )

        assert isinstance(config, VertexAILyriaTextToSpeechConfig)

    @pytest.mark.parametrize(
        ("model", "vertex_ai_audio_api", "supported_audio_formats", "expected_url"),
        [
            (
                "future-lyria-predict",
                "lyria_predict",
                ["wav"],
                "https://us-central1-aiplatform.googleapis.com/v1/projects/music-project/locations/"
                "us-central1/publishers/google/models/future-lyria-predict:predict",
            ),
            (
                "future-music-interactions",
                "lyria_interactions",
                ["mp3", "wav"],
                "https://aiplatform.googleapis.com/v1beta1/projects/music-project/locations/global/interactions",
            ),
        ],
    )
    def test_dispatches_from_model_metadata(
        self,
        monkeypatch,
        model,
        vertex_ai_audio_api,
        supported_audio_formats,
        expected_url,
    ):
        monkeypatch.setitem(
            litellm.model_cost,
            f"vertex_ai/{model}",
            {
                "vertex_ai_audio_api": vertex_ai_audio_api,
                "supported_audio_formats": supported_audio_formats,
            },
        )

        config = ProviderConfigManager.get_provider_text_to_speech_config(
            model=model,
            provider=LlmProviders.VERTEX_AI,
        )

        assert isinstance(config, VertexAILyriaTextToSpeechConfig)
        assert (
            config.get_complete_url(
                model=model,
                api_base=None,
                litellm_params={
                    "vertex_project": "music-project",
                    "vertex_location": "us-central1",
                },
            )
            == expected_url
        )

    def test_vertex_chirp_does_not_select_lyria_config(self):
        config = ProviderConfigManager.get_provider_text_to_speech_config(
            model="chirp",
            provider=LlmProviders.VERTEX_AI,
        )

        assert isinstance(config, VertexAITextToSpeechConfig)
        assert not isinstance(config, VertexAILyriaTextToSpeechConfig)

    def test_get_complete_url_for_lyria_2(self):
        config = VertexAILyriaTextToSpeechConfig()

        url = config.get_complete_url(
            model="lyria-002",
            api_base=None,
            litellm_params={
                "vertex_project": "music-project",
                "vertex_location": "europe-west4",
            },
        )

        assert url == (
            "https://europe-west4-aiplatform.googleapis.com/v1/projects/music-project/"
            "locations/europe-west4/publishers/google/models/lyria-002:predict"
        )

    def test_get_complete_url_encodes_injected_predict_path_segments(self, monkeypatch: pytest.MonkeyPatch) -> None:
        injected: Final = "victim-project/locations/us-central1/publishers/google/models/other-model:predict?ignored="
        encoded: Final = (
            "victim-project%2Flocations%2Fus-central1%2Fpublishers%2Fgoogle"
            "%2Fmodels%2Fother-model%3Apredict%3Fignored%3D"
        )
        monkeypatch.setitem(
            litellm.model_cost,
            f"vertex_ai/{injected}",
            {
                "vertex_ai_audio_api": "lyria_predict",
                "supported_audio_formats": ["wav"],
            },
        )

        url: Final = VertexAILyriaTextToSpeechConfig().get_complete_url(
            model=injected,
            api_base="https://us-central1-aiplatform.googleapis.com",
            litellm_params={
                "vertex_project": injected,
                "vertex_location": injected,
            },
        )

        assert url == (
            "https://us-central1-aiplatform.googleapis.com"
            f"/v1/projects/{encoded}/locations/{encoded}/publishers/google/models/{encoded}:predict"
        )

    def test_get_complete_url_for_lyria_3(self):
        config = VertexAILyriaTextToSpeechConfig()

        url = config.get_complete_url(
            model="lyria-3-pro-preview",
            api_base=None,
            litellm_params={"vertex_project": "music-project"},
        )

        assert url == ("https://aiplatform.googleapis.com/v1beta1/projects/music-project/locations/global/interactions")

    @pytest.mark.parametrize(
        ("model", "response_format", "expected_body"),
        [
            (
                "lyria-002",
                "wav",
                {
                    "instances": [{"prompt": "A bright synth track"}],
                    "parameters": {"sample_count": 1},
                },
            ),
            (
                "lyria-3-clip-preview",
                "mp3",
                {
                    "model": "lyria-3-clip-preview",
                    "input": "A bright synth track",
                },
            ),
            (
                "lyria-3-pro-preview",
                "wav",
                {
                    "model": "lyria-3-pro-preview",
                    "input": "A bright synth track",
                    "response_format": {
                        "type": "audio",
                        "mime_type": "audio/wav",
                    },
                },
            ),
        ],
    )
    def test_transform_request(
        self,
        model,
        response_format,
        expected_body,
    ):
        class _LyriaConfig(VertexAILyriaTextToSpeechConfig):
            def _ensure_access_token(self, *args: object, **kwargs: object) -> tuple[str, str]:
                return "mock-token", "music-project"

        config = _LyriaConfig()

        request = config.transform_text_to_speech_request(
            model=model,
            input="A bright synth track",
            voice="alloy",
            optional_params={"response_format": response_format},
            litellm_params={"vertex_project": "music-project"},
            headers={},
        )

        assert request["dict_body"] == expected_body
        assert request["headers"]["Authorization"] == "Bearer mock-token"
        assert request["headers"]["x-goog-user-project"] == "music-project"

    @pytest.mark.parametrize(
        ("model", "response_json", "expected_audio", "expected_mime_type"),
        [
            (
                "lyria-002",
                {
                    "predictions": [
                        {
                            "bytesBase64Encoded": "UklGRiQAAABXQVZFZm10IA==",
                        }
                    ]
                },
                b"RIFF$\x00\x00\x00WAVEfmt ",
                "audio/wav",
            ),
            (
                "lyria-3-pro-preview",
                {
                    "steps": [
                        {
                            "type": "model_output",
                            "content": [
                                {"type": "text", "text": "Generated lyrics"},
                                {
                                    "type": "audio",
                                    "data": "bHlyaWEtMy1hdWRpbw==",
                                    "mime_type": "audio/mpeg",
                                },
                            ],
                        }
                    ]
                },
                b"lyria-3-audio",
                "audio/mpeg",
            ),
            (
                "lyria-3-clip-preview",
                {
                    "outputs": [
                        {"type": "text", "text": "Generated lyrics"},
                        {
                            "type": "audio",
                            "data": "bHlyaWEtMy1hdWRpbw==",
                            "mime_type": "audio/mpeg",
                        },
                    ]
                },
                b"lyria-3-audio",
                "audio/mpeg",
            ),
            (
                "lyria-3-pro-preview",
                {
                    "outputs": [
                        {
                            "type": "audio",
                            "data": "UklGRiQAAABXQVZFZm10IA==",
                        }
                    ]
                },
                b"RIFF$\x00\x00\x00WAVEfmt ",
                "audio/wav",
            ),
        ],
    )
    def test_transform_response(
        self,
        model,
        response_json,
        expected_audio,
        expected_mime_type,
    ):
        config = VertexAILyriaTextToSpeechConfig()
        raw_response = httpx.Response(200, json=response_json)

        response = config.transform_text_to_speech_response(
            model=model,
            raw_response=raw_response,
            logging_obj=MagicMock(),
        )

        assert response.content == expected_audio
        assert response.response.headers["content-type"] == expected_mime_type

    @pytest.mark.parametrize(
        ("model", "response_format"),
        [
            ("lyria-002", "mp3"),
            ("lyria-3-clip-preview", "wav"),
            ("lyria-3-pro-preview", "opus"),
        ],
    )
    def test_rejects_unsupported_response_format(self, model, response_format):
        config = VertexAILyriaTextToSpeechConfig()

        with pytest.raises(litellm.UnsupportedParamsError):
            config.map_openai_params(
                model=model,
                optional_params={"response_format": response_format},
            )

    @pytest.mark.parametrize("param", ["speed", "instructions"])
    def test_rejects_unsupported_openai_params(self, param):
        config = VertexAILyriaTextToSpeechConfig()

        with pytest.raises(litellm.UnsupportedParamsError):
            config.map_openai_params(
                model="lyria-3-pro-preview",
                optional_params={param: "unsupported"},
            )

    @pytest.mark.parametrize(
        ("model", "response_format", "response_json", "expected_url", "expected_body"),
        [
            (
                "lyria-002",
                "wav",
                {
                    "predictions": [
                        {
                            "audioContent": "bHlyaWEtMi1hdWRpbw==",
                            "mimeType": "audio/wav",
                        }
                    ]
                },
                "https://us-central1-aiplatform.googleapis.com/v1/projects/music-project/locations/us-central1/publishers/google/models/lyria-002:predict",
                {
                    "instances": [{"prompt": "A bright synth track"}],
                    "parameters": {"sample_count": 1},
                },
            ),
            (
                "lyria-3-pro-preview",
                "mp3",
                {
                    "steps": [
                        {
                            "type": "model_output",
                            "content": [
                                {
                                    "type": "audio",
                                    "data": "bHlyaWEtMy1hdWRpbw==",
                                    "mime_type": "audio/mpeg",
                                }
                            ],
                        }
                    ]
                },
                "https://aiplatform.googleapis.com/v1beta1/projects/music-project/locations/global/interactions",
                {
                    "model": "lyria-3-pro-preview",
                    "input": "A bright synth track",
                },
            ),
        ],
    )
    def test_litellm_speech_dispatches_to_lyria_api(
        self,
        model,
        response_format,
        response_json,
        expected_url,
        expected_body,
    ):
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = response_json
        with (
            patch.object(  # test-quality-ok: litellm.speech has no seam for Vertex token minting
                VertexAILyriaTextToSpeechConfig,
                "_ensure_access_token",
                return_value=("mock-token", "music-project"),
            ),
            patch(  # test-quality-ok: litellm.speech has no seam for the HTTP handler
                "litellm.llms.custom_httpx.llm_http_handler.HTTPHandler.post",
                return_value=mock_response,
            ) as mock_post,
        ):
            response = litellm.speech(
                model=f"vertex_ai/{model}",
                input="A bright synth track",
                voice="alloy",
                response_format=response_format,
                vertex_project="music-project",
                vertex_location="us-central1",
            )

        assert response.content in {b"lyria-2-audio", b"lyria-3-audio"}
        mock_post.assert_called_once()
        assert mock_post.call_args.kwargs["url"] == expected_url
        assert mock_post.call_args.kwargs["json"] == expected_body


@pytest.mark.parametrize("endpoint_kwarg", ["api_base", "base_url"])
def test_litellm_speech_vertex_ai_sends_request_to_the_configured_endpoint(endpoint_kwarg: str):
    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = {"audioContent": "SGVsbG8gV29ybGQ="}
    with (
        patch.object(  # test-quality-ok: litellm.speech has no seam for Vertex token minting
            VertexAITextToSpeechConfig, "_ensure_access_token", return_value=("mock-token", "test-project")
        ),
        patch(  # test-quality-ok: litellm.speech has no seam for the HTTP handler
            "litellm.llms.custom_httpx.llm_http_handler.HTTPHandler.post", return_value=mock_response
        ) as mock_post,
    ):
        response = litellm.speech(
            model="vertex_ai/chirp",
            input="Hello",
            voice="en-US-Chirp3-HD-Charon",
            vertex_project="test-project",
            vertex_location="us-central1",
            **{endpoint_kwarg: "https://tts.gateway.internal/v1/text:synthesize"},
        )

    assert mock_post.call_args.kwargs["url"] == "https://tts.gateway.internal/v1/text:synthesize"
    assert response.content == b"Hello World"


@patch("litellm.llms.custom_httpx.llm_http_handler.HTTPHandler.post")
@patch.object(VertexAITextToSpeechConfig, "_ensure_access_token")
@patch.object(VertexAITextToSpeechConfig, "_get_token_and_url")
def test_litellm_speech_vertex_ai_chirp(mock_get_token, mock_ensure_token, mock_post):
    """
    Test that litellm.speech(model="vertex_ai/chirp") sends the correct URL and request body
    """
    # Mock authentication
    mock_ensure_token.return_value = ("mock-token", "test-project")
    mock_get_token.return_value = ("mock-token", "mock-url")

    # Mock HTTP response
    mock_response = Mock(spec=httpx.Response)
    mock_response.content = b'{"audioContent": "SGVsbG8gV29ybGQ="}'  # base64 encoded "Hello World"
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response.json.return_value = {"audioContent": "SGVsbG8gV29ybGQ="}
    mock_post.return_value = mock_response

    litellm.speech(
        model="vertex_ai/chirp",
        input="Hello, this is a test",
        voice="en-US-Chirp3-HD-Charon",
        vertex_project="test-project",
        vertex_location="us-central1",
    )

    # Verify the HTTP call was made
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args.kwargs

    # Verify the URL is the Google Cloud TTS API
    assert call_kwargs["url"] == "https://texttospeech.googleapis.com/v1/text:synthesize"

    # Verify request body structure
    assert "json" in call_kwargs
    request_body = call_kwargs["json"]

    # Verify input
    assert "input" in request_body
    assert request_body["input"] == {"text": "Hello, this is a test"}

    # Verify voice
    assert "voice" in request_body
    assert request_body["voice"]["name"] == "en-US-Chirp3-HD-Charon"
    assert request_body["voice"]["languageCode"] == "en-US"

    # Verify audioConfig
    assert "audioConfig" in request_body

    # Verify headers contain authorization
    assert "headers" in call_kwargs
    assert "Authorization" in call_kwargs["headers"]
    assert call_kwargs["headers"]["Authorization"] == "Bearer mock-token"


@pytest.mark.parametrize("payload", [{}, {"audioContent": ""}, {"audioContent": None}, {"audioContent": []}])
def test_transform_text_to_speech_response_without_audio_content_reports_it_missing(payload: dict[str, object]):
    with pytest.raises(ValueError, match="No audioContent in Vertex AI TTS response"):
        VertexAITextToSpeechConfig().transform_text_to_speech_response(
            model="vertex_ai/chirp",
            raw_response=httpx.Response(200, json=payload),
            logging_obj=MagicMock(),
        )


@pytest.mark.parametrize(
    "payload",
    [
        ["not", "an", "object"],
        {"audioContent": 7},
        {"audioContent": ["UklGRiQAAABXQVZFZm10IA=="]},
    ],
)
def test_transform_text_to_speech_response_rejects_malformed_payloads_without_echoing_them(payload: object):
    with pytest.raises(ValidationError) as exc_info:
        VertexAITextToSpeechConfig().transform_text_to_speech_response(
            model="vertex_ai/chirp",
            raw_response=httpx.Response(200, json=payload),
            logging_obj=MagicMock(),
        )

    assert "input_value" not in str(exc_info.value)


_SYNTHESIZE_URL: Final = "https://texttospeech.googleapis.com/v1/text:synthesize"
_AUTHORIZED_USER: Final = json.dumps(
    {
        "type": "authorized_user",
        "client_id": "synthetic-client-id",
        "client_secret": "synthetic-client-secret",
        "refresh_token": "synthetic-refresh-token",
        "quota_project_id": "test-project",
    }
)
_ASYNC_INPUT: Final = "async hello what llm guardrail do you have"
_UK_VOICE: Final = {"languageCode": "en-UK", "name": "en-UK-Studio-O"}
_UK_AUDIO_CONFIG: Final = {"audioEncoding": "LINEAR22", "speakingRate": "10"}


@pytest.fixture
def google_token_endpoint(monkeypatch: pytest.MonkeyPatch) -> Iterator[responses.RequestsMock]:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    with responses.RequestsMock(assert_all_requests_are_fired=False) as token_endpoint:
        token_endpoint.post(
            "https://oauth2.googleapis.com/token",
            json={"access_token": "minted-google-token", "expires_in": 3600, "token_type": "Bearer"},
        )
        yield token_endpoint
    litellm.in_memory_llm_clients_cache.flush_cache()


async def _aspeech_vertex(
    respx_mock: respx.MockRouter, speech_input: str, voice_params: Mapping[str, object]
) -> httpx.Request:
    route: Final = respx_mock.post(_SYNTHESIZE_URL).mock(
        return_value=httpx.Response(200, json={"audioContent": base64.b64encode(b"vertex-audio").decode()})
    )
    response: Final = await litellm.aspeech(
        model="vertex_ai/test",
        input=speech_input,
        vertex_credentials=_AUTHORIZED_USER,
        **voice_params,
    )
    assert response.content == b"vertex-audio"
    assert route.call_count == 1
    sent: Final = route.calls[0].request
    assert sent.headers["x-goog-user-project"] == "test-project"
    assert sent.headers["authorization"] == "Bearer minted-google-token"
    return sent


@pytest.mark.asyncio
async def test_aspeech_vertex_ai_default_voice_posts_synthesize_request(
    respx_mock: respx.MockRouter, google_token_endpoint: responses.RequestsMock
) -> None:
    sent: Final = await _aspeech_vertex(respx_mock, _ASYNC_INPUT, {})

    assert json.loads(sent.content) == {
        "input": {"text": _ASYNC_INPUT},
        "voice": {"languageCode": "en-US", "name": "en-US-Studio-O"},
        "audioConfig": {"audioEncoding": "LINEAR16", "speakingRate": "1"},
    }


@pytest.mark.asyncio
async def test_aspeech_vertex_ai_forwards_caller_voice_and_audio_config(
    respx_mock: respx.MockRouter, google_token_endpoint: responses.RequestsMock
) -> None:
    sent: Final = await _aspeech_vertex(respx_mock, _ASYNC_INPUT, {"voice": _UK_VOICE, "audioConfig": _UK_AUDIO_CONFIG})

    assert json.loads(sent.content) == {
        "input": {"text": _ASYNC_INPUT},
        "voice": _UK_VOICE,
        "audioConfig": _UK_AUDIO_CONFIG,
    }


@pytest.mark.asyncio
async def test_aspeech_vertex_ai_sends_ssml_input(
    respx_mock: respx.MockRouter, google_token_endpoint: responses.RequestsMock
) -> None:
    ssml: Final = """
    <speak>
        <p>Hello, world!</p>
        <p>This is a test of the <break strength="medium" /> text-to-speech API.</p>
    </speak>
    """

    sent: Final = await _aspeech_vertex(respx_mock, ssml, {"voice": _UK_VOICE, "audioConfig": _UK_AUDIO_CONFIG})

    assert json.loads(sent.content) == {
        "input": {"ssml": ssml},
        "voice": _UK_VOICE,
        "audioConfig": _UK_AUDIO_CONFIG,
    }


_WAV_AUDIO_B64: Final = "UklGRiQAAABXQVZFZm10IA=="
_WAV_AUDIO: Final = b"RIFF$\x00\x00\x00WAVEfmt "
_UNLABELED_AUDIO_B64: Final = "bHlyaWEtMy1hdWRpbw=="
_UNLABELED_AUDIO: Final = b"lyria-3-audio"


@pytest.mark.parametrize(
    ("model", "payload", "expected_audio", "expected_mime_type"),
    [
        pytest.param(
            "lyria-002",
            {
                "predictions": [
                    {"audioContent": None, "bytesBase64Encoded": _WAV_AUDIO_B64, "mimeType": None},
                    "trailing entry that is never read",
                ],
                "deployedModelId": "123",
            },
            _WAV_AUDIO,
            "audio/wav",
            id="predict-null-fields-and-unread-trailing-prediction",
        ),
        pytest.param(
            "lyria-002",
            {"predictions": [{"audioContent": _WAV_AUDIO_B64, "mimeType": ""}]},
            _WAV_AUDIO,
            "audio/wav",
            id="predict-empty-mime-type-is-sniffed",
        ),
        pytest.param(
            "lyria-002",
            {"predictions": [{"audioContent": _WAV_AUDIO_B64, "mimeType": "audio/x-custom"}]},
            _WAV_AUDIO,
            "audio/x-custom",
            id="predict-declared-mime-type-wins-over-sniffing",
        ),
        pytest.param(
            "lyria-002",
            {"predictions": [{"audioContent": "", "bytesBase64Encoded": ""}]},
            b"",
            "audio/mpeg",
            id="predict-empty-audio-strings-yield-empty-audio",
        ),
        pytest.param(
            "lyria-3-pro-preview",
            {
                "steps": [
                    {"type": "thought", "content": "free text that is never read as content parts"},
                    {"type": "model_output", "content": None},
                    {"type": "model_output", "content": [{"type": "audio", "data": _WAV_AUDIO_B64, "mime_type": None}]},
                ]
            },
            _WAV_AUDIO,
            "audio/wav",
            id="interactions-unread-step-content-and-null-fields",
        ),
        pytest.param(
            "lyria-3-pro-preview",
            {"steps": None, "outputs": [{"type": "audio", "data": _UNLABELED_AUDIO_B64, "mime_type": "audio/mpeg"}]},
            _UNLABELED_AUDIO,
            "audio/mpeg",
            id="interactions-null-steps-fall-back-to-outputs",
        ),
        pytest.param(
            "lyria-3-clip-preview",
            {
                "outputs": [
                    {"type": "audio", "data": 7, "mime_type": 5},
                    {"type": "audio", "data": _UNLABELED_AUDIO_B64, "mime_type": "audio/mpeg"},
                    {"type": "audio", "data": ""},
                ]
            },
            _UNLABELED_AUDIO,
            "audio/mpeg",
            id="interactions-last-non-empty-audio-part-wins-over-an-earlier-malformed-one",
        ),
    ],
)
def test_lyria_transform_response_tolerates_optional_and_unread_provider_fields(
    model: str, payload: dict[str, object], expected_audio: bytes, expected_mime_type: str
):
    response: Final = VertexAILyriaTextToSpeechConfig().transform_text_to_speech_response(
        model=model,
        raw_response=httpx.Response(200, json=payload),
        logging_obj=MagicMock(),
    )

    assert response.content == expected_audio
    assert response.response.headers["content-type"] == expected_mime_type


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        ("lyria-002", {}),
        ("lyria-002", {"predictions": None}),
        ("lyria-002", {"predictions": []}),
        ("lyria-002", {"predictions": [{}]}),
        ("lyria-002", {"predictions": [{"audioContent": "", "bytesBase64Encoded": None}]}),
        ("lyria-3-pro-preview", {}),
        ("lyria-3-pro-preview", {"steps": [], "outputs": None}),
        ("lyria-3-pro-preview", {"steps": [{"type": "model_output", "content": []}]}),
        ("lyria-3-pro-preview", {"outputs": [{"type": "audio", "data": ""}, {"type": "text", "data": "ignored"}]}),
    ],
)
def test_lyria_transform_response_without_audio_reports_it_missing(model: str, payload: dict[str, object]):
    with pytest.raises(ValueError, match=f"No generated audio found in Vertex AI {model} response"):
        VertexAILyriaTextToSpeechConfig().transform_text_to_speech_response(
            model=model,
            raw_response=httpx.Response(200, json=payload),
            logging_obj=MagicMock(),
        )


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        ("lyria-002", ["not", "an", "object"]),
        ("lyria-002", {"predictions": {"audioContent": _WAV_AUDIO_B64}}),
        ("lyria-002", {"predictions": ["not-an-object"]}),
        ("lyria-002", {"predictions": [{"audioContent": 7}]}),
        ("lyria-002", {"predictions": [{"audioContent": _WAV_AUDIO_B64, "mimeType": ["audio/wav"]}]}),
        ("lyria-3-pro-preview", ["not", "an", "object"]),
        ("lyria-3-pro-preview", {"steps": ["not-an-object"]}),
        ("lyria-3-pro-preview", {"steps": [{"type": "model_output", "content": ["not-an-object"]}]}),
        ("lyria-3-pro-preview", {"outputs": [{"type": "audio", "data": [_WAV_AUDIO_B64]}]}),
        ("lyria-3-pro-preview", {"outputs": [{"type": "audio", "data": _WAV_AUDIO_B64, "mime_type": 5}]}),
    ],
)
def test_lyria_transform_response_rejects_malformed_payloads_without_echoing_them(model: str, payload: object):
    with pytest.raises(ValidationError) as exc_info:
        VertexAILyriaTextToSpeechConfig().transform_text_to_speech_response(
            model=model,
            raw_response=httpx.Response(200, json=payload),
            logging_obj=MagicMock(),
        )

    assert "input_value" not in str(exc_info.value)


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"steps": [{"type": "thought"}] * 403 + ["not-an-object"]}, id="step"),
        pytest.param({"outputs": [{"type": "text"}] * 403 + [None]}, id="output"),
        pytest.param(
            {"steps": [{"type": "model_output", "content": [{"type": "text"}] * 403 + [7]}]},
            id="content-part",
        ),
    ],
)
def test_lyria_malformed_item_position_does_not_change_the_mapped_error(payload: dict[str, object]):
    with pytest.raises(ValidationError) as exc_info:
        VertexAILyriaTextToSpeechConfig().transform_text_to_speech_response(
            model="lyria-3-pro-preview",
            raw_response=httpx.Response(200, json=payload),
            logging_obj=MagicMock(),
        )

    with pytest.raises(litellm.APIConnectionError):
        exception_type(
            model="vertex_ai/lyria-3-pro-preview",
            custom_llm_provider="vertex_ai",
            original_exception=exc_info.value,
        )


_PCM_AUDIO: Final = b"\x00\x00\x01\x00" * 2400
_GENERATE_CONTENT_RESPONSE: Final = {
    "candidates": [
        {
            "content": {
                "role": "model",
                "parts": [
                    {
                        "inlineData": {
                            "mimeType": "audio/L16;codec=pcm;rate=24000",
                            "data": base64.b64encode(_PCM_AUDIO).decode(),
                        }
                    }
                ],
            },
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 10, "totalTokenCount": 15},
}


@pytest.mark.parametrize(
    ("model", "response_format"),
    (
        ("gemini-3.8-flash-tts", None),
        ("gemini-3.1-flash-tts-preview", None),
        ("gemini-3.1-flash-tts-preview", "wav"),
        ("gemini-3.1-flash-tts-preview", "pcm"),
    ),
)
@pytest.mark.asyncio
async def test_aspeech_vertex_ai_gemini_tts_formats_the_bridge_serves_stay_on_generate_content(
    respx_mock: respx.MockRouter,
    google_token_endpoint: responses.RequestsMock,
    model: str,
    response_format: str | None,
) -> None:
    route: Final = respx_mock.post(
        url__regex=rf"https://aiplatform\.googleapis\.com/.*/models/{model}:generateContent$"
    ).mock(return_value=httpx.Response(200, json=_GENERATE_CONTENT_RESPONSE))

    response: Final = await litellm.aspeech(
        model=f"vertex_ai/{model}",
        input="Say hello",
        voice="Kore",
        vertex_credentials=_AUTHORIZED_USER,
        vertex_location="global",
        **({} if response_format is None else {"response_format": response_format}),
    )

    assert route.call_count == 1
    assert _PCM_AUDIO in response.content


@pytest.mark.asyncio
async def test_aspeech_vertex_ai_gemini_tts_mp3_reaches_cloud_tts_for_any_gemini_tts_model(
    respx_mock: respx.MockRouter, google_token_endpoint: responses.RequestsMock
) -> None:
    audio_bytes: Final = (Path(__file__).resolve().parents[4] / "audio_tests/gemini_tts_speech.mp3").read_bytes()
    route: Final = respx_mock.post(_SYNTHESIZE_URL).mock(
        return_value=httpx.Response(200, json={"audioContent": base64.b64encode(audio_bytes).decode()})
    )

    response: Final = await litellm.aspeech(
        model="vertex_ai/gemini-3.8-flash-tts",
        input="Say hello",
        voice="Kore",
        response_format="mp3",
        vertex_credentials=_AUTHORIZED_USER,
        vertex_location="global",
    )

    assert response.content == audio_bytes
    sent: Final = json.loads(route.calls[0].request.content)
    assert sent["voice"]["modelName"] == "gemini-3.8-flash-tts"
    assert sent["voice"]["name"] == "Kore"
    assert sent["audioConfig"]["audioEncoding"] == "MP3"


@pytest.mark.asyncio
async def test_aspeech_vertex_ai_gemini_tts_mp3_without_voice_keeps_the_gemini_model(
    respx_mock: respx.MockRouter, google_token_endpoint: responses.RequestsMock
) -> None:
    audio_bytes: Final = (Path(__file__).resolve().parents[4] / "audio_tests/gemini_tts_speech.mp3").read_bytes()
    route: Final = respx_mock.post(_SYNTHESIZE_URL).mock(
        return_value=httpx.Response(200, json={"audioContent": base64.b64encode(audio_bytes).decode()})
    )

    await litellm.aspeech(
        model="vertex_ai/gemini-3.1-flash-tts-preview",
        input="Say hello",
        response_format="mp3",
        vertex_credentials=_AUTHORIZED_USER,
        vertex_location="global",
    )

    sent: Final = json.loads(route.calls[0].request.content)
    assert sent["voice"] == {"languageCode": "en-US", "modelName": "gemini-3.1-flash-tts-preview", "name": "Kore"}


@pytest.mark.parametrize("voice", ("en-US-Studio-O", "en-US-Neural2-F"))
@pytest.mark.asyncio
async def test_aspeech_vertex_ai_gemini_tts_mp3_rejects_a_cloud_tts_voice(
    voice: str, respx_mock: respx.MockRouter, google_token_endpoint: responses.RequestsMock
) -> None:
    with pytest.raises(litellm.BadRequestError, match=f"{voice} is a Cloud Text-to-Speech voice"):
        await litellm.aspeech(
            model="vertex_ai/gemini-3.1-flash-tts-preview",
            input="Say hello",
            voice=voice,
            response_format="mp3",
            vertex_credentials=_AUTHORIZED_USER,
            vertex_location="global",
        )

    assert not respx_mock.calls


@pytest.mark.asyncio
async def test_aspeech_vertex_ai_gemini_tts_mp3_drop_params_swaps_a_cloud_tts_voice_for_a_gemini_voice(
    respx_mock: respx.MockRouter, google_token_endpoint: responses.RequestsMock
) -> None:
    audio_bytes: Final = (Path(__file__).resolve().parents[4] / "audio_tests/gemini_tts_speech.mp3").read_bytes()
    route: Final = respx_mock.post(_SYNTHESIZE_URL).mock(
        return_value=httpx.Response(200, json={"audioContent": base64.b64encode(audio_bytes).decode()})
    )

    await litellm.aspeech(
        model="vertex_ai/gemini-3.1-flash-tts-preview",
        input="Say hello",
        voice="en-US-Studio-O",
        response_format="mp3",
        drop_params=True,
        vertex_credentials=_AUTHORIZED_USER,
        vertex_location="global",
    )

    assert json.loads(route.calls[0].request.content)["voice"]["name"] == "Kore"


@pytest.mark.asyncio
async def test_aspeech_vertex_ai_gemini_tts_mp3_without_a_location_uses_the_vertex_default_region(
    respx_mock: respx.MockRouter, google_token_endpoint: responses.RequestsMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("VERTEXAI_LOCATION", raising=False)
    monkeypatch.delenv("VERTEX_LOCATION", raising=False)
    audio_bytes: Final = (Path(__file__).resolve().parents[4] / "audio_tests/gemini_tts_speech.mp3").read_bytes()
    route: Final = respx_mock.post("https://us-texttospeech.googleapis.com/v1/text:synthesize").mock(
        return_value=httpx.Response(200, json={"audioContent": base64.b64encode(audio_bytes).decode()})
    )

    await litellm.aspeech(
        model="vertex_ai/gemini-3.1-flash-tts-preview",
        input="Say hello",
        voice="Kore",
        response_format="mp3",
        vertex_credentials=_AUTHORIZED_USER,
    )

    assert route.called


@pytest.mark.asyncio
async def test_aspeech_vertex_ai_gemini_tts_mp3_sends_instructions_as_the_style_prompt(
    respx_mock: respx.MockRouter, google_token_endpoint: responses.RequestsMock
) -> None:
    audio_bytes: Final = (Path(__file__).resolve().parents[4] / "audio_tests/gemini_tts_speech.mp3").read_bytes()
    route: Final = respx_mock.post(_SYNTHESIZE_URL).mock(
        return_value=httpx.Response(200, json={"audioContent": base64.b64encode(audio_bytes).decode()})
    )

    await litellm.aspeech(
        model="vertex_ai/gemini-3.1-flash-tts-preview",
        input="Have a wonderful day",
        voice="Kore",
        response_format="mp3",
        instructions="Read this slowly and calmly",
        vertex_credentials=_AUTHORIZED_USER,
        vertex_location="global",
    )

    sent: Final = json.loads(route.calls[0].request.content)
    assert sent["input"] == {"text": "Have a wonderful day", "prompt": "Read this slowly and calmly"}


@pytest.mark.asyncio
async def test_aspeech_vertex_ai_gemini_tts_mp3_with_a_numeric_speaking_rate_returns_billed_audio(
    respx_mock: respx.MockRouter, google_token_endpoint: responses.RequestsMock
) -> None:
    audio_bytes: Final = (Path(__file__).resolve().parents[4] / "audio_tests/gemini_tts_speech.mp3").read_bytes()
    route: Final = respx_mock.post(_SYNTHESIZE_URL).mock(
        return_value=httpx.Response(200, json={"audioContent": base64.b64encode(audio_bytes).decode()})
    )

    result: Final = await litellm.aspeech(
        model="vertex_ai/gemini-3.1-flash-tts-preview",
        input="Hello there",
        voice="Kore",
        response_format="mp3",
        audioConfig={"audioEncoding": "MP3", "speakingRate": 1.25},
        vertex_credentials=_AUTHORIZED_USER,
        vertex_location="global",
    )

    assert json.loads(route.calls[0].request.content)["audioConfig"]["speakingRate"] == 1.25
    assert result.content == audio_bytes
    assert result.usage is not None
    assert result.usage.completion_tokens > 0


@pytest.mark.parametrize(
    ("vertex_location", "synthesize_url"),
    (
        ("us-central1", "https://us-texttospeech.googleapis.com/v1/text:synthesize"),
        ("us-east5", "https://us-texttospeech.googleapis.com/v1/text:synthesize"),
        ("us", "https://us-texttospeech.googleapis.com/v1/text:synthesize"),
        ("europe-west4", "https://eu-texttospeech.googleapis.com/v1/text:synthesize"),
        ("europe-north1", "https://eu-texttospeech.googleapis.com/v1/text:synthesize"),
        ("eu", "https://eu-texttospeech.googleapis.com/v1/text:synthesize"),
        ("northamerica-northeast1", "https://northamerica-northeast1-texttospeech.googleapis.com/v1/text:synthesize"),
        ("global", _SYNTHESIZE_URL),
    ),
)
@pytest.mark.asyncio
async def test_aspeech_vertex_ai_gemini_tts_mp3_keeps_the_deployment_region(
    respx_mock: respx.MockRouter,
    google_token_endpoint: responses.RequestsMock,
    vertex_location: str,
    synthesize_url: str,
) -> None:
    audio_bytes: Final = (Path(__file__).resolve().parents[4] / "audio_tests/gemini_tts_speech.mp3").read_bytes()
    route: Final = respx_mock.post(synthesize_url).mock(
        return_value=httpx.Response(200, json={"audioContent": base64.b64encode(audio_bytes).decode()})
    )

    response: Final = await litellm.aspeech(
        model="vertex_ai/gemini-3.1-flash-tts-preview",
        input="Say hello",
        voice="Kore",
        response_format="mp3",
        vertex_credentials=_AUTHORIZED_USER,
        vertex_location=vertex_location,
    )

    assert route.call_count == 1
    assert response.content == audio_bytes


@pytest.mark.parametrize("vertex_location", ["asia-northeast1", "europe-west2", "australia-southeast1"])
@pytest.mark.asyncio
async def test_aspeech_vertex_ai_gemini_tts_mp3_rejects_a_region_cloud_tts_does_not_serve(
    respx_mock: respx.MockRouter, google_token_endpoint: responses.RequestsMock, vertex_location: str
) -> None:
    with pytest.raises(litellm.BadRequestError, match=f"no Gemini TTS location for vertex_location {vertex_location}"):
        await litellm.aspeech(
            model="vertex_ai/gemini-3.1-flash-tts-preview",
            input="Say hello",
            voice="Kore",
            response_format="mp3",
            vertex_credentials=_AUTHORIZED_USER,
            vertex_location=vertex_location,
        )

    assert not respx_mock.calls


@pytest.mark.asyncio
async def test_router_gemini_tts_mp3_unknown_voice_is_a_bad_request_that_keeps_the_deployment_serving(
    respx_mock: respx.MockRouter, google_token_endpoint: responses.RequestsMock
) -> None:
    audio_bytes: Final = (Path(__file__).resolve().parents[4] / "audio_tests/gemini_tts_speech.mp3").read_bytes()
    respx_mock.post(_SYNTHESIZE_URL).mock(
        side_effect=lambda request: (
            httpx.Response(
                404, json={"error": {"code": 404, "message": "Voice Koree not found.", "status": "NOT_FOUND"}}
            )
            if json.loads(request.content)["voice"]["name"] == "Koree"
            else httpx.Response(200, json={"audioContent": base64.b64encode(audio_bytes).decode()})
        )
    )
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "gemini-tts",
                "litellm_params": {
                    "model": "vertex_ai/gemini-3.1-flash-tts-preview",
                    "vertex_credentials": _AUTHORIZED_USER,
                    "vertex_location": "global",
                },
            }
        ],
        num_retries=0,
    )

    with pytest.raises(litellm.BadRequestError, match="Voice Koree not found"):
        await router.aspeech(model="gemini-tts", input="Say hello", voice="Koree", response_format="mp3")
    response: Final = await router.aspeech(model="gemini-tts", input="Say hello", voice="Kore", response_format="mp3")

    assert response.content == audio_bytes


@pytest.mark.asyncio
async def test_aspeech_vertex_ai_chirp_unknown_voice_stays_not_found(
    respx_mock: respx.MockRouter, google_token_endpoint: responses.RequestsMock
) -> None:
    respx_mock.post(_SYNTHESIZE_URL).mock(
        return_value=httpx.Response(
            404, json={"error": {"code": 404, "message": "Voice en-US-Nope not found.", "status": "NOT_FOUND"}}
        )
    )

    with pytest.raises(litellm.NotFoundError, match="Voice en-US-Nope not found"):
        await litellm.aspeech(
            model="vertex_ai/chirp",
            input="Say hello",
            voice={"languageCode": "en-US", "name": "en-US-Nope"},
            vertex_credentials=_AUTHORIZED_USER,
        )


@pytest.mark.asyncio
async def test_aspeech_vertex_ai_gemini_tts_mp3_rejects_a_deployment_api_base(
    respx_mock: respx.MockRouter, google_token_endpoint: responses.RequestsMock
) -> None:
    with pytest.raises(litellm.BadRequestError, match="Request wav or pcm"):
        await litellm.aspeech(
            model="vertex_ai/gemini-3.1-flash-tts-preview",
            input="Say hello",
            voice="Kore",
            response_format="mp3",
            vertex_credentials=_AUTHORIZED_USER,
            vertex_location="us-central1",
            api_base="https://vertex.gateway.internal",
        )


@pytest.mark.asyncio
async def test_aspeech_vertex_ai_gemini_tts_mp3_sends_speak_tags_as_text(
    respx_mock: respx.MockRouter, google_token_endpoint: responses.RequestsMock
) -> None:
    audio_bytes: Final = (Path(__file__).resolve().parents[4] / "audio_tests/gemini_tts_speech.mp3").read_bytes()
    route: Final = respx_mock.post(_SYNTHESIZE_URL).mock(
        return_value=httpx.Response(200, json={"audioContent": base64.b64encode(audio_bytes).decode()})
    )
    speech_input: Final = "<speak>Have a wonderful day</speak>"

    await litellm.aspeech(
        model="vertex_ai/gemini-3.1-flash-tts-preview",
        input=speech_input,
        voice="Kore",
        response_format="mp3",
        vertex_credentials=_AUTHORIZED_USER,
        vertex_location="global",
    )

    sent: Final = json.loads(route.calls[0].request.content)
    assert sent["input"] == {"text": speech_input}


@pytest.mark.parametrize("response_format", ("aac", "flac", "pcm16", "WAV"))
@pytest.mark.asyncio
async def test_aspeech_vertex_ai_gemini_tts_unsupported_format_is_rejected_before_calling_google(
    respx_mock: respx.MockRouter, google_token_endpoint: responses.RequestsMock, response_format: str
) -> None:
    with pytest.raises(litellm.BadRequestError, match="Supported response formats: alaw, mp3, mulaw, opus, pcm, wav"):
        await litellm.aspeech(
            model="vertex_ai/gemini-3.1-flash-tts-preview",
            input="Say hello",
            voice="Kore",
            response_format=response_format,
            vertex_credentials=_AUTHORIZED_USER,
            vertex_location="global",
        )


@pytest.mark.asyncio
async def test_aspeech_vertex_ai_gemini_tts_mp3_sends_both_speakers_to_cloud_tts(
    respx_mock: respx.MockRouter, google_token_endpoint: responses.RequestsMock
) -> None:
    audio_bytes: Final = (Path(__file__).resolve().parents[4] / "audio_tests/gemini_tts_speech.mp3").read_bytes()
    route: Final = respx_mock.post(_SYNTHESIZE_URL).mock(
        return_value=httpx.Response(200, json={"audioContent": base64.b64encode(audio_bytes).decode()})
    )

    result: Final = await litellm.aspeech(
        model="vertex_ai/gemini-3.1-flash-tts-preview",
        input="Ryan: Hi.\nKatie: Hello.",
        voice={
            "multi_speaker_voice_config": {
                "speaker_voice_configs": [
                    {"speaker": "Ryan", "voice_config": {"prebuilt_voice_config": {"voice_name": "Umbriel"}}},
                    {"speaker": "Katie", "voice_config": {"prebuilt_voice_config": {"voice_name": "Leda"}}},
                ],
            },
        },
        response_format="mp3",
        vertex_credentials=_AUTHORIZED_USER,
        vertex_location="global",
    )

    assert result.content == audio_bytes
    assert result.usage is not None
    assert result.usage.prompt_tokens > 0
    assert result.usage.completion_tokens > 0
    assert route.call_count == 1
    sent: Final = json.loads(route.calls[0].request.content)
    assert sent["audioConfig"]["audioEncoding"] == "MP3"
    assert sent["voice"] == {
        "languageCode": "en-US",
        "modelName": "gemini-3.1-flash-tts-preview",
        "multiSpeakerVoiceConfig": {
            "speakerVoiceConfigs": [
                {"speakerAlias": "Ryan", "speakerId": "Umbriel"},
                {"speakerAlias": "Katie", "speakerId": "Leda"},
            ],
        },
    }


@pytest.mark.parametrize(
    ("speech_params", "error"),
    (
        ({"voice": ["Kore"]}, "needs voice to be a Gemini voice name"),
        ({"voice": ""}, "needs voice to be a Gemini voice name"),
        ({"voice": {"name": "Kore", "style": "calm"}}, "does not support the voice keys style"),
        ({"voice": {"speechConfig": {"voice": "Kore", "pitch": 2}}}, "does not support the voice keys pitch"),
        ({"voice": "Kore", "instructions": ["Read slowly"]}, "needs instructions to be a string"),
    ),
)
@pytest.mark.asyncio
async def test_aspeech_vertex_ai_gemini_tts_mp3_malformed_params_are_bad_requests_unless_dropped(
    respx_mock: respx.MockRouter,
    google_token_endpoint: responses.RequestsMock,
    speech_params: dict[str, object],
    error: str,
) -> None:
    audio_bytes: Final = (Path(__file__).resolve().parents[4] / "audio_tests/gemini_tts_speech.mp3").read_bytes()
    route: Final = respx_mock.post(_SYNTHESIZE_URL).mock(
        return_value=httpx.Response(200, json={"audioContent": base64.b64encode(audio_bytes).decode()})
    )
    request: Final = {
        "model": "vertex_ai/gemini-3.1-flash-tts-preview",
        "input": "Say hello",
        "response_format": "mp3",
        "vertex_credentials": _AUTHORIZED_USER,
        "vertex_location": "global",
        **speech_params,
    }

    with pytest.raises(litellm.BadRequestError, match=error):
        await litellm.aspeech(**request)
    assert not route.called

    response: Final = await litellm.aspeech(**request, drop_params=True)

    sent: Final = json.loads(route.calls[0].request.content)
    assert response.content == audio_bytes
    assert sent["input"] == {"text": "Say hello"}
    assert sent["voice"] == {"languageCode": "en-US", "modelName": "gemini-3.1-flash-tts-preview", "name": "Kore"}


@pytest.mark.asyncio
async def test_aspeech_vertex_ai_beta_lyria_2_serves_the_openai_params_it_ignores(
    respx_mock: respx.MockRouter, google_token_endpoint: responses.RequestsMock
) -> None:
    route: Final = respx_mock.post(url__regex=r"https://us-central1-aiplatform\.googleapis\.com/.*/lyria-002:predict$").mock(
        return_value=httpx.Response(200, json={"predictions": [{"audioContent": "bHlyaWEtMi1hdWRpbw==", "mimeType": "audio/wav"}]})
    )

    response: Final = await litellm.aspeech(
        model="vertex_ai_beta/lyria-002",
        input="A bright synth track",
        voice="alloy",
        response_format="mp3",
        speed=1.5,
        vertex_credentials=_AUTHORIZED_USER,
        vertex_project="test-project",
        vertex_location="us-central1",
    )

    assert route.call_count == 1
    assert response.content == b"lyria-2-audio"


@pytest.mark.asyncio
async def test_aspeech_vertex_ai_gemini_tts_wav_unknown_voice_keys_are_bad_requests_unless_dropped(
    respx_mock: respx.MockRouter, google_token_endpoint: responses.RequestsMock
) -> None:
    route: Final = respx_mock.post(
        url__regex=r"https://aiplatform\.googleapis\.com/.*/models/gemini-3.1-flash-tts-preview:generateContent$"
    ).mock(return_value=httpx.Response(200, json=_GENERATE_CONTENT_RESPONSE))
    request: Final = {
        "model": "vertex_ai/gemini-3.1-flash-tts-preview",
        "input": "Say hello",
        "voice": {"speech_config": {"voice": "Kore", "style": "calm"}, "modelName": "gemini-3.1-flash-tts-preview"},
        "response_format": "wav",
        "vertex_credentials": _AUTHORIZED_USER,
        "vertex_location": "global",
    }

    with pytest.raises(litellm.BadRequestError, match="does not support the voice keys style"):
        await litellm.aspeech(**request)
    assert not route.called

    response: Final = await litellm.aspeech(**request, drop_params=True)

    assert _PCM_AUDIO in response.content
    assert json.loads(route.calls[0].request.content)["generationConfig"]["speechConfig"] == {
        "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}}
    }
