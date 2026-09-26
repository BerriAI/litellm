import base64
import io
import math
import struct
import wave
from pathlib import Path
from typing import Final
from unittest.mock import MagicMock, Mock, patch

import httpx
import pytest

import litellm
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
            optional_params={"response_format": "pcm16"},
            voice=voice,
        )
        assert voice_str is None
        assert optional_params["audioEncoding"] == "LINEAR16"
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

    def test_gemini_tts_ignores_incomplete_speaker_entries(self):
        config: Final = VertexAITextToSpeechConfig()
        voice: Final = {
            "multiSpeakerVoiceConfig": {
                "speakerVoiceConfigs": [
                    None,
                    {"speakerAlias": "Ryan"},
                    {"speakerAlias": "Katie", "speakerId": "Leda"},
                ]
            }
        }

        _, optional_params = config.map_openai_params(
            model="gemini-3.1-flash-tts-preview",
            optional_params={"response_format": "mp3"},
            voice=voice,
        )

        assert optional_params["vertex_voice_dict"]["multiSpeakerVoiceConfig"] == {
            "speakerVoiceConfigs": [{"speakerAlias": "Katie", "speakerId": "Leda"}]
        }

    def test_gemini_tts_ignores_non_list_speaker_entries(self):
        config: Final = VertexAITextToSpeechConfig()

        assert config._extract_gemini_tts_speaker_configs(
            {"multiSpeakerVoiceConfig": {"speakerVoiceConfigs": "invalid"}}
        ) == []

    @pytest.mark.parametrize(
        ("voice", "expected_name"),
        [
            ({"speechConfig": {"voice": "Kore"}}, "Kore"),
            ({"speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}}}}, "Kore"),
            ({"speechConfig": {"voiceConfig": {}}}, None),
        ],
    )
    def test_gemini_tts_nested_voice_mapping(self, voice: dict[str, object], expected_name: str | None):
        config: Final = VertexAITextToSpeechConfig()

        _, optional_params = config.map_openai_params(
            model="gemini-3.1-flash-tts-preview",
            optional_params={"response_format": "mp3"},
            voice=voice,
        )

        mapped_voice: Final = optional_params["vertex_voice_dict"]
        if expected_name is None:
            assert mapped_voice["speechConfig"]["voiceConfig"] == {}
            assert "name" not in mapped_voice
        else:
            assert mapped_voice["name"] == expected_name

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

    @patch.object(  # test-quality-ok: isolates provider credentials while testing Gemini request serialization
        VertexAITextToSpeechConfig, "_ensure_access_token"
    )
    @patch.object(  # test-quality-ok: fixes the Gemini provider URL at the authentication boundary
        VertexAITextToSpeechConfig, "_get_token_and_url"
    )
    def test_gemini_tts_mp3_request_body(self, mock_get_token, mock_ensure_token):
        mock_ensure_token.return_value = ("mock-token", "test-project")
        mock_get_token.return_value = ("mock-token", "mock-url")
        config = VertexAITextToSpeechConfig()

        result = config.transform_text_to_speech_request(
            model="gemini-3.1-flash-tts-preview",
            input="Ryan: Hi.\nKatie: Hello.",
            voice=None,
            optional_params={
                "audioEncoding": "MP3",
                "vertex_voice_dict": {
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
                },
            },
            litellm_params={
                "vertex_credentials": None,
                "vertex_project": "test-project",
                "vertex_location": "global",
            },
            headers={},
        )

        request_body = result["dict_body"]
        assert request_body["input"] == {"text": "Ryan: Hi.\nKatie: Hello."}
        assert request_body["voice"] == {
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
        assert request_body["audioConfig"]["audioEncoding"] == "MP3"


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
def test_gemini_cloud_tts_response_bills_text_and_audio(encoding: str):
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
    logger.model_call_details = {
        "additional_args": {
            "complete_input_dict": {
                "dict_body": {
                    "input": {"text": input_text},
                    "audioConfig": {"audioEncoding": encoding, "sampleRateHertz": 24000},
                }
            }
        }
    }
    raw_response: Final = httpx.Response(200, json={"audioContent": base64.b64encode(audio_bytes).decode()})

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
    with patch("litellm.llms.vertex_ai.text_to_speech.transformation.calculate_request_duration", return_value=None):
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
    raw_response: Final = httpx.Response(200, json={"audioContent": base64.b64encode(audio).decode()})
    logger: Final = MagicMock()
    logger.model_call_details = {
        "additional_args": {
            "complete_input_dict": {
                "dict_body": {
                    "input": {"text": "Hello"},
                    "audioConfig": {"audioEncoding": encoding, "sampleRateHertz": 24000},
                }
            }
        }
    }

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


def test_gemini_cloud_tts_rejects_audio_without_measurable_duration():
    raw_response: Final = httpx.Response(200, json={"audioContent": base64.b64encode(b"invalid").decode()})
    logger: Final = MagicMock()
    logger.model_call_details = {
        "additional_args": {
            "complete_input_dict": {
                "dict_body": {
                    "input": {"text": "Hello"},
                    "audioConfig": {"audioEncoding": "MP3", "sampleRateHertz": 24000},
                }
            }
        }
    }

    with patch("litellm.llms.vertex_ai.text_to_speech.transformation.calculate_request_duration", return_value=None):
        with pytest.raises(ValueError, match="Cannot determine Gemini TTS output duration"):
            VertexAITextToSpeechConfig().transform_text_to_speech_response(
                "gemini-3.1-flash-tts-preview", raw_response, logger
            )


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


@patch(  # test-quality-ok: exercises public speech dispatch up to the outbound HTTP boundary
    "litellm.llms.custom_httpx.llm_http_handler.HTTPHandler.post"
)
@patch.object(  # test-quality-ok: isolates provider credentials in the public API test
    VertexAITextToSpeechConfig, "_ensure_access_token"
)
@patch.object(  # test-quality-ok: fixes the provider URL for deterministic dispatch assertions
    VertexAITextToSpeechConfig, "_get_token_and_url"
)
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


@patch(  # test-quality-ok: exercises public Gemini speech dispatch up to the outbound HTTP boundary
    "litellm.llms.custom_httpx.llm_http_handler.HTTPHandler.post"
)
@patch.object(  # test-quality-ok: isolates provider credentials in the public Gemini API test
    VertexAITextToSpeechConfig, "_ensure_access_token"
)
@patch.object(  # test-quality-ok: fixes the Gemini provider URL for deterministic dispatch assertions
    VertexAITextToSpeechConfig, "_get_token_and_url"
)
def test_litellm_speech_vertex_ai_gemini_tts_mp3_uses_cloud_tts(mock_get_token, mock_ensure_token, mock_post):
    mock_ensure_token.return_value = ("mock-token", "test-project")
    mock_get_token.return_value = ("mock-token", "mock-url")
    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    audio_bytes: Final = (Path(__file__).resolve().parents[4] / "audio_tests/gemini_tts_speech.mp3").read_bytes()
    mock_response.json.return_value = {"audioContent": base64.b64encode(audio_bytes).decode()}
    mock_post.return_value = mock_response

    result: Final = litellm.speech(
        model="vertex_ai/gemini-3.1-flash-tts-preview",
        input="Ryan: Hi.\nKatie: Hello.",
        voice={
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
        },
        response_format="mp3",
        vertex_project="test-project",
        vertex_location="global",
    )

    assert result.response.content == audio_bytes
    assert result.usage is not None
    assert result.usage.prompt_tokens > 0
    assert result.usage.completion_tokens > 0

    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["url"] == "https://texttospeech.googleapis.com/v1/text:synthesize"
    request_body = call_kwargs["json"]
    assert request_body["audioConfig"]["audioEncoding"] == "MP3"
    assert request_body["voice"] == {
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
