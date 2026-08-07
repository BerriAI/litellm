import io
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.main import _is_chirp_grpc_request, _resolve_chirp_hd_voice_name
from litellm.types.utils import TranscriptionResponse


class TestIsChirpGrpcRequest:
    @pytest.mark.parametrize(
        "model,optional_params,expected",
        [
            ("vertex_ai/chirp", {}, True),
            ("vertex_ai/chirp_2", {}, True),
            ("vertex_ai/chirp_telephony", {}, True),
            ("vertex_ai/chirp_3", {}, False),
            ("vertex_ai/chirp_3", {"use_grpc": True}, True),
            ("vertex_ai/latest_long", {}, False),
            ("vertex_ai/latest_long", {"use_grpc": True}, True),
            ("vertex_ai/latest_long", {"use_grpc": False}, False),
            ("openai/whisper-1", {}, False),
        ],
    )
    def test_routing_logic(self, model, optional_params, expected):
        assert _is_chirp_grpc_request(model, optional_params) is expected


class TestResolveChirpHdVoiceName:
    def test_chirp_hd_voice_string_returns_name(self):
        assert _resolve_chirp_hd_voice_name("en-US-Chirp3-HD-Charon", {}) == "en-US-Chirp3-HD-Charon"

    def test_chirp_hd_in_vertex_voice_dict_returns_name(self):
        optional_params = {"vertex_voice_dict": {"name": "es-ES-Chirp3-HD-Kore", "languageCode": "es-ES"}}
        assert _resolve_chirp_hd_voice_name(None, optional_params) == "es-ES-Chirp3-HD-Kore"

    def test_non_chirp_hd_voice_returns_none(self):
        assert _resolve_chirp_hd_voice_name("en-US-Studio-O", {}) is None

    def test_none_voice_returns_none(self):
        assert _resolve_chirp_hd_voice_name(None, {}) is None

    def test_non_chirp_hd_voice_dict_returns_none(self):
        optional_params = {"vertex_voice_dict": {"name": "en-US-Studio-O"}}
        assert _resolve_chirp_hd_voice_name(None, optional_params) is None

    def test_voice_string_when_dict_is_not_chirp_hd(self):
        optional_params = {"vertex_voice_dict": {"name": "en-US-Studio-O"}}
        assert _resolve_chirp_hd_voice_name("es-ES-Chirp3-HD-Charon", optional_params) == "es-ES-Chirp3-HD-Charon"

    def test_all_chirp_hd_variants(self):
        for voice in ["en-US-Chirp3-HD-Charon", "es-ES-Chirp3-HD-Aoede", "fr-FR-Chirp3-HD-Kore"]:
            assert _resolve_chirp_hd_voice_name(voice, {}) == voice

    def test_standard_voices_return_none(self):
        for voice in ["alloy", "en-US-Studio-O", "nova"]:
            assert _resolve_chirp_hd_voice_name(voice, {}) is None


class TestTranscriptionChirpGrpcDispatch:
    """Covers the elif branch in litellm.transcription() that routes to gRPC."""

    def test_chirp_model_dispatches_to_grpc_handler(self):
        mock_response = TranscriptionResponse(text="hola mundo")
        mock_response._hidden_params = {"audio_transcription_duration": 2.0}

        mock_handler = MagicMock()
        mock_handler.audio_transcriptions.return_value = mock_response

        with patch(
            "litellm.llms.vertex_ai.audio_transcription.grpc_handler.VertexAIChirpGrpcTranscription",
            return_value=mock_handler,
        ):
            result = litellm.transcription(
                model="vertex_ai/chirp",
                file=io.BytesIO(b"fake audio"),
                vertex_project="test-project",
                vertex_location="us-central1",
                vertex_credentials='{"type": "service_account", "project_id": "test"}',
            )

        mock_handler.audio_transcriptions.assert_called_once()
        assert result.text == "hola mundo"

    def test_chirp_2_model_dispatches_to_grpc_handler(self):
        mock_response = TranscriptionResponse(text="hello world")
        mock_response._hidden_params = {"audio_transcription_duration": 3.0}

        mock_handler = MagicMock()
        mock_handler.audio_transcriptions.return_value = mock_response

        with patch(
            "litellm.llms.vertex_ai.audio_transcription.grpc_handler.VertexAIChirpGrpcTranscription",
            return_value=mock_handler,
        ):
            result = litellm.transcription(
                model="vertex_ai/chirp_2",
                file=io.BytesIO(b"fake audio"),
                vertex_project="test-project",
                vertex_location="us-central1",
                vertex_credentials='{"type": "service_account", "project_id": "test"}',
            )

        mock_handler.audio_transcriptions.assert_called_once()
        assert result.text == "hello world"


class TestSpeechChirpHdGrpcDispatch:
    """Covers the Chirp 3 HD dispatch branch and model normalization in litellm.speech()."""

    def test_chirp_hd_voice_dispatches_to_grpc_handler(self):
        import httpx
        from litellm.types.llms.openai import HttpxBinaryResponseContent

        fake_audio = b"ogg opus audio bytes"
        mock_binary = HttpxBinaryResponseContent(
            httpx.Response(status_code=200, content=fake_audio)
        )

        mock_handler = MagicMock()
        mock_handler.speech.return_value = mock_binary

        with patch(
            "litellm.llms.vertex_ai.text_to_speech.grpc_handler.VertexAIChirpGrpcTextToSpeech",
            return_value=mock_handler,
        ):
            result = litellm.speech(
                model="vertex_ai/tts-1",
                input="Hola mundo",
                voice="es-ES-Chirp3-HD-Charon",
                vertex_project="test-project",
                vertex_location="us-central1",
                vertex_credentials='{"type": "service_account", "project_id": "test"}',
            )

        mock_handler.speech.assert_called_once()
        assert result.content == fake_audio

    def test_non_chirp_hd_voice_does_not_dispatch_to_grpc(self):
        import httpx
        from litellm.types.llms.openai import HttpxBinaryResponseContent

        fake_audio = b"rest audio"
        mock_binary = HttpxBinaryResponseContent(
            httpx.Response(status_code=200, content=fake_audio)
        )

        with (
            patch(
                "litellm.llms.vertex_ai.text_to_speech.grpc_handler.VertexAIChirpGrpcTextToSpeech"
            ) as mock_grpc_cls,
            patch(
                "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
                return_value=MagicMock(
                    status_code=200,
                    json=lambda: {"audioContent": "aGVsbG8="},
                    headers={},
                ),
            ),
        ):
            try:
                litellm.speech(
                    model="vertex_ai/tts-1",
                    input="Hello",
                    voice="en-US-Studio-O",
                    vertex_project="test-project",
                    vertex_location="us-central1",
                    vertex_credentials='{"type": "service_account", "project_id": "test"}',
                )
            except Exception:
                pass

            mock_grpc_cls.assert_not_called()
