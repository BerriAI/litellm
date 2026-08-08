import os
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.vertex_ai.audio_transcription.grpc_handler import (
    VertexAIChirpGrpcTranscription,
    _build_grpc_request_config,
    _collect_transcripts,
)
from litellm.llms.vertex_ai.audio_transcription.transformation import (
    is_chirp_grpc_model,
    vertex_model_to_speech_v2_name,
)
from litellm.llms.vertex_ai.common_utils import VertexAIError
from litellm.types.llms.vertex_ai_speech_to_text import ChirpGrpcRequestConfig
from litellm.types.utils import TranscriptionResponse


class TestIsChirpGrpcModel:
    @pytest.mark.parametrize(
        "model,expected",
        [
            ("chirp", True),
            ("vertex_ai/chirp", True),
            ("chirp-2", True),
            ("chirp_2", True),
            ("vertex_ai/chirp-2", True),
            ("chirp-telephony", True),
            ("chirp_telephony", True),
            # chirp-3-hd is a TTS voice name, not an STT model
            ("chirp-3-hd", False),
            ("chirp_3_hd", False),
            ("chirp_3", False),
            ("vertex_ai/chirp_3", False),
            ("whisper-1", False),
            ("", False),
        ],
    )
    def test_model_detection(self, model, expected):
        assert is_chirp_grpc_model(model) is expected


class TestVertexModelToSpeechV2Name:
    @pytest.mark.parametrize(
        "model,expected",
        [
            ("chirp", "chirp"),
            ("vertex_ai/chirp", "chirp"),
            ("chirp-2", "chirp_2"),
            ("chirp_2", "chirp_2"),
            ("vertex_ai/chirp-2", "chirp_2"),
            ("chirp-telephony", "chirp_telephony"),
            ("unknown-model", "unknown-model"),
        ],
    )
    def test_name_mapping(self, model, expected):
        assert vertex_model_to_speech_v2_name(model) == expected


class TestBuildGrpcRequestConfig:
    def test_recognizer_path_always_uses_global(self):
        config = _build_grpc_request_config(
            model="chirp-3-hd",
            project_id="my-project",
            location="us-central1",
            language="en-US",
        )
        assert config.recognizer == "projects/my-project/locations/global/recognizers/_"

    def test_model_name_is_mapped(self):
        config = _build_grpc_request_config(
            model="chirp-2",
            project_id="my-project",
            location="us-central1",
            language=None,
        )
        assert config.model_name == "chirp_2"

    def test_language_none_defaults_to_en_us(self):
        config = _build_grpc_request_config(
            model="chirp-3-hd",
            project_id="my-project",
            location="us-central1",
            language=None,
        )
        assert config.language_codes == ("en-US",)

    def test_explicit_language_is_preserved(self):
        config = _build_grpc_request_config(
            model="chirp-3-hd",
            project_id="my-project",
            location="us-central1",
            language="es-ES",
        )
        assert config.language_codes == ("es-ES",)

    def test_returns_frozen_dataclass(self):
        config = _build_grpc_request_config(
            model="chirp-3-hd",
            project_id="my-project",
            location="us-central1",
            language=None,
        )
        assert isinstance(config, ChirpGrpcRequestConfig)
        with pytest.raises((AttributeError, TypeError)):
            config.model_name = "other"  # type: ignore[misc]


class TestCollectTranscripts:
    def _make_result(self, transcript: str, is_final: bool, language_code: str = "") -> Any:
        alternative = MagicMock()
        alternative.transcript = transcript
        result = MagicMock()
        result.is_final = is_final
        result.alternatives = [alternative]
        result.language_code = language_code
        return result

    def _make_response(self, results: list) -> Any:
        response = MagicMock()
        response.results = results
        return response

    async def _stream(self, responses):
        for response in responses:
            yield response

    @pytest.mark.asyncio
    async def test_collects_only_final_results(self):
        responses = [
            self._make_response([self._make_result("Hello", is_final=True)]),
            self._make_response([self._make_result("interim", is_final=False)]),
            self._make_response([self._make_result("world", is_final=True)]),
        ]
        transcript, _ = await _collect_transcripts(self._stream(responses))
        assert transcript == "Hello world"

    @pytest.mark.asyncio
    async def test_empty_stream_returns_empty_string(self):
        transcript, detected_language = await _collect_transcripts(self._stream([]))
        assert transcript == ""
        assert detected_language is None

    @pytest.mark.asyncio
    async def test_detects_language_from_first_final_result(self):
        responses = [
            self._make_response([self._make_result("Hola", is_final=True, language_code="es-ES")]),
        ]
        _, detected_language = await _collect_transcripts(self._stream(responses))
        assert detected_language == "es-ES"

    @pytest.mark.asyncio
    async def test_results_with_no_alternatives_are_skipped(self):
        result_no_alt = MagicMock()
        result_no_alt.is_final = True
        result_no_alt.alternatives = []
        result_no_alt.language_code = ""

        responses = [
            self._make_response([result_no_alt, self._make_result("Real text", is_final=True)]),
        ]
        transcript, _ = await _collect_transcripts(self._stream(responses))
        assert transcript == "Real text"

    @pytest.mark.asyncio
    async def test_multiple_final_results_are_joined_with_space(self):
        responses = [
            self._make_response([
                self._make_result("First sentence.", is_final=True),
                self._make_result("Second sentence.", is_final=True),
            ]),
        ]
        transcript, _ = await _collect_transcripts(self._stream(responses))
        assert transcript == "First sentence. Second sentence."


class TestVertexAIChirpGrpcTranscription:

    def _make_mock_speech_v2(self, transcript: str = "Hello world", language_code: str = "en-US"):
        alternative = MagicMock()
        alternative.transcript = transcript

        result = MagicMock()
        result.is_final = True
        result.alternatives = [alternative]
        result.language_code = language_code

        response = MagicMock()
        response.results = [result]

        async def _response_stream():
            yield response

        client = MagicMock()
        client.streaming_recognize = AsyncMock(return_value=_response_stream())

        speech_v2 = MagicMock()
        speech_v2.SpeechAsyncClient.return_value = client
        cs = MagicMock()
        speech_v2.types.cloud_speech = cs
        return speech_v2

    @pytest.mark.asyncio
    async def test_run_async_returns_transcription_response(self):
        class StubbedHandler(VertexAIChirpGrpcTranscription):
            def load_auth(self, credentials, project_id):
                return MagicMock(), "test-project"

            @staticmethod
            def safe_get_vertex_ai_credentials(litellm_params):
                return None

            @staticmethod
            def safe_get_vertex_ai_project(litellm_params):
                return "test-project"

            @staticmethod
            def safe_get_vertex_ai_location(litellm_params):
                return "us-central1"

        speech_v2 = self._make_mock_speech_v2(transcript="Hello world", language_code="en-US")

        with patch(
            "litellm.llms.vertex_ai.audio_transcription.grpc_handler._import_speech_v2",
            return_value=speech_v2,
        ):
            logging_obj = MagicMock()
            model_response = TranscriptionResponse(text="")
            result = await StubbedHandler()._run_async(
                model="chirp-3-hd",
                audio_file=b"fake-audio",
                optional_params={},
                litellm_params={"vertex_location": "us-central1"},
                model_response=model_response,
                timeout=30.0,
                logging_obj=logging_obj,
                api_key=None,
                api_base=None,
            )

        assert isinstance(result, TranscriptionResponse)
        assert result.text == "Hello world"
        assert result["language"] == "en-US"

    @pytest.mark.asyncio
    async def test_api_error_is_wrapped_as_vertex_ai_error(self):
        class StubbedHandler(VertexAIChirpGrpcTranscription):
            def load_auth(self, credentials, project_id):
                return MagicMock(), "test-project"

            @staticmethod
            def safe_get_vertex_ai_credentials(litellm_params):
                return None

            @staticmethod
            def safe_get_vertex_ai_project(litellm_params):
                return "test-project"

            @staticmethod
            def safe_get_vertex_ai_location(litellm_params):
                return "us-central1"

        client = MagicMock()
        client.streaming_recognize = AsyncMock(side_effect=RuntimeError("network failure"))
        speech_v2 = MagicMock()
        speech_v2.SpeechAsyncClient.return_value = client
        speech_v2.types.cloud_speech = MagicMock()

        with patch(
            "litellm.llms.vertex_ai.audio_transcription.grpc_handler._import_speech_v2",
            return_value=speech_v2,
        ):
            with pytest.raises(VertexAIError) as exc_info:
                await StubbedHandler()._run_async(
                    model="chirp-3-hd",
                    audio_file=b"fake-audio",
                    optional_params={},
                    litellm_params={"vertex_location": "us-central1"},
                    model_response=TranscriptionResponse(text=""),
                    timeout=30.0,
                    logging_obj=MagicMock(),
                    api_key=None,
                    api_base=None,
                )
        assert "network failure" in str(exc_info.value)

    def test_missing_sdk_raises_vertex_ai_error_with_install_hint(self):
        with patch.dict("sys.modules", {"google.cloud.speech_v2": None, "google.cloud": None}):
            with pytest.raises(VertexAIError) as exc_info:
                from litellm.llms.vertex_ai.audio_transcription.grpc_handler import _import_speech_v2
                _import_speech_v2()
        assert "litellm[stt-vertex-chirp-grpc]" in str(exc_info.value)

    def test_resolve_location_invalid_raises_vertex_ai_error(self):
        handler = VertexAIChirpGrpcTranscription()
        with pytest.raises(VertexAIError):
            handler._resolve_location({"vertex_location": "INVALID/location"})

    def test_resolve_location_defaults_to_us(self):
        handler = VertexAIChirpGrpcTranscription()
        location = handler._resolve_location({})
        assert location == "us"
