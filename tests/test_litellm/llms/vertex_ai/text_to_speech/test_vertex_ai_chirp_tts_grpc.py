import os
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.vertex_ai.text_to_speech.grpc_handler import (
    VertexAIChirpGrpcTextToSpeech,
    _collect_audio,
    _import_texttospeech,
    _parse_speaking_rate,
    _resolve_timeout,
)
from litellm.llms.vertex_ai.text_to_speech.transformation import (
    extract_language_code_from_voice,
    is_chirp_hd_voice,
)
from litellm.llms.vertex_ai.common_utils import VertexAIError


class TestIsChirpHdVoice:
    @pytest.mark.parametrize(
        "voice,expected",
        [
            ("en-US-Chirp3-HD-Charon", True),
            ("es-ES-Chirp3-HD-Kore", True),
            ("en-US-Chirp3-HD-Aoede", True),
            ("EN-US-CHIRP3-HD-Charon", True),
            ({"name": "en-US-Chirp3-HD-Charon", "languageCode": "en-US"}, True),
            ({"name": "en-US-Studio-O"}, False),
            ("en-US-Studio-O", False),
            ("en-US-Chirp3-Charon", False),
            ("en-US-Chirp-D", False),
            ("alloy", False),
            ("", False),
            (None, False),
        ],
    )
    def test_detection(self, voice, expected):
        assert is_chirp_hd_voice(voice) is expected


class TestExtractLanguageCode:
    @pytest.mark.parametrize(
        "voice_name,expected_language_code",
        [
            ("en-US-Chirp3-HD-Charon", "en-US"),
            ("es-ES-Chirp3-HD-Kore", "es-ES"),
            ("fr-FR-Chirp3-HD-Aoede", "fr-FR"),
            ("de-DE-Chirp3-HD-Charon", "de-DE"),
            ("en", "en-US"),
        ],
    )
    def test_extraction(self, voice_name, expected_language_code):
        assert extract_language_code_from_voice(voice_name) == expected_language_code


class TestParseExpectedRate:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (None, None),
            (1.0, None),
            ("1", None),
            (1.5, 1.5),
            ("0.75", 0.75),
            ("invalid", None),
            (0, 0.0),
        ],
    )
    def test_parse(self, value, expected):
        assert _parse_speaking_rate(value) == expected


class TestResolveTimeout:
    def test_none_returns_none(self):
        assert _resolve_timeout(None) is None

    def test_float_returned_as_is(self):
        assert _resolve_timeout(30.0) == 30.0

    def test_httpx_timeout_returns_read(self):
        t = httpx.Timeout(read=45.0, connect=5.0, write=5.0, pool=5.0)
        assert _resolve_timeout(t) == 45.0


class TestCollectAudio:
    async def _stream(self, items):
        for item in items:
            yield item

    @pytest.mark.asyncio
    async def test_concatenates_chunks(self):
        r1, r2 = MagicMock(), MagicMock()
        r1.audio_content = b"hello"
        r2.audio_content = b" world"
        result = await _collect_audio(self._stream([r1, r2]))
        assert result == b"hello world"

    @pytest.mark.asyncio
    async def test_empty_stream_returns_empty_bytes(self):
        result = await _collect_audio(self._stream([]))
        assert result == b""

    @pytest.mark.asyncio
    async def test_skips_empty_audio_content(self):
        r1, r2, r3 = MagicMock(), MagicMock(), MagicMock()
        r1.audio_content = b"audio"
        r2.audio_content = b""
        r3.audio_content = b"_more"
        result = await _collect_audio(self._stream([r1, r2, r3]))
        assert result == b"audio_more"


class TestVertexAIChirpGrpcTextToSpeech:

    def _make_mock_tts(self, audio_chunks: list[bytes]) -> Any:
        responses = []
        for chunk in audio_chunks:
            r = MagicMock()
            r.audio_content = chunk
            responses.append(r)

        async def _response_stream():
            for r in responses:
                yield r

        client = MagicMock()
        client.streaming_synthesize = AsyncMock(return_value=_response_stream())

        tts_module = MagicMock()
        tts_module.TextToSpeechAsyncClient.return_value = client
        return tts_module

    @pytest.mark.asyncio
    async def test_run_async_returns_binary_response(self):
        class StubbedHandler(VertexAIChirpGrpcTextToSpeech):
            def load_auth(self, credentials, project_id):
                return MagicMock(), "test-project"

            @staticmethod
            def safe_get_vertex_ai_credentials(litellm_params):
                return None

            @staticmethod
            def safe_get_vertex_ai_project(litellm_params):
                return "test-project"

        tts_module = self._make_mock_tts([b"audio_chunk_1", b"audio_chunk_2"])

        with patch(
            "litellm.llms.vertex_ai.text_to_speech.grpc_handler._import_texttospeech",
            return_value=tts_module,
        ):
            from litellm.types.llms.openai import HttpxBinaryResponseContent

            result = await StubbedHandler()._run_async(
                input="Hello, world!",
                voice_name="en-US-Chirp3-HD-Charon",
                optional_params={"audioEncoding": "LINEAR16"},
                litellm_params={},
                logging_obj=MagicMock(),
                timeout=30.0,
            )

        assert isinstance(result, HttpxBinaryResponseContent)
        assert result.content == b"audio_chunk_1audio_chunk_2"

    @pytest.mark.asyncio
    async def test_api_error_wrapped_as_vertex_ai_error(self):
        class StubbedHandler(VertexAIChirpGrpcTextToSpeech):
            def load_auth(self, credentials, project_id):
                return MagicMock(), "test-project"

            @staticmethod
            def safe_get_vertex_ai_credentials(litellm_params):
                return None

            @staticmethod
            def safe_get_vertex_ai_project(litellm_params):
                return "test-project"

        client = MagicMock()
        client.streaming_synthesize = AsyncMock(side_effect=RuntimeError("quota exceeded"))
        tts_module = MagicMock()
        tts_module.TextToSpeechAsyncClient.return_value = client

        with patch(
            "litellm.llms.vertex_ai.text_to_speech.grpc_handler._import_texttospeech",
            return_value=tts_module,
        ):
            with pytest.raises(VertexAIError) as exc_info:
                await StubbedHandler()._run_async(
                    input="Hello",
                    voice_name="en-US-Chirp3-HD-Charon",
                    optional_params={},
                    litellm_params={},
                    logging_obj=MagicMock(),
                    timeout=None,
                )
        assert "quota exceeded" in str(exc_info.value)

    def test_missing_sdk_raises_install_hint(self):
        with patch.dict("sys.modules", {"google.cloud.texttospeech": None, "google.cloud": None}):
            with pytest.raises(VertexAIError) as exc_info:
                _import_texttospeech()
        assert "litellm[tts-vertex-chirp-grpc]" in str(exc_info.value)
