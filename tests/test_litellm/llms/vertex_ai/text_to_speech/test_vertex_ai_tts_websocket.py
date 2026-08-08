"""
Unit tests for the Chirp TTS WebSocket-to-gRPC bridge.

All tests use mocked WebSocket and gRPC clients — no real network calls.
"""

import json
import os
import sys
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.vertex_ai.text_to_speech.websocket_handler import (
    ChirpTtsWebSocketSession,
    _send_error,
)
from litellm.llms.vertex_ai.common_utils import VertexAIError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeWebSocket:
    def __init__(self, receive_sequence: list):
        self._recv = iter(receive_sequence)
        self.sent_text: list[str] = []
        self.sent_bytes: list[bytes] = []
        self.closed = False
        self.accepted = False

    async def accept(self) -> None:
        self.accepted = True

    async def receive(self) -> dict:
        try:
            msg = next(self._recv)
            if "type" not in msg:
                msg = {"type": "websocket.receive", **msg}
            return msg
        except StopIteration:
            return {"type": "websocket.disconnect"}

    async def receive_text(self) -> str:
        msg = await self.receive()
        return msg.get("text", "{}")

    async def send_text(self, text: str) -> None:
        self.sent_text.append(text)

    async def send_bytes(self, data: bytes) -> None:
        self.sent_bytes.append(data)

    async def close(self) -> None:
        self.closed = True


def _make_audio_response(audio: bytes) -> Any:
    r = MagicMock()
    r.audio_content = audio
    return r


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestChirpTtsWebSocketSession:

    def _config_msg(self, **overrides) -> dict:
        return {
            "text": json.dumps({
                "voice": "es-ES-Chirp3-HD-Charon",
                "encoding": "OGG_OPUS",
                "vertex_project": "test-project",
                "vertex_location": "us-central1",
                **overrides,
            })
        }

    def _make_mock_tts(self, audio_chunks: list[bytes]):
        async def _stream() -> AsyncIterator:
            for chunk in audio_chunks:
                yield _make_audio_response(chunk)

        client = MagicMock()
        client.streaming_synthesize = AsyncMock(return_value=_stream())
        tts = MagicMock()
        tts.TextToSpeechAsyncClient.return_value = client
        return tts

    @pytest.mark.asyncio
    async def test_audio_chunks_forwarded_as_binary(self):
        audio = [b"chunk1" * 100, b"chunk2" * 100, b"chunk3" * 50]
        tts = self._make_mock_tts(audio)

        ws = _FakeWebSocket([
            self._config_msg(),
            {"text": json.dumps({"type": "text", "content": "Hola, ¿cómo estás?"})},
            {"text": json.dumps({"type": "text", "content": " Estoy muy bien, gracias."})},
            {"text": json.dumps({"type": "end"})},
        ])

        class StubbedSession(ChirpTtsWebSocketSession):
            def load_auth(self, credentials, project_id):
                return MagicMock(), "test-project"
            @staticmethod
            def safe_get_vertex_ai_credentials(p): return None
            @staticmethod
            def safe_get_vertex_ai_project(p): return "test-project"
            @staticmethod
            def safe_get_vertex_ai_location(p): return "us-central1"

        with patch(
            "litellm.llms.vertex_ai.text_to_speech.websocket_handler._import_texttospeech",
            return_value=tts,
        ):
            await StubbedSession()._session(ws, user_api_key_dict=MagicMock())

        assert ws.sent_bytes == audio
        done_msgs = [json.loads(t) for t in ws.sent_text if '"done"' in t]
        assert done_msgs
        done = done_msgs[0]
        assert done["chunks"] == 3
        assert done["total_bytes"] == sum(len(c) for c in audio)
        assert done["chars"] == len("Hola, ¿cómo estás?") + len(" Estoy muy bien, gracias.")

    @pytest.mark.asyncio
    async def test_unsupported_encoding_falls_back_to_ogg_opus(self):
        """LINEAR16 is not supported in streaming — must silently use OGG_OPUS."""
        tts = self._make_mock_tts([b"audio"])
        ws = _FakeWebSocket([
            self._config_msg(encoding="LINEAR16"),
            {"text": json.dumps({"type": "text", "content": "test"})},
            {"text": json.dumps({"type": "end"})},
        ])

        captured_config: list = []

        class StubbedSession(ChirpTtsWebSocketSession):
            def load_auth(self, credentials, project_id):
                return MagicMock(), "test-project"
            @staticmethod
            def safe_get_vertex_ai_credentials(p): return None
            @staticmethod
            def safe_get_vertex_ai_project(p): return "test-project"
            @staticmethod
            def safe_get_vertex_ai_location(p): return "us-central1"

        with patch(
            "litellm.llms.vertex_ai.text_to_speech.websocket_handler._import_texttospeech",
            return_value=tts,
        ) as mock_tts:
            mock_tts.return_value = tts
            # Capture what encoding was passed to StreamingAudioConfig
            original_sac = tts.StreamingAudioConfig
            def capturing_sac(**kwargs):
                captured_config.append(kwargs.get("audio_encoding"))
                return original_sac(**kwargs)
            tts.StreamingAudioConfig = capturing_sac
            await StubbedSession()._session(ws, user_api_key_dict=MagicMock())

        # OGG_OPUS enum should have been used, not LINEAR16
        assert captured_config  # at least one call
        assert captured_config[0] == tts.AudioEncoding.OGG_OPUS

    @pytest.mark.asyncio
    async def test_client_disconnect_stops_session(self):
        import asyncio

        async def _blocking_stream():
            yield _make_audio_response(b"first_chunk")
            await asyncio.sleep(999)

        client = MagicMock()
        client.streaming_synthesize = AsyncMock(return_value=_blocking_stream())
        tts = MagicMock()
        tts.TextToSpeechAsyncClient.return_value = client

        ws = _FakeWebSocket([
            self._config_msg(),
            {"type": "websocket.disconnect"},
        ])

        class StubbedSession(ChirpTtsWebSocketSession):
            def load_auth(self, credentials, project_id):
                return MagicMock(), "test-project"
            @staticmethod
            def safe_get_vertex_ai_credentials(p): return None
            @staticmethod
            def safe_get_vertex_ai_project(p): return "test-project"
            @staticmethod
            def safe_get_vertex_ai_location(p): return "us-central1"

        with patch(
            "litellm.llms.vertex_ai.text_to_speech.websocket_handler._import_texttospeech",
            return_value=tts,
        ):
            await asyncio.wait_for(
                StubbedSession()._session(ws, user_api_key_dict=MagicMock()),
                timeout=2.0,
            )

    @pytest.mark.asyncio
    async def test_gRPC_error_propagates_as_error_message(self):
        tts = MagicMock()
        client = MagicMock()
        client.streaming_synthesize = AsyncMock(side_effect=RuntimeError("gRPC unavailable"))
        tts.TextToSpeechAsyncClient.return_value = client

        ws = _FakeWebSocket([self._config_msg()])

        class StubbedSession(ChirpTtsWebSocketSession):
            def load_auth(self, credentials, project_id):
                return MagicMock(), "test-project"
            @staticmethod
            def safe_get_vertex_ai_credentials(p): return None
            @staticmethod
            def safe_get_vertex_ai_project(p): return "test-project"
            @staticmethod
            def safe_get_vertex_ai_location(p): return "us-central1"

        with patch(
            "litellm.llms.vertex_ai.text_to_speech.websocket_handler._import_texttospeech",
            return_value=tts,
        ):
            await StubbedSession().run(ws, user_api_key_dict=MagicMock())

        error_msgs = [json.loads(t) for t in ws.sent_text if '"error"' in t]
        assert error_msgs
        assert "gRPC unavailable" in error_msgs[0]["message"]

    @pytest.mark.asyncio
    async def test_missing_sdk_sends_error(self):
        ws = _FakeWebSocket([
            {"text": json.dumps({"vertex_project": "p", "vertex_location": "us-central1"})}
        ])

        class StubbedSession(ChirpTtsWebSocketSession):
            def load_auth(self, credentials, project_id):
                return MagicMock(), "test-project"
            @staticmethod
            def safe_get_vertex_ai_credentials(p): return None
            @staticmethod
            def safe_get_vertex_ai_project(p): return "test-project"
            @staticmethod
            def safe_get_vertex_ai_location(p): return "us-central1"

        with patch(
            "litellm.llms.vertex_ai.text_to_speech.websocket_handler._import_texttospeech",
            side_effect=VertexAIError(status_code=500, message="tts-vertex-chirp-grpc"),
        ):
            await StubbedSession().run(ws, user_api_key_dict=MagicMock())

        error_msgs = [json.loads(t) for t in ws.sent_text if '"error"' in t]
        assert error_msgs
        assert "tts-vertex-chirp-grpc" in error_msgs[0]["message"]

    @pytest.mark.asyncio
    async def test_messages_without_type_are_ignored(self):
        """Malformed messages should not crash the session."""
        tts = self._make_mock_tts([b"audio"])
        ws = _FakeWebSocket([
            self._config_msg(),
            {"text": "not json at all"},
            {"text": json.dumps({"type": "text", "content": "hello"})},
            {"text": json.dumps({"type": "end"})},
        ])

        class StubbedSession(ChirpTtsWebSocketSession):
            def load_auth(self, credentials, project_id):
                return MagicMock(), "test-project"
            @staticmethod
            def safe_get_vertex_ai_credentials(p): return None
            @staticmethod
            def safe_get_vertex_ai_project(p): return "test-project"
            @staticmethod
            def safe_get_vertex_ai_location(p): return "us-central1"

        with patch(
            "litellm.llms.vertex_ai.text_to_speech.websocket_handler._import_texttospeech",
            return_value=tts,
        ):
            await StubbedSession()._session(ws, user_api_key_dict=MagicMock())

        assert ws.sent_bytes == [b"audio"]


class TestSendError:
    @pytest.mark.asyncio
    async def test_sends_error_json(self):
        ws = _FakeWebSocket([])
        await _send_error(ws, "synthesis failed")
        assert ws.sent_text
        msg = json.loads(ws.sent_text[0])
        assert msg["type"] == "error"
        assert "synthesis failed" in msg["message"]

    @pytest.mark.asyncio
    async def test_swallows_send_exception(self):
        ws = MagicMock()
        ws.send_text = AsyncMock(side_effect=RuntimeError("closed"))
        await _send_error(ws, "test")
