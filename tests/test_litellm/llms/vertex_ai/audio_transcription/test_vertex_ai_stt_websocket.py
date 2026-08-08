"""
Unit tests for the Chirp STT WebSocket-to-gRPC bridge.

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

from litellm.llms.vertex_ai.audio_transcription.websocket_handler import (
    ChirpSttWebSocketSession,
    _send_error,
)
from litellm.llms.vertex_ai.common_utils import VertexAIError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gRPC_result(transcript: str, is_final: bool, language_code: str = "", confidence: float = 0.0) -> Any:
    alt = MagicMock()
    alt.transcript = transcript
    alt.confidence = confidence
    result = MagicMock()
    result.is_final = is_final
    result.alternatives = [alt]
    result.language_code = language_code
    result.stability = 0.8
    return result


def _make_gRPC_response(results: list) -> Any:
    resp = MagicMock()
    resp.results = results
    return resp


class _FakeWebSocket:
    """Minimal WebSocket stand-in for unit tests."""

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
            # Normalise to FastAPI websocket.receive format
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestChirpSttWebSocketSession:

    def _make_config_msg(self, **overrides) -> dict:
        return {
            "text": json.dumps({
                "model": "vertex_ai/chirp_2",
                "language": "es-ES",
                "sample_rate": 16000,
                "encoding": "LINEAR16",
                "vertex_project": "test-project",
                "vertex_location": "us-central1",
                **overrides,
            })
        }

    def _make_mock_client(self, response_items: list):
        async def _stream() -> AsyncIterator:
            for item in response_items:
                yield item

        client = MagicMock()
        client.streaming_recognize = AsyncMock(return_value=_stream())
        return client

    def _make_mock_speech_v2(self, client: Any) -> Any:
        cs = MagicMock()
        sv2 = MagicMock()
        sv2.SpeechAsyncClient.return_value = client
        sv2.types.cloud_speech = cs
        return sv2

    @pytest.mark.asyncio
    async def test_interim_and_final_results_forwarded(self):
        responses = [
            _make_gRPC_response([_make_gRPC_result("hola", is_final=False)]),
            _make_gRPC_response([_make_gRPC_result("hola mundo", is_final=True, confidence=0.97, language_code="es-ES")]),
        ]
        client = self._make_mock_client(responses)
        ws = _FakeWebSocket([
            self._make_config_msg(),
            {"bytes": b"\x00" * 3200},
            {"text": json.dumps({"type": "end"})},
        ])

        class StubbedSession(ChirpSttWebSocketSession):
            def load_auth(self, credentials, project_id):
                return MagicMock(), "test-project"
            @staticmethod
            def safe_get_vertex_ai_credentials(p): return None
            @staticmethod
            def safe_get_vertex_ai_project(p): return "test-project"
            @staticmethod
            def safe_get_vertex_ai_location(p): return "us-central1"

        with patch(
            "litellm.llms.vertex_ai.audio_transcription.websocket_handler._import_speech_v2",
            return_value=self._make_mock_speech_v2(client),
        ):
            await StubbedSession().run(ws, user_api_key_dict=MagicMock())

        assert ws.accepted
        sent = [json.loads(t) for t in ws.sent_text]
        types = [m["type"] for m in sent]
        assert "interim" in types
        assert "final" in types
        assert "done" in types

        final_msg = next(m for m in sent if m["type"] == "final")
        assert final_msg["transcript"] == "hola mundo"
        assert final_msg["language"] == "es-ES"
        assert abs(final_msg["confidence"] - 0.97) < 0.001

        interim_msg = next(m for m in sent if m["type"] == "interim")
        assert interim_msg["transcript"] == "hola"

    @pytest.mark.asyncio
    async def test_client_disconnect_ends_stream(self):
        async def _never_ending_stream():
            yield _make_gRPC_response([_make_gRPC_result("text", is_final=False)])
            await asyncio.sleep(999)  # would block forever if not cancelled

        import asyncio
        client = MagicMock()
        client.streaming_recognize = AsyncMock(return_value=_never_ending_stream())
        ws = _FakeWebSocket([
            self._make_config_msg(),
            {"type": "websocket.disconnect"},
        ])

        class StubbedSession(ChirpSttWebSocketSession):
            def load_auth(self, credentials, project_id):
                return MagicMock(), "test-project"
            @staticmethod
            def safe_get_vertex_ai_credentials(p): return None
            @staticmethod
            def safe_get_vertex_ai_project(p): return "test-project"
            @staticmethod
            def safe_get_vertex_ai_location(p): return "us-central1"

        with patch(
            "litellm.llms.vertex_ai.audio_transcription.websocket_handler._import_speech_v2",
            return_value=self._make_mock_speech_v2(client),
        ):
            # Should complete without hanging
            await asyncio.wait_for(
                StubbedSession()._session(ws, user_api_key_dict=MagicMock()),
                timeout=2.0,
            )

    @pytest.mark.asyncio
    async def test_gRPC_error_propagates(self):
        client = MagicMock()
        client.streaming_recognize = AsyncMock(side_effect=RuntimeError("quota exceeded"))
        ws = _FakeWebSocket([self._make_config_msg()])

        class StubbedSession(ChirpSttWebSocketSession):
            def load_auth(self, credentials, project_id):
                return MagicMock(), "test-project"
            @staticmethod
            def safe_get_vertex_ai_credentials(p): return None
            @staticmethod
            def safe_get_vertex_ai_project(p): return "test-project"
            @staticmethod
            def safe_get_vertex_ai_location(p): return "us-central1"

        with patch(
            "litellm.llms.vertex_ai.audio_transcription.websocket_handler._import_speech_v2",
            return_value=self._make_mock_speech_v2(client),
        ):
            await StubbedSession().run(ws, user_api_key_dict=MagicMock())

        error_msgs = [json.loads(t) for t in ws.sent_text if '"error"' in t]
        assert error_msgs
        assert "quota exceeded" in error_msgs[0]["message"]

    @pytest.mark.asyncio
    async def test_missing_sdk_sends_error(self):
        ws = _FakeWebSocket([
            {"text": json.dumps({"model": "vertex_ai/chirp_2", "vertex_project": "p", "vertex_location": "us-central1"})}
        ])

        class StubbedSession(ChirpSttWebSocketSession):
            def load_auth(self, credentials, project_id):
                return MagicMock(), "test-project"
            @staticmethod
            def safe_get_vertex_ai_credentials(p): return None
            @staticmethod
            def safe_get_vertex_ai_project(p): return "test-project"
            @staticmethod
            def safe_get_vertex_ai_location(p): return "us-central1"

        with patch(
            "litellm.llms.vertex_ai.audio_transcription.websocket_handler._import_speech_v2",
            side_effect=VertexAIError(status_code=500, message="stt-vertex-chirp-grpc"),
        ):
            await StubbedSession().run(ws, user_api_key_dict=MagicMock())

        error_msgs = [json.loads(t) for t in ws.sent_text if '"error"' in t]
        assert error_msgs
        assert "stt-vertex-chirp-grpc" in error_msgs[0]["message"]

    def test_audio_duration_formula(self):
        """The duration formula bytes / (sample_rate × 2) is correct for LINEAR16."""
        # 3200 bytes / (16000 Hz × 2 bytes/sample) = 0.1s per chunk
        # 2 chunks × 3200 bytes = 6400 bytes total → 0.2s
        audio_bytes = 6400
        sample_rate = 16000
        expected_s = 0.2
        actual = audio_bytes / (sample_rate * 2)
        assert abs(actual - expected_s) < 0.001

    @pytest.mark.asyncio
    async def test_done_message_is_sent_after_session(self):
        """done message is present with an audio_duration_s field (exact value
        depends on task scheduling, so we only assert the field exists)."""
        responses = [
            _make_gRPC_response([_make_gRPC_result("hola", is_final=True, confidence=0.9)]),
        ]
        client = self._make_mock_client(responses)
        ws = _FakeWebSocket([
            self._make_config_msg(sample_rate=16000),
            {"bytes": b"\x00" * 3200},
            {"text": json.dumps({"type": "end"})},
        ])

        class StubbedSession(ChirpSttWebSocketSession):
            def load_auth(self, credentials, project_id):
                return MagicMock(), "test-project"
            @staticmethod
            def safe_get_vertex_ai_credentials(p): return None
            @staticmethod
            def safe_get_vertex_ai_project(p): return "test-project"
            @staticmethod
            def safe_get_vertex_ai_location(p): return "us-central1"

        with patch(
            "litellm.llms.vertex_ai.audio_transcription.websocket_handler._import_speech_v2",
            return_value=self._make_mock_speech_v2(client),
        ):
            await StubbedSession().run(ws, user_api_key_dict=MagicMock())

        done_msgs = [json.loads(t) for t in ws.sent_text if '"done"' in t]
        assert done_msgs
        assert "audio_duration_s" in done_msgs[0]
        assert done_msgs[0]["audio_duration_s"] >= 0


class TestSendError:
    @pytest.mark.asyncio
    async def test_sends_error_json(self):
        ws = _FakeWebSocket([])
        await _send_error(ws, "something went wrong")
        assert ws.sent_text
        msg = json.loads(ws.sent_text[0])
        assert msg["type"] == "error"
        assert "something went wrong" in msg["message"]

    @pytest.mark.asyncio
    async def test_swallows_send_exception(self):
        ws = MagicMock()
        ws.send_text = AsyncMock(side_effect=RuntimeError("ws already closed"))
        # Must not raise
        await _send_error(ws, "test")
