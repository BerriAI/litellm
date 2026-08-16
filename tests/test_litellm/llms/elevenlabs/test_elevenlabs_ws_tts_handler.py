"""
Unit tests for litellm/llms/elevenlabs/text_to_speech/ws_handler.py.

These tests verify the WebSocket URL builder and the relay helpers without
establishing real network connections.
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.llms.elevenlabs.text_to_speech.ws_handler import (
    _relay_client_to_upstream,
    _relay_upstream_to_client,
    build_elevenlabs_ws_url,
    stream_input_tts,
)


class TestBuildElevenLabsWsUrl:
    def test_default_base_url(self) -> None:
        url = build_elevenlabs_ws_url(
            model="eleven_multilingual_v2",
            voice_id="21m00Tcm4TlvDq8ikWAM",
            output_format="mp3_44100_128",
        )
        assert url.startswith("wss://api.elevenlabs.io/v1/text-to-speech/")
        assert "?model_id=eleven_multilingual_v2&output_format=mp3_44100_128" in url
        assert "21m00Tcm4TlvDq8ikWAM" in url

    def test_custom_api_base_https_converted_to_wss(self) -> None:
        url = build_elevenlabs_ws_url(
            model="eleven_multilingual_v2",
            voice_id="voice123",
            output_format="pcm_44100",
            api_base="https://custom.elevenlabs.io",
        )
        assert url.startswith("wss://custom.elevenlabs.io/")
        assert "http" not in url.split("?")[0]

    def test_custom_api_base_http_converted_to_ws(self) -> None:
        url = build_elevenlabs_ws_url(
            model="eleven_multilingual_v2",
            voice_id="voice123",
            output_format="pcm_44100",
            api_base="http://localhost:8080",
        )
        assert url.startswith("ws://localhost:8080/")

    def test_voice_id_with_special_chars_is_encoded(self) -> None:
        url = build_elevenlabs_ws_url(
            model="eleven_multilingual_v2",
            voice_id="voice with spaces",
            output_format="mp3_44100_128",
        )
        assert " " not in url
        assert "voice%20with%20spaces" in url

    def test_query_params_include_model_and_format(self) -> None:
        url = build_elevenlabs_ws_url(
            model="eleven_turbo_v2",
            voice_id="abc",
            output_format="ulaw_8000",
        )
        assert "model_id=eleven_turbo_v2" in url
        assert "output_format=ulaw_8000" in url

    def test_trailing_slash_stripped_from_base(self) -> None:
        url = build_elevenlabs_ws_url(
            model="eleven_multilingual_v2",
            voice_id="abc",
            output_format="mp3_44100_128",
            api_base="wss://api.elevenlabs.io/",
        )
        assert "//" not in url.replace("wss://", "")


class TestRelayClientToUpstream:
    @pytest.mark.asyncio
    async def test_forwards_text_chunks_and_counts_chars(self) -> None:
        chunks = [
            json.dumps({"text": "Hello, "}),
            json.dumps({"text": "world! "}),
            json.dumps({"text": ""}),
        ]
        client_ws = MagicMock()
        client_ws.iter_text = self._make_iter(chunks)

        upstream = AsyncMock()
        upstream.send = AsyncMock()

        result = await _relay_client_to_upstream(client_ws, upstream)

        assert result == (7, 7, 0)
        assert upstream.send.call_count == 3

    @pytest.mark.asyncio
    async def test_stops_on_empty_text_eos(self) -> None:
        chunks = [
            json.dumps({"text": "First "}),
            json.dumps({"text": ""}),
            json.dumps({"text": "Should not be sent"}),
        ]
        client_ws = MagicMock()
        client_ws.iter_text = self._make_iter(chunks)

        upstream = AsyncMock()
        upstream.send = AsyncMock()

        result = await _relay_client_to_upstream(client_ws, upstream)

        assert sum(result) == 6
        assert upstream.send.call_count == 2

    @pytest.mark.asyncio
    async def test_passes_flush_and_try_trigger_fields(self) -> None:
        msg = {"text": "Go! ", "flush": True, "try_trigger_generation": True}
        chunks = [json.dumps(msg), json.dumps({"text": ""})]
        client_ws = MagicMock()
        client_ws.iter_text = self._make_iter(chunks)

        upstream = AsyncMock()
        sent_payloads: list[dict[str, Any]] = []

        async def capture_send(payload: str) -> None:
            sent_payloads.append(json.loads(payload))

        upstream.send = capture_send

        await _relay_client_to_upstream(client_ws, upstream)

        assert sent_payloads[0]["flush"] is True
        assert sent_payloads[0]["try_trigger_generation"] is True

    @staticmethod
    def _make_iter(items: list[str]):
        async def _gen():
            for item in items:
                yield item

        return lambda: _gen()


class TestRelayUpstreamToClient:
    @pytest.mark.asyncio
    async def test_forwards_audio_messages_to_client(self) -> None:
        messages = [
            json.dumps({"audio": "base64audio1"}),
            json.dumps({"audio": "base64audio2"}),
            json.dumps({"isFinal": True}),
        ]
        upstream = MagicMock()
        upstream.__aiter__ = self._make_iter(messages)

        client_ws = AsyncMock()
        client_ws.send_text = AsyncMock()

        await _relay_upstream_to_client(upstream, client_ws)

        assert client_ws.send_text.call_count == 3

    @pytest.mark.asyncio
    async def test_stops_after_is_final(self) -> None:
        messages = [
            json.dumps({"isFinal": True}),
            json.dumps({"audio": "should_not_be_forwarded"}),
        ]
        upstream = MagicMock()
        upstream.__aiter__ = self._make_iter(messages)

        client_ws = AsyncMock()
        client_ws.send_text = AsyncMock()

        await _relay_upstream_to_client(upstream, client_ws)

        assert client_ws.send_text.call_count == 1

    @pytest.mark.asyncio
    async def test_handles_binary_upstream_messages(self) -> None:
        messages = [b'{"isFinal": true}']
        upstream = MagicMock()
        upstream.__aiter__ = self._make_iter(messages)

        client_ws = AsyncMock()
        client_ws.send_text = AsyncMock()

        await _relay_upstream_to_client(upstream, client_ws)

        client_ws.send_text.assert_called_once_with('{"isFinal": true}')

    @staticmethod
    def _make_iter(items: list[str | bytes]):
        async def _gen(self):
            for item in items:
                yield item

        return _gen


def _make_upstream_mock(messages: list[str], send_capture: list[dict[str, Any]] | None = None):
    """Build a minimal async context manager that acts as a websockets connection."""

    class _FakeUpstream:
        def __aiter__(self):
            return self._gen()

        async def _gen(self):
            for msg in messages:
                yield msg

        async def send(self, payload: str) -> None:
            if send_capture is not None:
                send_capture.append(json.loads(payload))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args: object) -> None:
            pass

    return _FakeUpstream()


class TestStreamInputTts:
    @pytest.mark.asyncio
    async def test_returns_total_char_count(self) -> None:
        client_ws = AsyncMock()
        client_ws.iter_text = lambda: self._aiter([
            json.dumps({"text": "Hello, "}),
            json.dumps({"text": "world! "}),
            json.dumps({"text": ""}),
        ])

        upstream_messages = [
            json.dumps({"audio": "dGVzdA=="}),
            json.dumps({"isFinal": True}),
        ]

        with (
            patch("litellm.llms.elevenlabs.text_to_speech.ws_handler.get_secret_str", return_value="test-key"),
            patch("websockets.connect", return_value=_make_upstream_mock(upstream_messages)),
        ):
            total = await stream_input_tts(
                client_ws=client_ws,
                model="eleven_multilingual_v2",
                voice_id="21m00Tcm4TlvDq8ikWAM",
                output_format="mp3_44100_128",
            )

        assert total == 14

    @pytest.mark.asyncio
    async def test_sends_bos_with_voice_settings(self) -> None:
        client_ws = AsyncMock()
        client_ws.iter_text = lambda: self._aiter([json.dumps({"text": ""})])

        sent_messages: list[dict[str, Any]] = []
        upstream = _make_upstream_mock([json.dumps({"isFinal": True})], send_capture=sent_messages)

        with (
            patch("litellm.llms.elevenlabs.text_to_speech.ws_handler.get_secret_str", return_value="test-key"),
            patch("websockets.connect", return_value=upstream),
        ):
            await stream_input_tts(
                client_ws=client_ws,
                model="eleven_multilingual_v2",
                voice_id="abc",
                output_format="mp3_44100_128",
                voice_settings={"stability": 0.5, "similarity_boost": 0.75},
                generation_config={"chunk_length_schedule": [120]},
            )

        bos = sent_messages[0]
        assert bos["text"] == " "
        assert bos["voice_settings"] == {"stability": 0.5, "similarity_boost": 0.75}
        assert bos["generation_config"] == {"chunk_length_schedule": [120]}

    @pytest.mark.asyncio
    async def test_raises_when_api_key_missing(self) -> None:
        client_ws = MagicMock()

        with patch("litellm.llms.elevenlabs.text_to_speech.ws_handler.get_secret_str", return_value=None):
            with pytest.raises(ValueError, match="ELEVENLABS_API_KEY"):
                await stream_input_tts(
                    client_ws=client_ws,
                    model="eleven_multilingual_v2",
                    voice_id="abc",
                )

    @staticmethod
    async def _aiter(items: list[str]):
        for item in items:
            yield item
