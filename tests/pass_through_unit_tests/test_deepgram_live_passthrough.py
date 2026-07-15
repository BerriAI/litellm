"""
Test Deepgram streaming Speech-to-Text (`/v1/listen`) WebSocket passthrough.

Covers duration-based cost tracking, transcript aggregation, route detection, and the
endpoint wiring (auth header, query forwarding, model resolution).
"""

import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.deepgram_live_passthrough_logging_handler import (
    DEEPGRAM_DEFAULT_MODEL,
    DeepgramLivePassthroughLoggingHandler,
)
from litellm.proxy.pass_through_endpoints.pass_through_endpoints import decode_ws_frame_for_logging
from litellm.proxy.pass_through_endpoints.success_handler import PassThroughEndpointLogging

NOVA_3_COST_PER_SECOND = 7.167e-05


def _results(start, duration, transcript, is_final=True):
    return {
        "type": "Results",
        "start": start,
        "duration": duration,
        "is_final": is_final,
        "channel": {"alternatives": [{"transcript": transcript}]},
    }


class TestDeepgramDurationExtraction:
    def setup_method(self):
        self.handler = DeepgramLivePassthroughLoggingHandler()

    def test_prefers_metadata_duration_over_transcript_end(self):
        messages = [
            _results(0.0, 2.0, "hello"),
            _results(2.0, 3.0, "world"),
            {"type": "Metadata", "duration": 42.0},
        ]
        assert self.handler.extract_audio_duration_seconds(messages) == 42.0

    def test_falls_back_to_transcript_end_without_metadata(self):
        messages = [
            _results(0.0, 2.0, "hello"),
            _results(2.0, 3.0, "world"),
        ]
        assert self.handler.extract_audio_duration_seconds(messages) == 5.0

    def test_uses_last_metadata_duration(self):
        messages = [
            {"type": "Metadata", "duration": 10.0},
            {"type": "Metadata", "duration": 12.5},
        ]
        assert self.handler.extract_audio_duration_seconds(messages) == 12.5

    def test_empty_messages_zero_duration(self):
        assert self.handler.extract_audio_duration_seconds([]) == 0.0

    def test_ignores_non_numeric_and_bool_durations(self):
        messages = [
            {"type": "Metadata", "duration": True},
            {"type": "Metadata", "duration": "oops"},
            _results(0.0, 4.0, "hi"),
        ]
        assert self.handler.extract_audio_duration_seconds(messages) == 4.0


class TestDeepgramTranscriptExtraction:
    def setup_method(self):
        self.handler = DeepgramLivePassthroughLoggingHandler()

    def test_joins_only_final_segments(self):
        messages = [
            _results(0.0, 1.0, "interim ignored", is_final=False),
            _results(0.0, 1.0, "final one"),
            _results(1.0, 1.0, "final two"),
        ]
        assert self.handler.extract_transcript(messages) == "final one final two"

    def test_handles_unicode(self):
        messages = [_results(0.0, 1.0, "héllo wörld")]
        assert self.handler.extract_transcript(messages) == "héllo wörld"


class TestDeepgramCostTracking:
    def setup_method(self):
        self.handler = DeepgramLivePassthroughLoggingHandler()

    @patch(
        "litellm.proxy.pass_through_endpoints.llm_provider_handlers.deepgram_live_passthrough_logging_handler.get_model_info"
    )
    def test_cost_is_duration_times_per_second(self, mock_get_model_info):
        mock_get_model_info.return_value = {"input_cost_per_second": NOVA_3_COST_PER_SECOND}
        messages = [{"type": "Metadata", "duration": 60.0}]

        result = self.handler.deepgram_live_passthrough_handler(
            websocket_messages=messages,
            start_time=datetime.now(),
            end_time=datetime.now(),
            model="nova-3",
        )

        assert result["kwargs"]["response_cost"] == pytest.approx(60.0 * NOVA_3_COST_PER_SECOND)
        assert result["kwargs"]["custom_llm_provider"] == "deepgram"
        assert result["kwargs"]["model"] == "nova-3"
        mock_get_model_info.assert_called_once_with(model="deepgram/nova-3", custom_llm_provider="deepgram")

    @patch(
        "litellm.proxy.pass_through_endpoints.llm_provider_handlers.deepgram_live_passthrough_logging_handler.get_model_info"
    )
    def test_defaults_model_when_missing(self, mock_get_model_info):
        mock_get_model_info.return_value = {"input_cost_per_second": NOVA_3_COST_PER_SECOND}

        result = self.handler.deepgram_live_passthrough_handler(
            websocket_messages=[{"type": "Metadata", "duration": 1.0}],
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

        assert result["kwargs"]["model"] == DEEPGRAM_DEFAULT_MODEL
        mock_get_model_info.assert_called_once_with(
            model=f"deepgram/{DEEPGRAM_DEFAULT_MODEL}", custom_llm_provider="deepgram"
        )

    @patch(
        "litellm.proxy.pass_through_endpoints.llm_provider_handlers.deepgram_live_passthrough_logging_handler.get_model_info"
    )
    def test_missing_pricing_bills_zero(self, mock_get_model_info):
        mock_get_model_info.side_effect = Exception("no pricing")

        result = self.handler.deepgram_live_passthrough_handler(
            websocket_messages=[{"type": "Metadata", "duration": 99.0}],
            start_time=datetime.now(),
            end_time=datetime.now(),
            model="made-up-model",
        )

        assert result["kwargs"]["response_cost"] == 0.0

    def test_cost_uses_real_pricing_from_cost_map(self):
        result = self.handler.deepgram_live_passthrough_handler(
            websocket_messages=[{"type": "Metadata", "duration": 60.0}],
            start_time=datetime.now(),
            end_time=datetime.now(),
            model="nova-3",
        )
        assert result["kwargs"]["response_cost"] == pytest.approx(60.0 * NOVA_3_COST_PER_SECOND)


class TestDeepgramRouteDetection:
    def test_route_detection(self):
        handler = PassThroughEndpointLogging()
        assert handler.is_deepgram_live_route("/deepgram/v1/listen") is True
        assert handler.is_deepgram_live_route("/deepgram/listen") is True
        assert handler.is_deepgram_live_route("/vertex_ai/live") is False
        assert handler.is_deepgram_live_route("/openai/chat/completions") is False
        assert handler.is_deepgram_live_route("") is False

    def test_deepgram_route_not_matched_as_openai(self):
        handler = PassThroughEndpointLogging()
        assert handler.is_openai_route("/deepgram/v1/listen") is False


class TestDeepgramEndpointWiring:
    @pytest.fixture
    def user_api_key(self):
        return UserAPIKeyAuth(api_key="sk-test", user_id="u", team_id="t", user_role="customer")

    def _mock_ws(self, query):
        ws = AsyncMock()
        ws.url = MagicMock()
        ws.url.query = query
        return ws

    @pytest.mark.asyncio
    @patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.websocket_passthrough_request")
    @patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router")
    async def test_forwards_query_and_auth(self, mock_router, mock_passthrough, user_api_key):
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            deepgram_listen_websocket_passthrough,
        )

        mock_router.get_credentials.return_value = "dg-secret"
        mock_passthrough.return_value = None
        ws = self._mock_ws("model=nova-2&interim_results=true&language=de")

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DEEPGRAM_API_BASE", None)
            await deepgram_listen_websocket_passthrough(websocket=ws, user_api_key_dict=user_api_key)

        mock_passthrough.assert_called_once()
        call = mock_passthrough.call_args[1]
        assert call["target"] == "wss://api.deepgram.com/v1/listen?model=nova-2&interim_results=true&language=de"
        assert call["custom_headers"] == {"Authorization": "Token dg-secret"}
        assert call["model"] == "nova-2"
        assert call["custom_llm_provider"] == "deepgram"
        assert call["endpoint"] == "/deepgram/v1/listen"

    @pytest.mark.asyncio
    @patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.websocket_passthrough_request")
    @patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router")
    async def test_defaults_model_when_query_missing(self, mock_router, mock_passthrough, user_api_key):
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            DEEPGRAM_LIVE_DEFAULT_MODEL,
            deepgram_listen_websocket_passthrough,
        )

        mock_router.get_credentials.return_value = "dg-secret"
        mock_passthrough.return_value = None
        ws = self._mock_ws("")

        await deepgram_listen_websocket_passthrough(websocket=ws, user_api_key_dict=user_api_key)

        call = mock_passthrough.call_args[1]
        assert call["target"] == "wss://api.deepgram.com/v1/listen"
        assert call["model"] == DEEPGRAM_LIVE_DEFAULT_MODEL

    @pytest.mark.asyncio
    @patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.websocket_passthrough_request")
    @patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router")
    async def test_closes_when_no_api_key(self, mock_router, mock_passthrough, user_api_key):
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            deepgram_listen_websocket_passthrough,
        )

        mock_router.get_credentials.return_value = None
        ws = self._mock_ws("model=nova-3")

        await deepgram_listen_websocket_passthrough(websocket=ws, user_api_key_dict=user_api_key)

        ws.accept.assert_awaited_once()
        ws.close.assert_awaited_once()
        mock_passthrough.assert_not_called()

    @pytest.mark.asyncio
    @patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.websocket_passthrough_request")
    @patch("litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints.passthrough_endpoint_router")
    async def test_honors_deepgram_api_base_override(self, mock_router, mock_passthrough, user_api_key):
        from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
            deepgram_listen_websocket_passthrough,
        )

        mock_router.get_credentials.return_value = "dg-secret"
        mock_passthrough.return_value = None
        ws = self._mock_ws("model=nova-3")

        with patch.dict(os.environ, {"DEEPGRAM_API_BASE": "https://deepgram.internal.bank"}, clear=False):
            await deepgram_listen_websocket_passthrough(websocket=ws, user_api_key_dict=user_api_key)

        call = mock_passthrough.call_args[1]
        assert call["target"] == "wss://deepgram.internal.bank/v1/listen?model=nova-3"


class TestDeepgramSuccessHandlerDispatch:
    @pytest.mark.asyncio
    async def test_dispatch_routes_to_deepgram_handler(self):
        success_handler = PassThroughEndpointLogging()

        with patch(
            "litellm.proxy.pass_through_endpoints.llm_provider_handlers.deepgram_live_passthrough_logging_handler.DeepgramLivePassthroughLoggingHandler.deepgram_live_passthrough_handler"
        ) as mock_handler:
            mock_handler.return_value = {"result": MagicMock(), "kwargs": {"model": "nova-3"}}

            result = success_handler.normalize_llm_passthrough_logging_payload(
                httpx_response=MagicMock(),
                response_body=[{"type": "Metadata", "duration": 3.0}],
                request_body={},
                logging_obj=MagicMock(),
                url_route="/deepgram/v1/listen",
                result="ok",
                start_time=datetime.now(),
                end_time=datetime.now(),
                cache_hit=False,
                model="nova-3",
            )

        mock_handler.assert_called_once()
        assert mock_handler.call_args[1]["websocket_messages"] == [{"type": "Metadata", "duration": 3.0}]
        assert result["standard_logging_response_object"] is not None


class TestDecodeWsFrameForLogging:
    """Regression tests for the shared frame decoder used by every WebSocket passthrough."""

    def test_decodes_unicode_json_text_frame(self):
        frame = json.dumps({"type": "Results", "transcript": "grüße"})
        assert decode_ws_frame_for_logging(frame) == {"type": "Results", "transcript": "grüße"}

    def test_decodes_utf8_json_bytes_frame(self):
        frame = json.dumps({"type": "Metadata", "duration": 5.0}).encode("utf-8")
        assert decode_ws_frame_for_logging(frame) == {"type": "Metadata", "duration": 5.0}

    def test_returns_none_for_non_json_text(self):
        assert decode_ws_frame_for_logging("KeepAlive") is None

    def test_returns_none_for_non_utf8_binary_audio(self):
        assert decode_ws_frame_for_logging(b"\xff\xfe\x00\x01audio") is None

    def test_returns_none_for_non_object_json(self):
        assert decode_ws_frame_for_logging("[1, 2, 3]") is None
        assert decode_ws_frame_for_logging("42") is None
