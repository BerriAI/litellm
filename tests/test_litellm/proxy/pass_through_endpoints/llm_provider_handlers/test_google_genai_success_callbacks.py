"""
Regression tests for GitHub issue #24097:
  success_callback functions silently skipped for /models/{model}:streamGenerateContent

Root cause: streaming_iterator tagged endpoint as VERTEX_AI instead of GOOGLE_GENAI,
so _route_streaming_logging_to_handler had no branch for it and skipped all callbacks.

Run:
    cd /Users/awais.qureshi/Documents/devstack/forks/litellm
    pyenv activate wagtail-chat-env
    python -m pytest tests/test_litellm/proxy/pass_through_endpoints/llm_provider_handlers/test_google_genai_success_callbacks.py -v
"""

import inspect
import sys
import os

import pytest

sys.path.insert(0, os.path.abspath("../../.."))


# ---------------------------------------------------------------------------
# Step 1 — Enum check
# ---------------------------------------------------------------------------

class TestEndpointTypeEnum:
    def test_google_genai_enum_exists(self):
        """GOOGLE_GENAI must be a member of EndpointType."""
        from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType
        assert hasattr(EndpointType, "GOOGLE_GENAI"), (
            "EndpointType.GOOGLE_GENAI is missing — fix not applied"
        )
        assert EndpointType.GOOGLE_GENAI == "google-genai"

    def test_vertex_ai_enum_still_exists(self):
        """VERTEX_AI must still exist (no regression)."""
        from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType
        assert hasattr(EndpointType, "VERTEX_AI")


# ---------------------------------------------------------------------------
# Step 2 — streaming_iterator uses GOOGLE_GENAI, not VERTEX_AI
# ---------------------------------------------------------------------------

class TestStreamingIteratorEndpointType:
    def test_uses_google_genai_not_vertex_ai(self):
        """streaming_iterator must tag chunks as GOOGLE_GENAI."""
        import litellm.google_genai.streaming_iterator as si
        src = inspect.getsource(si.BaseGoogleGenAIGenerateContentStreamingIterator)
        assert "EndpointType.GOOGLE_GENAI" in src, (
            "streaming_iterator still references VERTEX_AI — fix not applied"
        )
        assert "EndpointType.VERTEX_AI" not in src, (
            "streaming_iterator still uses VERTEX_AI — must use GOOGLE_GENAI"
        )


# ---------------------------------------------------------------------------
# Step 3 — streaming_handler has a GOOGLE_GENAI branch
# ---------------------------------------------------------------------------

class TestStreamingHandlerRouting:
    def test_google_genai_branch_exists(self):
        """_route_streaming_logging_to_handler must handle GOOGLE_GENAI."""
        import litellm.proxy.pass_through_endpoints.streaming_handler as sh
        src = inspect.getsource(sh.PassThroughStreamingHandler._route_streaming_logging_to_handler)
        assert "EndpointType.GOOGLE_GENAI" in src, (
            "No GOOGLE_GENAI branch in _route_streaming_logging_to_handler — fix not applied"
        )

    def test_gemini_handler_imported(self):
        """GeminiPassthroughLoggingHandler must be imported in streaming_handler."""
        import litellm.proxy.pass_through_endpoints.streaming_handler as sh
        assert hasattr(sh, "GeminiPassthroughLoggingHandler"), (
            "GeminiPassthroughLoggingHandler not imported in streaming_handler"
        )


# ---------------------------------------------------------------------------
# Step 4 — Unit test: GOOGLE_GENAI endpoint actually routes to Gemini handler
# ---------------------------------------------------------------------------

class TestStreamingHandlerGoogleGenAIRouting:
    @pytest.mark.asyncio
    async def test_google_genai_routes_to_gemini_handler(self):
        """GOOGLE_GENAI endpoint_type must call GeminiPassthroughLoggingHandler."""
        from unittest.mock import MagicMock, AsyncMock, patch
        from datetime import datetime
        from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType

        mock_result = {
            "result": MagicMock(),
            "kwargs": {"response_cost": 0.0001, "model": "gemini-1.5-flash"},
        }

        with patch(
            "litellm.proxy.pass_through_endpoints.streaming_handler"
            ".GeminiPassthroughLoggingHandler._handle_logging_gemini_collected_chunks",
            return_value=mock_result,
        ) as mock_gemini, patch(
            "litellm.proxy.pass_through_endpoints.streaming_handler"
            ".PassThroughStreamingHandler._convert_raw_bytes_to_str_lines",
            return_value=["data: {}\n"],
        ):
            from litellm.proxy.pass_through_endpoints.streaming_handler import (
                PassThroughStreamingHandler,
            )

            mock_logging = MagicMock()
            mock_logging.model_call_details = {}
            mock_logging.async_success_handler = AsyncMock()

            await PassThroughStreamingHandler._route_streaming_logging_to_handler(
                litellm_logging_obj=mock_logging,
                passthrough_success_handler_obj=MagicMock(),
                url_route="/v1/models/gemini-1.5-flash:streamGenerateContent",
                request_body={},
                endpoint_type=EndpointType.GOOGLE_GENAI,
                start_time=datetime.now(),
                end_time=datetime.now(),
                raw_bytes=b"data: {}\n",
                model="gemini-1.5-flash",
            )

            mock_gemini.assert_called_once(), (
                "GeminiPassthroughLoggingHandler was NOT called for GOOGLE_GENAI endpoint — "
                "callbacks would be silently skipped (issue #24097 not fixed)"
            )
            assert (
                "async_complete_streaming_response" in mock_logging.model_call_details
            ), (
                "async_complete_streaming_response not set on model_call_details — "
                "function-based success_callbacks will be silently skipped (self.stream=True guard)"
            )

    @pytest.mark.asyncio
    async def test_vertex_ai_does_not_route_to_gemini_handler(self):
        """VERTEX_AI endpoint_type must NOT call GeminiPassthroughLoggingHandler."""
        from unittest.mock import MagicMock, AsyncMock, patch
        from datetime import datetime
        from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType

        with patch(
            "litellm.proxy.pass_through_endpoints.streaming_handler"
            ".GeminiPassthroughLoggingHandler._handle_logging_gemini_collected_chunks",
        ) as mock_gemini, patch(
            "litellm.proxy.pass_through_endpoints.streaming_handler"
            ".PassThroughStreamingHandler._convert_raw_bytes_to_str_lines",
            return_value=["data: {}\n"],
        ), patch(
            "litellm.proxy.pass_through_endpoints.streaming_handler"
            ".VertexPassthroughLoggingHandler._handle_logging_vertex_collected_chunks",
            return_value={"result": MagicMock(), "kwargs": {}},
        ):
            from litellm.proxy.pass_through_endpoints.streaming_handler import (
                PassThroughStreamingHandler,
            )

            mock_logging = MagicMock()
            mock_logging.async_success_handler = AsyncMock()

            await PassThroughStreamingHandler._route_streaming_logging_to_handler(
                litellm_logging_obj=mock_logging,
                passthrough_success_handler_obj=MagicMock(),
                url_route="/v1/projects/proj/locations/us/publishers/google/models/gemini:streamGenerateContent",
                request_body={},
                endpoint_type=EndpointType.VERTEX_AI,
                start_time=datetime.now(),
                end_time=datetime.now(),
                raw_bytes=b"data: {}\n",
                model="gemini-1.5-pro",
            )

            mock_gemini.assert_not_called(), (
                "GeminiPassthroughLoggingHandler was called for VERTEX_AI endpoint — "
                "routing regression detected"
            )
