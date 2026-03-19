"""
Regression tests for GitHub issue #24097:
  success_callback functions silently skipped for /models/{model}:streamGenerateContent

Root cause: streaming_iterator tagged endpoint as VERTEX_AI instead of GOOGLE_GENAI,
so _route_streaming_logging_to_handler had no branch for it and skipped all callbacks.

Run:
    python -m pytest tests/test_litellm/proxy/pass_through_endpoints/llm_provider_handlers/test_google_genai_success_callbacks.py -v
"""

import inspect
import sys
import os

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from litellm.integrations.custom_logger import CustomLogger


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
#           AND that success_callbacks are fired (not silently skipped).
# ---------------------------------------------------------------------------

class _SpyCustomLogger(CustomLogger):
    """Minimal spy that records async_log_success_event calls."""

    def __init__(self):
        super().__init__()
        self.success_events = []

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.success_events.append({"kwargs": kwargs, "response_obj": response_obj})


class TestStreamingHandlerGoogleGenAIRouting:
    @pytest.mark.asyncio
    async def test_google_genai_routes_to_gemini_handler_and_fires_callbacks(self):
        """GOOGLE_GENAI endpoint_type must call GeminiPassthroughLoggingHandler AND
        fire registered success_callbacks — not silently skip them.

        Previous regression: streaming_handler.py pre-set async_complete_streaming_response
        before calling async_success_handler, which triggered the early-return guard in
        litellm_logging.py:2492, causing every callback to be silently skipped.
        """
        from unittest.mock import MagicMock, patch
        from datetime import datetime
        from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
        from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType
        from litellm.types.utils import StandardPassThroughResponseObject

        spy = _SpyCustomLogger()
        standard_response = StandardPassThroughResponseObject(response="chunk data")
        mock_result = {
            "result": standard_response,
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

            # Use a real LiteLLMLoggingObj so async_success_handler runs its actual
            # callback-dispatch logic (including the guard at litellm_logging.py:2492).
            real_logging = LiteLLMLoggingObj(
                model="gemini-1.5-flash",
                messages=[],
                stream=False,
                call_type="pass_through_endpoint",
                start_time=datetime.now(),
                litellm_call_id="test-call-id",
                function_id="test-fn",
                # Register the spy directly so we don't pollute global litellm state.
                dynamic_async_success_callbacks=[spy],
            )

            await PassThroughStreamingHandler._route_streaming_logging_to_handler(
                litellm_logging_obj=real_logging,
                passthrough_success_handler_obj=MagicMock(),
                url_route="/v1/models/gemini-1.5-flash:streamGenerateContent",
                request_body={},
                endpoint_type=EndpointType.GOOGLE_GENAI,
                start_time=datetime.now(),
                end_time=datetime.now(),
                raw_bytes=[b"data: {}\n"],
                model="gemini-1.5-flash",
            )

        # 1. Gemini handler was called → routing is correct.
        assert mock_gemini.call_count == 1, (
            "GeminiPassthroughLoggingHandler was NOT called for GOOGLE_GENAI endpoint — "
            "callbacks would be silently skipped (issue #24097 not fixed)"
        )

        # 2. The spy was invoked → async_success_handler ran past the guard and
        #    actually dispatched callbacks instead of returning early.
        assert len(spy.success_events) == 1, (
            "success_callback was NOT invoked — async_success_handler returned before "
            "the callback loop (early-return guard at litellm_logging.py:2492 still "
            "triggered, meaning async_complete_streaming_response was pre-set before "
            "async_success_handler was called)"
        )

        # 3. async_complete_streaming_response is set on model_call_details by
        #    async_success_handler's call_type=="pass_through_endpoint" branch (not pre-set).
        assert (
            "async_complete_streaming_response" in real_logging.model_call_details
        ), "async_complete_streaming_response not set by async_success_handler"

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
                raw_bytes=[b"data: {}\n"],
                model="gemini-1.5-pro",
            )

            assert mock_gemini.call_count == 0, (
                "GeminiPassthroughLoggingHandler was called for VERTEX_AI endpoint — "
                "routing regression detected"
            )
