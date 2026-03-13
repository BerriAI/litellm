"""
Test that custom pricing from model_info is correctly propagated through
the metadata flow for all API endpoints.

Background:
- PR #20679 correctly strips custom pricing from the shared litellm.model_cost key
  to prevent cross-deployment pollution.
- The deployment's model_info retains custom pricing and must reach cost calculation
  via litellm_params["metadata"]["model_info"].
- /v1/chat/completions works because it explicitly extracts pricing into litellm_params.
- /v1/responses and /v1/messages fail because model_info with custom pricing
  does not reach the logging object's litellm_params["metadata"].

These tests verify that use_custom_pricing_for_model() returns True when called
with the litellm_params as constructed by each code path.
"""

import asyncio
import json
import os
import sys
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../.."))

import litellm
from litellm import Router
from litellm.litellm_core_utils.core_helpers import (
    get_model_info_from_litellm_params,
    merge_metadata_preserving_deployment_model_info,
)
from litellm.litellm_core_utils.litellm_logging import (
    Logging as LiteLLMLoggingObj,
    use_custom_pricing_for_model,
)
from litellm.litellm_core_utils.litellm_logging import CustomLogger


# ---------------------------------------------------------------------------
# Custom callback to capture response cost from the logging flow
# ---------------------------------------------------------------------------
class CostCapturingCallback(CustomLogger):
    """Captures response_cost from the async success callback kwargs."""

    def __init__(self):
        super().__init__()
        self.response_cost: Optional[float] = None
        self.custom_pricing: Optional[bool] = None
        self.event = asyncio.Event()

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.response_cost = kwargs.get("response_cost")
        # Extract custom_pricing from standard_logging_object if available
        slo = kwargs.get("standard_logging_object", {})
        if slo:
            self.custom_pricing = slo.get("custom_pricing")
        self.event.set()

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.response_cost = kwargs.get("response_cost")
        slo = kwargs.get("standard_logging_object", {})
        if slo:
            self.custom_pricing = slo.get("custom_pricing")


# ---------------------------------------------------------------------------
# Fixtures: Router with custom pricing deployment
# ---------------------------------------------------------------------------
CUSTOM_INPUT_COST = 0.50  # $0.50 per token — absurdly high, easy to spot
CUSTOM_OUTPUT_COST = 1.00


DEPLOYMENT_MODEL_ID = "deployment-custom-pricing-test"


def _make_router_with_custom_pricing(backend_model: str, api_key: str = "fake-key"):
    """Create a Router with a single deployment that has custom pricing."""
    return Router(
        model_list=[
            {
                "model_name": "test-custom-pricing",
                "litellm_params": {
                    "model": backend_model,
                    "api_key": api_key,
                },
                "model_info": {
                    "id": DEPLOYMENT_MODEL_ID,
                    "input_cost_per_token": CUSTOM_INPUT_COST,
                    "output_cost_per_token": CUSTOM_OUTPUT_COST,
                },
            },
        ],
    )

@pytest.fixture(autouse=True)
def cleanup_model_cost():
    """Remove callback and pricing state between tests."""
    yield
    litellm.callbacks = []
    litellm.model_cost.pop(DEPLOYMENT_MODEL_ID, None)


# ---------------------------------------------------------------------------
# Mock HTTP response helpers
# ---------------------------------------------------------------------------
class MockHTTPResponse:
    """Mimics httpx.Response for non-streaming and streaming."""

    def __init__(self, json_data, status_code=200, headers=None):
        self._json_data = json_data
        self.status_code = status_code
        self.text = json.dumps(json_data)
        self.headers = httpx.Headers(headers or {"content-type": "application/json"})

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "error", request=Mock(), response=self
            )

    async def aiter_bytes(self):
        yield self.text.encode("utf-8")

    async def aiter_lines(self):
        for line in self.text.split("\n"):
            yield line

    def iter_lines(self):
        for line in self.text.split("\n"):
            yield line


class MockStreamingHTTPResponse:
    """Mimics httpx.Response for streaming (SSE)."""

    def __init__(self, sse_lines: list[str], status_code=200, headers=None):
        self._sse_lines = sse_lines
        self.status_code = status_code
        self.headers = httpx.Headers(
            headers or {"content-type": "text/event-stream"}
        )
        self.text = "\n".join(sse_lines)

    def json(self):
        return json.loads(self.text)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "error", request=Mock(), response=self
            )

    def iter_lines(self):
        for line in self._sse_lines:
            yield line

    async def aiter_lines(self):
        for line in self._sse_lines:
            yield line

    async def aiter_bytes(self):
        for line in self._sse_lines:
            yield (line + "\n").encode("utf-8")


# ---------------------------------------------------------------------------
# Standard mock response payloads
# ---------------------------------------------------------------------------
RESPONSES_API_MOCK = {
    "id": "resp_test_custom_pricing",
    "object": "response",
    "created_at": 1741476542,
    "status": "completed",
    "model": "gpt-4o",
    "output": [
        {
            "type": "message",
            "id": "msg_test",
            "status": "completed",
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": "Hello!", "annotations": []}
            ],
        }
    ],
    "parallel_tool_calls": True,
    "usage": {
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
        "output_tokens_details": {"reasoning_tokens": 0},
    },
    "text": {"format": {"type": "text"}},
    "error": None,
    "incomplete_details": None,
    "instructions": None,
    "metadata": {},
    "temperature": 1.0,
    "tool_choice": "auto",
    "tools": [],
    "top_p": 1.0,
    "max_output_tokens": None,
    "previous_response_id": None,
    "reasoning": {"effort": None, "summary": None},
    "truncation": "disabled",
    "user": None,
}

ANTHROPIC_MESSAGES_MOCK = {
    "id": "msg_test_custom_pricing",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "Hello!"}],
    "model": "claude-sonnet-4-20250514",
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {
        "input_tokens": 100,
        "output_tokens": 50,
    },
}


# =========================================================================
# UNIT TESTS: Metadata propagation to use_custom_pricing_for_model
# =========================================================================


class TestRouterMetadataPropagation:
    """
    Verify that Router._update_kwargs_with_deployment places model_info
    with custom pricing into the correct metadata location.
    """

    def test_chat_completions_puts_model_info_in_metadata(self):
        """
        For chat/completions, function_name is None or "acompletion" etc.,
        so metadata_variable_name = "metadata". model_info should be in
        kwargs["metadata"]["model_info"].
        """
        router = _make_router_with_custom_pricing("openai/gpt-4o")
        deployment = router.model_list[0]

        kwargs: dict = {}
        router._update_kwargs_with_deployment(
            deployment=deployment, kwargs=kwargs, function_name="acompletion"
        )

        # model_info should be in kwargs["metadata"]["model_info"]
        assert "metadata" in kwargs
        model_info = kwargs["metadata"].get("model_info", {})
        assert model_info.get("input_cost_per_token") == CUSTOM_INPUT_COST
        assert model_info.get("output_cost_per_token") == CUSTOM_OUTPUT_COST

        # use_custom_pricing_for_model should return True when litellm_params
        # includes this metadata
        litellm_params = {"metadata": kwargs["metadata"]}
        assert use_custom_pricing_for_model(litellm_params) is True

    def test_generic_api_call_puts_model_info_in_litellm_metadata(self):
        """
        For _ageneric_api_call_with_fallbacks (used by /v1/responses and /v1/messages),
        metadata_variable_name = "litellm_metadata". model_info should be in
        kwargs["litellm_metadata"]["model_info"].
        """
        router = _make_router_with_custom_pricing("openai/gpt-4o")
        deployment = router.model_list[0]

        kwargs: dict = {}
        router._update_kwargs_with_deployment(
            deployment=deployment,
            kwargs=kwargs,
            function_name="_ageneric_api_call_with_fallbacks",
        )

        # model_info should be in kwargs["litellm_metadata"]["model_info"]
        assert "litellm_metadata" in kwargs
        model_info = kwargs["litellm_metadata"].get("model_info", {})
        assert model_info.get("input_cost_per_token") == CUSTOM_INPUT_COST
        assert model_info.get("output_cost_per_token") == CUSTOM_OUTPUT_COST

    def test_responses_api_metadata_for_callbacks_gets_model_info(self):
        """
        In responses/main.py, metadata_for_callbacks should merge
        litellm_metadata (which has model_info) with explicit metadata.
        """
        router = _make_router_with_custom_pricing("openai/gpt-4o")
        deployment = router.model_list[0]

        kwargs: dict = {}
        router._update_kwargs_with_deployment(
            deployment=deployment,
            kwargs=kwargs,
            function_name="_ageneric_api_call_with_fallbacks",
        )

        metadata = {"user_field": "present"}
        metadata_for_callbacks = merge_metadata_preserving_deployment_model_info(
            kwargs.get("litellm_metadata"), metadata
        )

        model_info = metadata_for_callbacks.get("model_info", {})
        assert model_info.get("input_cost_per_token") == CUSTOM_INPUT_COST, (
            "metadata_for_callbacks should contain model_info with custom pricing "
            "when explicit metadata is also passed"
        )
        assert metadata_for_callbacks["user_field"] == "present"

        litellm_params = {"metadata": metadata_for_callbacks}
        assert use_custom_pricing_for_model(litellm_params) is True

    def test_responses_api_user_model_info_does_not_override_deployment(self):
        """
        User metadata should not overwrite router-provided model_info for
        responses callback pricing calculation.
        """
        router = _make_router_with_custom_pricing("openai/gpt-4o")
        deployment = router.model_list[0]

        kwargs: dict = {}
        router._update_kwargs_with_deployment(
            deployment=deployment,
            kwargs=kwargs,
            function_name="_ageneric_api_call_with_fallbacks",
        )

        user_metadata = {
            "user_field": "present",
            "model_info": {"id": "user-supplied", "input_cost_per_token": 0.0},
        }
        metadata_for_callbacks = merge_metadata_preserving_deployment_model_info(
            kwargs.get("litellm_metadata"), user_metadata
        )

        model_info = metadata_for_callbacks.get("model_info", {})
        assert model_info.get("id") == DEPLOYMENT_MODEL_ID
        assert model_info.get("input_cost_per_token") == CUSTOM_INPUT_COST
        assert metadata_for_callbacks["user_field"] == "present"

    def test_messages_api_user_model_info_does_not_override_deployment(self):
        """
        For /v1/messages, the handler should preserve router-provided model_info
        even if the request metadata includes its own model_info payload.
        """
        router = _make_router_with_custom_pricing("anthropic/claude-sonnet-4-20250514")
        deployment = router.model_list[0]

        kwargs: dict = {}
        router._update_kwargs_with_deployment(
            deployment=deployment,
            kwargs=kwargs,
            function_name="_ageneric_api_call_with_fallbacks",
        )

        kwargs["metadata"] = {
            "user_field": "present",
            "model_info": {"id": "user-supplied", "output_cost_per_token": 0.0},
        }
        metadata_from_handler = merge_metadata_preserving_deployment_model_info(
            kwargs.get("litellm_metadata"), kwargs.get("metadata")
        )
        litellm_params = {"metadata": metadata_from_handler}

        model_info = metadata_from_handler.get("model_info", {})
        assert model_info.get("id") == DEPLOYMENT_MODEL_ID
        assert model_info.get("output_cost_per_token") == CUSTOM_OUTPUT_COST
        assert metadata_from_handler["user_field"] == "present"
        assert use_custom_pricing_for_model(litellm_params) is True

    def test_empty_deployment_model_info_still_overrides_user_metadata(self):
        """
        An explicitly empty deployment model_info should remain explicit during
        merge rather than being treated like a missing value.
        """
        metadata_for_callbacks = merge_metadata_preserving_deployment_model_info(
            {"model_info": {}},
            {
                "user_field": "present",
                "model_info": {"id": "user-supplied", "input_cost_per_token": 0.0},
            },
        )

        assert metadata_for_callbacks["model_info"] == {}
        assert metadata_for_callbacks["user_field"] == "present"

    def test_get_model_info_preserves_explicit_empty_metadata_model_info(self):
        """Explicit empty metadata.model_info should win over fallback locations."""
        model_info = get_model_info_from_litellm_params(
            {
                "metadata": {"model_info": {}},
                "model_info": {"id": "top-level"},
                "litellm_metadata": {"model_info": {"id": "litellm-metadata"}},
            }
        )

        assert model_info == {}

    def test_use_custom_pricing_detects_top_level_model_info(self):
        """Custom pricing detection should work when model_info is top-level."""
        litellm_params = {
            "metadata": {"user_field": "present"},
            "model_info": {
                "id": DEPLOYMENT_MODEL_ID,
                "input_cost_per_token": CUSTOM_INPUT_COST,
                "output_cost_per_token": CUSTOM_OUTPUT_COST,
            },
        }

        assert use_custom_pricing_for_model(litellm_params) is True


# =========================================================================
# INTEGRATION TESTS: Full Router → cost calculation with HTTP mocking
# =========================================================================


class TestResponsesAPICustomPricingCost:
    """
    Test that /v1/responses (via router.aresponses) uses custom pricing
    for cost calculation when model_info has custom pricing fields.
    """

    @pytest.mark.asyncio
    async def test_nonstreaming_responses_uses_custom_pricing(self):
        """Non-streaming /v1/responses should use custom pricing for cost."""
        cost_callback = CostCapturingCallback()
        litellm.callbacks = [cost_callback]

        try:
            router = _make_router_with_custom_pricing("openai/gpt-4o")

            with patch(
                "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
                new_callable=AsyncMock,
            ) as mock_post:
                mock_post.return_value = MockHTTPResponse(RESPONSES_API_MOCK)

                response = await router.aresponses(
                    model="test-custom-pricing",
                    input="Hello, how are you?",
                )

                # Wait for async callback
                try:
                    await asyncio.wait_for(cost_callback.event.wait(), timeout=10.0)
                except asyncio.TimeoutError:
                    pass

            assert cost_callback.response_cost is not None, (
                "response_cost should be set in the callback"
            )

            # With 100 input + 50 output tokens at custom pricing:
            # expected = 100 * 0.50 + 50 * 1.00 = 50 + 50 = 100.0
            expected_custom_cost = 100 * CUSTOM_INPUT_COST + 50 * CUSTOM_OUTPUT_COST
            assert cost_callback.response_cost == pytest.approx(
                expected_custom_cost, rel=0.01
            ), (
                f"Cost should use custom pricing ({expected_custom_cost}), "
                f"got {cost_callback.response_cost}"
            )
        finally:
            litellm.callbacks = []

    @pytest.mark.asyncio
    async def test_streaming_responses_uses_custom_pricing(self):
        """Streaming /v1/responses should use custom pricing for cost."""
        cost_callback = CostCapturingCallback()
        litellm.callbacks = [cost_callback]

        try:
            router = _make_router_with_custom_pricing("openai/gpt-4o")

            # SSE events for streaming responses API
            sse_events = [
                'data: {"type":"response.created","response":{"id":"resp_test","object":"response","created_at":1741476542,"status":"in_progress","model":"gpt-4o","output":[],"usage":null}}',
                'data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_test","status":"in_progress","role":"assistant","content":[]}}',
                'data: {"type":"response.content_part.added","output_index":0,"content_index":0,"part":{"type":"output_text","text":"","annotations":[]}}',
                'data: {"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"Hello!"}',
                'data: {"type":"response.output_text.done","output_index":0,"content_index":0,"text":"Hello!"}',
                'data: {"type":"response.content_part.done","output_index":0,"content_index":0,"part":{"type":"output_text","text":"Hello!","annotations":[]}}',
                'data: {"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg_test","status":"completed","role":"assistant","content":[{"type":"output_text","text":"Hello!","annotations":[]}]}}',
                f'data: {{"type":"response.completed","response":{json.dumps(RESPONSES_API_MOCK)}}}',
                "data: [DONE]",
            ]

            mock_stream_response = MockStreamingHTTPResponse(sse_events)

            with patch(
                "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
                new_callable=AsyncMock,
            ) as mock_post:
                mock_post.return_value = mock_stream_response

                response = await router.aresponses(
                    model="test-custom-pricing",
                    input="Hello, how are you?",
                    stream=True,
                )

                # Consume the stream
                async for chunk in response:
                    pass

                # Wait for async callback
                try:
                    await asyncio.wait_for(cost_callback.event.wait(), timeout=10.0)
                except asyncio.TimeoutError:
                    pass

            assert cost_callback.response_cost is not None, (
                "response_cost should be set in the callback"
            )

            expected_custom_cost = 100 * CUSTOM_INPUT_COST + 50 * CUSTOM_OUTPUT_COST
            assert cost_callback.response_cost == pytest.approx(
                expected_custom_cost, rel=0.01
            ), (
                f"Streaming cost should use custom pricing ({expected_custom_cost}), "
                f"got {cost_callback.response_cost}"
            )
        finally:
            litellm.callbacks = []

    def test_sync_streaming_responses_uses_custom_pricing(self):
        """Sync streaming /v1/responses should use custom pricing for cost."""
        cost_callback = CostCapturingCallback()
        litellm.callbacks = [cost_callback]

        try:
            router = _make_router_with_custom_pricing("openai/gpt-4o")

            sse_events = [
                'data: {"type":"response.created","response":{"id":"resp_test","object":"response","created_at":1741476542,"status":"in_progress","model":"gpt-4o","output":[],"usage":null}}',
                'data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_test","status":"in_progress","role":"assistant","content":[]}}',
                'data: {"type":"response.content_part.added","output_index":0,"content_index":0,"part":{"type":"output_text","text":"","annotations":[]}}',
                'data: {"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"Hello!"}',
                'data: {"type":"response.output_text.done","output_index":0,"content_index":0,"text":"Hello!"}',
                'data: {"type":"response.content_part.done","output_index":0,"content_index":0,"part":{"type":"output_text","text":"Hello!","annotations":[]}}',
                'data: {"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg_test","status":"completed","role":"assistant","content":[{"type":"output_text","text":"Hello!","annotations":[]}]}}',
                f'data: {{"type":"response.completed","response":{json.dumps(RESPONSES_API_MOCK)}}}',
                "data: [DONE]",
            ]

            mock_stream_response = MockStreamingHTTPResponse(sse_events)

            with patch(
                "litellm.llms.custom_httpx.http_handler.HTTPHandler.post",
                new_callable=MagicMock,
            ) as mock_post:
                mock_post.return_value = mock_stream_response

                response = router.responses(
                    model="test-custom-pricing",
                    input="Hello, how are you?",
                    stream=True,
                )

                for _chunk in response:
                    pass

            assert cost_callback.response_cost is not None, (
                "response_cost should be set in the callback"
            )

            expected_custom_cost = 100 * CUSTOM_INPUT_COST + 50 * CUSTOM_OUTPUT_COST
            assert cost_callback.response_cost == pytest.approx(
                expected_custom_cost, rel=0.01
            ), (
                f"Sync streaming cost should use custom pricing ({expected_custom_cost}), "
                f"got {cost_callback.response_cost}"
            )
        finally:
            litellm.callbacks = []

    @pytest.mark.asyncio
    async def test_streaming_responses_with_user_metadata_uses_custom_pricing(self):
        """Streaming /v1/responses should preserve custom pricing when metadata is also passed."""
        cost_callback = CostCapturingCallback()
        litellm.callbacks = [cost_callback]

        try:
            router = _make_router_with_custom_pricing("openai/gpt-4o")

            sse_events = [
                'data: {"type":"response.created","response":{"id":"resp_test","object":"response","created_at":1741476542,"status":"in_progress","model":"gpt-4o","output":[],"usage":null}}',
                'data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_test","status":"in_progress","role":"assistant","content":[]}}',
                'data: {"type":"response.content_part.added","output_index":0,"content_index":0,"part":{"type":"output_text","text":"","annotations":[]}}',
                'data: {"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"Hello!"}',
                'data: {"type":"response.output_text.done","output_index":0,"content_index":0,"text":"Hello!"}',
                'data: {"type":"response.content_part.done","output_index":0,"content_index":0,"part":{"type":"output_text","text":"Hello!","annotations":[]}}',
                'data: {"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg_test","status":"completed","role":"assistant","content":[{"type":"output_text","text":"Hello!","annotations":[]}]}}',
                f'data: {{"type":"response.completed","response":{json.dumps(RESPONSES_API_MOCK)}}}',
                "data: [DONE]",
            ]

            mock_stream_response = MockStreamingHTTPResponse(sse_events)

            with patch(
                "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
                new_callable=AsyncMock,
            ) as mock_post:
                mock_post.return_value = mock_stream_response

                response = await router.aresponses(
                    model="test-custom-pricing",
                    input="Hello, how are you?",
                    stream=True,
                    metadata={"user_field": "present"},
                )

                async for chunk in response:
                    pass

                try:
                    await asyncio.wait_for(cost_callback.event.wait(), timeout=10.0)
                except asyncio.TimeoutError:
                    pass

            assert cost_callback.response_cost is not None
            expected_custom_cost = 100 * CUSTOM_INPUT_COST + 50 * CUSTOM_OUTPUT_COST
            assert cost_callback.response_cost == pytest.approx(
                expected_custom_cost, rel=0.01
            )
        finally:
            litellm.callbacks = []

    @pytest.mark.asyncio
    async def test_streaming_responses_with_user_model_info_ignored_for_pricing(self):
        """Streaming /v1/responses should ignore user-supplied metadata.model_info."""
        cost_callback = CostCapturingCallback()
        litellm.callbacks = [cost_callback]

        try:
            router = _make_router_with_custom_pricing("openai/gpt-4o")

            sse_events = [
                'data: {"type":"response.created","response":{"id":"resp_test","object":"response","created_at":1741476542,"status":"in_progress","model":"gpt-4o","output":[],"usage":null}}',
                'data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_test","status":"in_progress","role":"assistant","content":[]}}',
                'data: {"type":"response.content_part.added","output_index":0,"content_index":0,"part":{"type":"output_text","text":"","annotations":[]}}',
                'data: {"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"Hello!"}',
                'data: {"type":"response.output_text.done","output_index":0,"content_index":0,"text":"Hello!"}',
                'data: {"type":"response.content_part.done","output_index":0,"content_index":0,"part":{"type":"output_text","text":"Hello!","annotations":[]}}',
                'data: {"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg_test","status":"completed","role":"assistant","content":[{"type":"output_text","text":"Hello!","annotations":[]}]}}',
                f'data: {{"type":"response.completed","response":{json.dumps(RESPONSES_API_MOCK)}}}',
                "data: [DONE]",
            ]

            mock_stream_response = MockStreamingHTTPResponse(sse_events)

            with patch(
                "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
                new_callable=AsyncMock,
            ) as mock_post:
                mock_post.return_value = mock_stream_response

                response = await router.aresponses(
                    model="test-custom-pricing",
                    input="Hello, how are you?",
                    stream=True,
                    metadata={
                        "user_field": "present",
                        "model_info": {
                            "id": "user-garbage",
                            "input_cost_per_token": 0.0,
                            "output_cost_per_token": 0.0,
                        },
                    },
                )

                async for chunk in response:
                    pass

                try:
                    await asyncio.wait_for(cost_callback.event.wait(), timeout=10.0)
                except asyncio.TimeoutError:
                    pass

            assert cost_callback.response_cost is not None
            expected_custom_cost = 100 * CUSTOM_INPUT_COST + 50 * CUSTOM_OUTPUT_COST
            assert cost_callback.response_cost == pytest.approx(
                expected_custom_cost, rel=0.01
            )
        finally:
            litellm.callbacks = []


class TestAnthropicMessagesCustomPricingCost:
    """
    Test that /v1/messages (Anthropic via router) uses custom pricing
    for cost calculation when model_info has custom pricing fields.
    """

    @pytest.mark.asyncio
    async def test_nonstreaming_messages_uses_custom_pricing(self):
        """Non-streaming /v1/messages should use custom pricing for cost."""
        cost_callback = CostCapturingCallback()
        litellm.callbacks = [cost_callback]

        try:
            router = _make_router_with_custom_pricing(
                "anthropic/claude-sonnet-4-20250514"
            )

            with patch(
                "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
                new_callable=AsyncMock,
            ) as mock_post:
                mock_post.return_value = MockHTTPResponse(
                    ANTHROPIC_MESSAGES_MOCK,
                    headers={
                        "content-type": "application/json",
                        "request-id": "req_test",
                    },
                )

                response = await router.aanthropic_messages(
                    model="test-custom-pricing",
                    messages=[{"role": "user", "content": "Hello!"}],
                    max_tokens=100,
                )

                # Wait for async callback
                try:
                    await asyncio.wait_for(cost_callback.event.wait(), timeout=10.0)
                except asyncio.TimeoutError:
                    pass

            assert cost_callback.response_cost is not None, (
                "response_cost should be set in the callback"
            )

            expected_custom_cost = 100 * CUSTOM_INPUT_COST + 50 * CUSTOM_OUTPUT_COST
            assert cost_callback.response_cost == pytest.approx(
                expected_custom_cost, rel=0.01
            ), (
                f"Cost should use custom pricing ({expected_custom_cost}), "
                f"got {cost_callback.response_cost}"
            )
        finally:
            litellm.callbacks = []

    @pytest.mark.asyncio
    async def test_nonstreaming_messages_with_user_metadata_uses_custom_pricing(self):
        """Non-streaming /v1/messages should preserve custom pricing when metadata is also passed."""
        cost_callback = CostCapturingCallback()
        litellm.callbacks = [cost_callback]

        try:
            router = _make_router_with_custom_pricing(
                "anthropic/claude-sonnet-4-20250514"
            )

            with patch(
                "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
                new_callable=AsyncMock,
            ) as mock_post:
                mock_post.return_value = MockHTTPResponse(
                    ANTHROPIC_MESSAGES_MOCK,
                    headers={
                        "content-type": "application/json",
                        "request-id": "req_test",
                    },
                )

                response = await router.aanthropic_messages(
                    model="test-custom-pricing",
                    messages=[{"role": "user", "content": "Hello!"}],
                    max_tokens=100,
                    metadata={"user_field": "present"},
                )

                assert response is not None

                try:
                    await asyncio.wait_for(cost_callback.event.wait(), timeout=10.0)
                except asyncio.TimeoutError:
                    pass

            assert cost_callback.response_cost is not None, (
                "response_cost should be set in the callback"
            )

            expected_custom_cost = 100 * CUSTOM_INPUT_COST + 50 * CUSTOM_OUTPUT_COST
            assert cost_callback.response_cost == pytest.approx(
                expected_custom_cost, rel=0.01
            ), (
                f"Cost should use custom pricing ({expected_custom_cost}), "
                f"got {cost_callback.response_cost}"
            )
        finally:
            litellm.callbacks = []
