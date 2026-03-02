import datetime
import json
import os
import sys
import unittest
from typing import List, Optional, Tuple
from unittest.mock import ANY, MagicMock, Mock, patch

import httpx
import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system-path
import litellm


@pytest.mark.asyncio
async def test_construct_request_headers_project_id_from_env(monkeypatch):
    """Test that construct_request_headers uses GCS_PUBSUB_PROJECT_ID environment variable."""
    from litellm.integrations.gcs_pubsub.pub_sub import GcsPubSubLogger

    # Set up test environment variable
    test_project_id = "test-project-123"
    monkeypatch.setenv("GCS_PUBSUB_PROJECT_ID", test_project_id)
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.premium_user",
        True,
    )

    try:
        # Create handler with no project_id
        handler = GcsPubSubLogger(
            topic_id="test-topic", credentials_path="test-path.json"
        )

        # Mock the Vertex AI auth calls
        mock_auth_header = "mock-auth-header"
        mock_token = "mock-token"

        with patch(
            "litellm.vertex_chat_completion._ensure_access_token_async"
        ) as mock_ensure_token:
            mock_ensure_token.return_value = (mock_auth_header, test_project_id)

            with patch(
                "litellm.vertex_chat_completion._get_token_and_url"
            ) as mock_get_token:
                mock_get_token.return_value = (mock_token, "mock-url")

                # Call construct_request_headers
                headers = await handler.construct_request_headers()

                # Verify headers
                assert headers == {
                    "Authorization": f"Bearer {mock_token}",
                    "Content-Type": "application/json",
                }

                # Verify _ensure_access_token_async was called with correct project_id
                mock_ensure_token.assert_called_once_with(
                    credentials="test-path.json",
                    project_id=test_project_id,
                    custom_llm_provider="vertex_ai",
                )
    finally:
        # Clean up environment variable
        del os.environ["GCS_PUBSUB_PROJECT_ID"]


# ---------------------------------------------------------------------------
# Unit tests for passthrough handler helpers
# These test the root-cause functions that produce wrong values in Pub/Sub.
# ---------------------------------------------------------------------------


class TestExtractModelFromUrl:
    """Tests for VertexPassthroughLoggingHandler.extract_model_from_url()"""

    @staticmethod
    def _extract(url: str) -> str:
        from litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler import (
            VertexPassthroughLoggingHandler,
        )

        return VertexPassthroughLoggingHandler.extract_model_from_url(url)

    def test_google_publisher_gemini(self):
        url = "https://us-central1-aiplatform.googleapis.com/v1/projects/proj/locations/us-central1/publishers/google/models/gemini-2.5-flash:generateContent"
        assert self._extract(url) == "gemini-2.5-flash"

    def test_meta_publisher_llama(self):
        """
        Bug: extract_model_from_url returns 'unknown' for meta publisher URLs
        because the regex /models/([^:]+) matches but also captures the publisher path.
        """
        url = "https://us-central1-aiplatform.googleapis.com/v1/projects/proj/locations/us-central1/publishers/meta/models/llama-3.1-70b-instruct-maas:rawPredict"
        result = self._extract(url)
        assert result != "unknown", (
            f"extract_model_from_url returned 'unknown' for meta publisher URL. "
            f"Got: '{result}'"
        )
        assert "llama" in result.lower(), f"Expected llama model name, got: '{result}'"

    def test_anthropic_publisher_claude(self):
        url = "https://us-central1-aiplatform.googleapis.com/v1/projects/proj/locations/us-central1/publishers/anthropic/models/claude-3-5-sonnet:rawPredict"
        result = self._extract(url)
        assert result != "unknown", f"Got 'unknown' for anthropic publisher URL"
        assert "claude" in result.lower()

    def test_no_models_segment(self):
        url = "https://example.com/v1/some/other/path"
        result = self._extract(url)
        assert result == "unknown"


class TestGetVertexPublisherFromUrl:
    """Tests for VertexPassthroughLoggingHandler._get_vertex_publisher_or_api_spec_from_url()"""

    @staticmethod
    def _get_publisher(url: str) -> Optional[str]:
        from litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler import (
            VertexPassthroughLoggingHandler,
        )

        return VertexPassthroughLoggingHandler._get_vertex_publisher_or_api_spec_from_url(url)

    def test_anthropic_publisher(self):
        url = "https://us-central1-aiplatform.googleapis.com/v1/projects/proj/locations/us-central1/publishers/anthropic/models/claude-3-5-sonnet:rawPredict"
        assert self._get_publisher(url) == "anthropic"

    def test_mistralai_publisher(self):
        url = "https://us-central1-aiplatform.googleapis.com/v1/projects/proj/locations/us-central1/publishers/mistralai/models/mistral-large:rawPredict"
        assert self._get_publisher(url) == "mistralai"

    @pytest.mark.xfail(reason="Bug: _get_vertex_publisher_or_api_spec_from_url does not detect meta publisher")
    def test_meta_publisher(self):
    def test_google_publisher_returns_none(self):
        """Google publisher models go through generateContent, not rawPredict."""
        url = "https://us-central1-aiplatform.googleapis.com/v1/projects/proj/locations/us-central1/publishers/google/models/gemini-2.5-flash:generateContent"
        result = self._get_publisher(url)
        # Google publisher is not expected to be returned (handled by generateContent path)
        assert result is None


class TestAnthropicPassthroughMessages:
    """Tests that Anthropic passthrough handler preserves request messages for logging."""

    @pytest.mark.asyncio
    async def test_anthropic_handler_includes_request_messages_in_kwargs(self):
        """
        Bug: anthropic_passthrough_handler calls transform_response with messages=[]
        (hardcoded at line 69). The request_body messages are never propagated to
        kwargs for downstream standard logging, causing messages={} in Pub/Sub.
        """
        from litellm.litellm_core_utils.litellm_logging import (
            Logging as LiteLLMLoggingObj,
        )
        from litellm.proxy.pass_through_endpoints.llm_provider_handlers.anthropic_passthrough_logging_handler import (
            AnthropicPassthroughLoggingHandler,
        )

        anthropic_response_body = {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-sonnet-20241022",
            "content": [{"type": "text", "text": "Response text"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        }

        mock_httpx_response = MagicMock(spec=httpx.Response)
        mock_httpx_response.json.return_value = anthropic_response_body
        mock_httpx_response.text = json.dumps(anthropic_response_body)
        mock_httpx_response.status_code = 200
        mock_httpx_response.headers = {"content-type": "application/json"}

        request_body = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
            ],
        }

        logging_obj = LiteLLMLoggingObj(
            model="claude-3-5-sonnet-20241022",
            messages=request_body["messages"],
            stream=False,
            call_type="acompletion",
            start_time=datetime.datetime.now(),
            litellm_call_id="test-call-id",
            function_id="test-function-id",
        )

        start_time = datetime.datetime.now()
        end_time = start_time + datetime.timedelta(seconds=1)

        result = AnthropicPassthroughLoggingHandler.anthropic_passthrough_handler(
            httpx_response=mock_httpx_response,
            response_body=anthropic_response_body,
            logging_obj=logging_obj,
            url_route="/v1/messages",
            result=json.dumps(anthropic_response_body),
            start_time=start_time,
            end_time=end_time,
            cache_hit=False,
            request_body=request_body,
        )

        kwargs = result["kwargs"]

        # The kwargs should contain messages from the request body so that
        # get_standard_logging_object_payload() can populate the messages field.
        # Currently this is NOT the case — kwargs["messages"] is never set.
        messages_in_kwargs = kwargs.get("messages")
        assert messages_in_kwargs is not None, (
            "kwargs['messages'] should be set from request_body['messages'] "
            "so that the StandardLoggingPayload messages field is populated. "
            "Currently the Anthropic passthrough handler does not set this."
        )
        assert isinstance(messages_in_kwargs, list), (
            f"kwargs['messages'] should be a list, got {type(messages_in_kwargs)}"
        )
        assert len(messages_in_kwargs) > 0
