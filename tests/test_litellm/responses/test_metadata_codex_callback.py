"""
Test that metadata is passed to custom callbacks during chat completion calls to codex models.

Fixes issue: Metadata is no longer passed to custom callback during chat completion
calls to codex models (#21204)

Codex models (gpt-5.1-codex, gpt-5.2-codex) use mode=responses and route through
responses_api_bridge. The bridge converts metadata to litellm_metadata. This test
verifies metadata is preserved for custom callbacks via kwargs['litellm_params']['metadata'].
"""

import asyncio
import os
import sys
from typing import Optional
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.abspath("../../.."))

import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger


def _make_mock_http_response(response_dict: dict):
    """Create a mock HTTP response that returns response_dict from .json()."""

    class MockResponse:
        def __init__(self, json_data, status_code=200):
            self._json_data = json_data
            self.status_code = status_code
            self.text = str(json_data)
            self.headers = {}

        def json(self):
            return self._json_data

    return MockResponse(response_dict, 200)


class MetadataCaptureCallback(CustomLogger):
    """Custom callback that captures kwargs passed to async_log_success_event."""

    def __init__(self):
        self.captured_kwargs: Optional[dict] = None
        self.event = asyncio.Event()

    async def async_log_success_event(
        self, kwargs, response_obj, start_time, end_time
    ):
        self.captured_kwargs = kwargs
        self.event.set()


@pytest.mark.asyncio
async def test_metadata_passed_to_custom_callback_codex_models():
    """
    Test that metadata passed to completion() is available in custom callback
    when using codex models (responses API bridge path).

    Codex models have mode=responses and route through responses_api_bridge,
    which passes litellm_metadata. The fix ensures this is preserved as
    litellm_params.metadata for callback compatibility.
    """
    from litellm.types.llms.openai import ResponsesAPIResponse

    mock_response = ResponsesAPIResponse.model_construct(
        id="resp-test",
        created_at=0,
        output=[
            {
                "type": "message",
                "id": "msg-1",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello!"}],
            }
        ],
        object="response",
        model="gpt-5.1-codex",
        status="completed",
        usage={
            "input_tokens": 5,
            "output_tokens": 10,
            "total_tokens": 15,
        },
    )

    test_metadata = {"foo": "bar", "trace_id": "test-123"}
    callback = MetadataCaptureCallback()
    original_callbacks = litellm.callbacks.copy() if litellm.callbacks else []
    litellm.callbacks = [callback]

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        mock_post.return_value = _make_mock_http_response(
            mock_response.model_dump()
        )
        # gpt-5.1-codex has mode=responses - routes through responses bridge
        await litellm.acompletion(
            model="gpt-5.1-codex",
            messages=[{"role": "user", "content": "Hello"}],
            metadata=test_metadata,
        )

    await asyncio.wait_for(callback.event.wait(), timeout=5.0)

    assert callback.captured_kwargs is not None, "Callback should have been invoked"

    litellm_params = callback.captured_kwargs.get("litellm_params", {})
    metadata = litellm_params.get("metadata") or {}

    assert "foo" in metadata, "metadata['foo'] should be accessible in callback"
    assert metadata["foo"] == "bar"
    assert metadata.get("trace_id") == "test-123"


@pytest.mark.asyncio
async def test_metadata_passed_via_litellm_metadata_responses_api():
    """
    Test that when calling responses() directly with litellm_metadata,
    metadata is preserved for custom callbacks.

    Uses HTTP mock since mock_response returns early before update_environment_variables.
    """
    from litellm.types.llms.openai import ResponsesAPIResponse

    mock_response = ResponsesAPIResponse.model_construct(
        id="resp-test-2",
        created_at=0,
        output=[
            {
                "type": "message",
                "id": "msg-2",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hi there!"}],
            }
        ],
        object="response",
        model="gpt-4o",
        status="completed",
        usage={
            "input_tokens": 2,
            "output_tokens": 3,
            "total_tokens": 5,
        },
    )

    test_metadata = {"request_id": "req-456"}
    callback = MetadataCaptureCallback()
    litellm.callbacks = [callback]

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        mock_post.return_value = _make_mock_http_response(
            mock_response.model_dump()
        )
        await litellm.aresponses(
            model="gpt-4o",
            input="hi",
            litellm_metadata=test_metadata,
        )

    await asyncio.wait_for(callback.event.wait(), timeout=5.0)

    assert callback.captured_kwargs is not None

    litellm_params = callback.captured_kwargs.get("litellm_params", {})
    metadata = litellm_params.get("metadata") or {}

    assert "request_id" in metadata
    assert metadata["request_id"] == "req-456"
