"""
Test that metadata is propagated to callbacks on the Responses API path.

Fixes #15466 - metadata passed to litellm.aresponses() was silently dropped
because the named `metadata` parameter is consumed by the function signature
and never appears in **kwargs. Downstream handlers read litellm_metadata from
kwargs, which was empty.

Tests cover:
1. Direct litellm.aresponses() call - metadata reaches callback
2. Router.aresponses() call - metadata merged with Router's model_group
3. Direct call stores a copy, not a reference to the caller's dict
"""

import asyncio
import os
import sys
from typing import Optional
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.abspath("../../.."))

import pytest

import litellm
from litellm import Router
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


def _mock_responses_api_response():
    """Build a minimal valid ResponsesAPIResponse for mocking."""
    from litellm.types.llms.openai import ResponsesAPIResponse

    return ResponsesAPIResponse.model_construct(
        id="resp-test",
        created_at=0,
        output=[
            {
                "type": "message",
                "id": "msg-1",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "OK"}],
            }
        ],
        object="response",
        model="gpt-4o-mini",
        status="completed",
        usage={
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
        },
    )


class MetadataCaptureCallback(CustomLogger):
    """Custom callback that captures kwargs from async_log_success_event."""

    def __init__(self):
        self.captured_kwargs: Optional[dict] = None
        self.event = asyncio.Event()

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.captured_kwargs = kwargs
        self.event.set()


@pytest.mark.asyncio
async def test_metadata_propagated_via_direct_aresponses():
    """
    When calling litellm.aresponses() directly with metadata, the metadata
    should be available in the callback via litellm_params.metadata.
    """
    mock_response = _mock_responses_api_response()
    test_metadata = {"session_id": "sess-123", "user_tag": "test"}
    callback = MetadataCaptureCallback()
    original_callbacks = litellm.callbacks.copy() if litellm.callbacks else []
    litellm.callbacks = [callback]

    try:
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
        ) as mock_post:
            mock_post.return_value = _make_mock_http_response(
                mock_response.model_dump()
            )
            await litellm.aresponses(
                input="Say OK",
                model="openai/gpt-4o-mini",
                metadata=test_metadata,
            )

        await asyncio.wait_for(callback.event.wait(), timeout=5.0)

        assert callback.captured_kwargs is not None, "Callback should have fired"

        litellm_params = callback.captured_kwargs.get("litellm_params", {})
        metadata = litellm_params.get("metadata") or {}

        assert metadata.get("session_id") == "sess-123"
        assert metadata.get("user_tag") == "test"
    finally:
        litellm.callbacks = original_callbacks


@pytest.mark.asyncio
async def test_metadata_propagated_via_router_aresponses():
    """
    When calling Router.aresponses() with metadata, the metadata should be
    merged with the Router's own litellm_metadata (model_group) and be
    available in the callback.
    """
    mock_response = _mock_responses_api_response()
    test_metadata = {"caller_service": "my-service", "request_id": "req-456"}
    callback = MetadataCaptureCallback()
    original_callbacks = litellm.callbacks.copy() if litellm.callbacks else []
    litellm.callbacks = [callback]

    router = Router(
        model_list=[
            {
                "model_name": "gpt-4o-mini",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "fake-key",
                },
            }
        ],
    )

    try:
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
        ) as mock_post:
            mock_post.return_value = _make_mock_http_response(
                mock_response.model_dump()
            )
            await router.aresponses(
                input="Say OK",
                model="gpt-4o-mini",
                metadata=test_metadata,
            )

        await asyncio.wait_for(callback.event.wait(), timeout=5.0)

        assert callback.captured_kwargs is not None, "Callback should have fired"

        litellm_params = callback.captured_kwargs.get("litellm_params", {})
        metadata = litellm_params.get("metadata") or {}

        assert metadata.get("caller_service") == "my-service"
        assert metadata.get("request_id") == "req-456"
        # Router's model_group should also be present
        assert "model_group" in metadata
    finally:
        litellm.callbacks = original_callbacks


@pytest.mark.asyncio
async def test_metadata_stored_by_copy_not_reference():
    """
    When litellm_metadata is absent (direct call path), the fix should store
    a copy of metadata, not a reference. Mutations by downstream handlers
    should not affect the caller's original dict.
    """
    mock_response = _mock_responses_api_response()
    original_metadata = {"key": "value"}
    callback = MetadataCaptureCallback()
    original_callbacks = litellm.callbacks.copy() if litellm.callbacks else []
    litellm.callbacks = [callback]

    try:
        with patch(
            "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
            new_callable=AsyncMock,
        ) as mock_post:
            mock_post.return_value = _make_mock_http_response(
                mock_response.model_dump()
            )
            await litellm.aresponses(
                input="Say OK",
                model="openai/gpt-4o-mini",
                metadata=original_metadata,
            )

        await asyncio.wait_for(callback.event.wait(), timeout=5.0)

        # The original dict should not have been mutated by litellm internals
        # (litellm stamps model_group, litellm_call_id, etc. into metadata)
        assert (
            "model_group" not in original_metadata
        ), "Caller's metadata dict should not be mutated by litellm internals"
    finally:
        litellm.callbacks = original_callbacks
