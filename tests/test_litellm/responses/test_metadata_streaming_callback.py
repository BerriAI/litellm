"""
Test that metadata is correctly forwarded to callbacks for both native and
completion-transformation Responses API paths.

Fixes issue #15466: Metadata not forwarded to Langfuse/custom callbacks when
using /v1/responses endpoint, especially for providers without native Responses
API support (bedrock, anthropic via litellm completion bridge).

Root causes:
  1. (Native path) `metadata or kwargs.get("litellm_metadata") or {}` uses Python
     truthiness — empty dict {} is falsy, falling through to litellm_metadata.
  2. (Completion transformation path) The `metadata` parameter is consumed by the
     responses() function signature and is NOT included in **kwargs. When
     response_api_handler(**kwargs) is called, metadata is lost entirely.

Fixes:
  1. Use explicit `is not None` check for the native path.
  2. Inject metadata into kwargs before the early return for the completion
     transformation path.
"""

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest

import litellm


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

        def iter_lines(self):
            yield f"data: {json.dumps(self._json_data)}"
            yield "data: [DONE]"

        async def aiter_lines(self):
            yield f"data: {json.dumps(self._json_data)}"
            yield "data: [DONE]"

    return MockResponse(response_dict, 200)


MOCK_RESPONSE_DICT = {
    "id": "resp-metadata-test",
    "created_at": 0,
    "output": [
        {
            "type": "message",
            "id": "msg-1",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Hello!"}],
        }
    ],
    "object": "response",
    "model": "gpt-4o",
    "status": "completed",
    "usage": {
        "input_tokens": 5,
        "output_tokens": 10,
        "total_tokens": 15,
    },
}

async def _consume_stream(response_stream):
    async for _ in response_stream:
        pass


@pytest.fixture(autouse=True)
def _reset_callbacks():
    """Reset and restore litellm callbacks around each test for isolation."""
    original = litellm.callbacks.copy() if litellm.callbacks else []
    litellm.callbacks = []
    yield
    litellm.callbacks = original


# ---------------------------------------------------------------------------
# Native Responses API path (providers with direct Responses API support)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_native_path_user_metadata_preserved():
    """
    Native path (streaming): user metadata must be forwarded to callbacks,
    not overridden by litellm_metadata.
    """
    from litellm.types.llms.openai import ResponsesAPIResponse

    mock_response = ResponsesAPIResponse.model_construct(**MOCK_RESPONSE_DICT)
    user_metadata = {"caller": "test_streaming", "organization_id": "org-123"}
    proxy_litellm_metadata = {
        "requester_metadata": {"caller": "test_streaming"},
        "user_api_key": "sk-1234",
        "headers": {"Authorization": "Bearer sk-1234"},
    }

    # MetadataCaptureCallback not needed here — assertions use
    # response_stream.logging_obj.model_call_details directly.

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        mock_post.return_value = _make_mock_http_response(
            mock_response.model_dump()
        )
        response_stream = await litellm.aresponses(
            model="gpt-4o",
            input="Say hello",
            metadata=user_metadata,
            litellm_metadata=proxy_litellm_metadata,
            stream=True,
        )
        await _consume_stream(response_stream)

    litellm_params = response_stream.logging_obj.model_call_details.get(
        "litellm_params", {}
    )
    metadata = litellm_params.get("metadata", {})

    assert metadata.get("caller") == "test_streaming"
    assert metadata.get("organization_id") == "org-123"
    assert "user_api_key" not in metadata
    assert "requester_metadata" not in metadata


@pytest.mark.asyncio
async def test_native_path_empty_metadata_no_fallthrough():
    """
    Native path (streaming): empty dict {} must not fall through to
    litellm_metadata (Python truthiness bug).
    """
    from litellm.types.llms.openai import ResponsesAPIResponse

    mock_response = ResponsesAPIResponse.model_construct(**MOCK_RESPONSE_DICT)
    proxy_litellm_metadata = {
        "requester_metadata": {},
        "user_api_key": "sk-secret",
        "headers": {"Authorization": "Bearer sk-secret"},
    }

    # MetadataCaptureCallback not needed here — assertions use
    # response_stream.logging_obj.model_call_details directly.

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        mock_post.return_value = _make_mock_http_response(
            mock_response.model_dump()
        )
        response_stream = await litellm.aresponses(
            model="gpt-4o",
            input="Say hello",
            metadata={},
            litellm_metadata=proxy_litellm_metadata,
            stream=True,
        )
        await _consume_stream(response_stream)

    litellm_params = response_stream.logging_obj.model_call_details.get(
        "litellm_params", {}
    )
    metadata = litellm_params.get("metadata", {})

    # Must be exactly empty dict — no litellm_metadata leak
    assert metadata == {}
    assert "user_api_key" not in metadata
    assert "requester_metadata" not in metadata
    assert "headers" not in metadata


@pytest.mark.asyncio
async def test_native_path_none_metadata_uses_litellm_metadata():
    """
    Native path: when metadata is None (codex bridge), litellm_metadata is used.
    """
    from litellm.types.llms.openai import ResponsesAPIResponse

    mock_response = ResponsesAPIResponse.model_construct(**MOCK_RESPONSE_DICT)
    bridge_litellm_metadata = {"request_id": "req-789", "trace_id": "trace-abc"}

    # MetadataCaptureCallback not needed here — assertions use
    # response_stream.logging_obj.model_call_details directly.

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        mock_post.return_value = _make_mock_http_response(
            mock_response.model_dump()
        )
        response_stream = await litellm.aresponses(
            model="gpt-4o",
            input="Say hello",
            litellm_metadata=bridge_litellm_metadata,
            stream=True,
        )
        await _consume_stream(response_stream)

    litellm_params = response_stream.logging_obj.model_call_details.get(
        "litellm_params", {}
    )
    metadata = litellm_params.get("metadata", {})

    assert metadata.get("request_id") == "req-789"
    assert metadata.get("trace_id") == "trace-abc"


@pytest.mark.asyncio
async def test_native_path_no_metadata_defaults_to_empty():
    """
    Native path: when neither metadata nor litellm_metadata is provided,
    callback metadata should be empty.
    """
    from litellm.types.llms.openai import ResponsesAPIResponse

    mock_response = ResponsesAPIResponse.model_construct(**MOCK_RESPONSE_DICT)

    # MetadataCaptureCallback not needed here — assertions use
    # response_stream.logging_obj.model_call_details directly.

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        mock_post.return_value = _make_mock_http_response(
            mock_response.model_dump()
        )
        response_stream = await litellm.aresponses(
            model="gpt-4o",
            input="Say hello",
            stream=True,
        )
        await _consume_stream(response_stream)

    litellm_params = response_stream.logging_obj.model_call_details.get(
        "litellm_params", {}
    )
    metadata = litellm_params.get("metadata", {})

    assert "user_api_key" not in metadata
    assert "requester_metadata" not in metadata


@pytest.mark.asyncio
async def test_native_path_proxy_keys_not_leaked():
    """
    Native path: proxy-internal keys must not appear in callback metadata
    when user provides their own metadata.
    """
    from litellm.types.llms.openai import ResponsesAPIResponse

    mock_response = ResponsesAPIResponse.model_construct(**MOCK_RESPONSE_DICT)
    user_metadata = {"session_id": "sess-001"}
    proxy_litellm_metadata = {
        "requester_metadata": {"session_id": "sess-001"},
        "user_api_key": "sk-proxy",
        "headers": {"Authorization": "Bearer sk-proxy"},
        "requester_ip_address": "127.0.0.1",
    }

    # MetadataCaptureCallback not needed here — assertions use
    # response_stream.logging_obj.model_call_details directly.

    with patch(
        "litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post",
        new_callable=AsyncMock,
    ) as mock_post:
        mock_post.return_value = _make_mock_http_response(
            mock_response.model_dump()
        )
        response_stream = await litellm.aresponses(
            model="gpt-4o",
            input="Say hello",
            metadata=user_metadata,
            litellm_metadata=proxy_litellm_metadata,
            stream=True,
        )
        await _consume_stream(response_stream)

    litellm_params = response_stream.logging_obj.model_call_details.get(
        "litellm_params", {}
    )
    metadata = litellm_params.get("metadata", {})

    assert metadata.get("session_id") == "sess-001"
    assert "user_api_key" not in metadata
    assert "requester_metadata" not in metadata
    assert "requester_ip_address" not in metadata


# ---------------------------------------------------------------------------
# Completion transformation path (providers WITHOUT native Responses API —
# bedrock, anthropic-without-responses, etc.)
# This is the MAIN bug from #15466: metadata is consumed by responses() as a
# named parameter and never forwarded into **kwargs for response_api_handler().
# ---------------------------------------------------------------------------


def test_completion_path_metadata_forwarded():
    """
    Completion transformation path: user metadata must reach callbacks
    even though responses() consumes it as a named parameter.

    This is the primary bug in #15466. Before the fix, metadata was lost
    because it was consumed by responses()'s function signature and not
    included in **kwargs when response_api_handler(**kwargs) was called.
    """
    user_metadata = {"caller": "langfuse-test", "environment": "staging"}

    # MetadataCaptureCallback not needed — assertions verify metadata via
    # mock_completion.call_args, not callback.captured_kwargs.

    with patch(
        "litellm.completion",
    ) as mock_completion:
        from litellm.types.utils import ModelResponse, Choices, Message, Usage

        mock_resp = ModelResponse(
            id="chatcmpl-test",
            model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
            choices=[Choices(index=0, message=Message(role="assistant", content="Hello!"), finish_reason="stop")],
            usage=Usage(prompt_tokens=5, completion_tokens=10, total_tokens=15),
        )
        mock_completion.return_value = mock_resp

        # Force the completion transformation path by making the provider
        # config return None
        with patch(
            "litellm.utils.ProviderConfigManager.get_provider_responses_api_config",
            return_value=None,
        ):
            litellm.responses(
                model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
                input="Say hello",
                metadata=user_metadata,
            )

    # Verify that metadata was passed to litellm.completion
    assert mock_completion.called
    call_kwargs = mock_completion.call_args
    # Use explicit `is not None` check — `or` treats {} as falsy (#15466)
    completion_metadata = call_kwargs.kwargs.get("metadata")
    if completion_metadata is None and len(call_kwargs) > 1:
        completion_metadata = call_kwargs[1].get("metadata")

    assert completion_metadata is not None, (
        "metadata was not forwarded to litellm.completion — "
        "the responses() function consumed it as a named parameter "
        "and did not inject it into **kwargs"
    )
    assert completion_metadata.get("caller") == "langfuse-test"
    assert completion_metadata.get("environment") == "staging"


@pytest.mark.asyncio
async def test_completion_path_metadata_forwarded_async():
    """
    Same as test_completion_path_metadata_forwarded but for the async path.
    """
    user_metadata = {"trace_id": "abc-123", "user_id": "u-42"}

    # MetadataCaptureCallback not needed — assertions verify metadata via
    # mock_acompletion.call_args, not callback.captured_kwargs.

    with patch(
        "litellm.acompletion",
        new_callable=AsyncMock,
    ) as mock_acompletion:
        from litellm.types.utils import ModelResponse, Choices, Message, Usage

        mock_resp = ModelResponse(
            id="chatcmpl-async-test",
            model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
            choices=[Choices(index=0, message=Message(role="assistant", content="Hi!"), finish_reason="stop")],
            usage=Usage(prompt_tokens=5, completion_tokens=10, total_tokens=15),
        )
        mock_acompletion.return_value = mock_resp

        with patch(
            "litellm.utils.ProviderConfigManager.get_provider_responses_api_config",
            return_value=None,
        ):
            await litellm.aresponses(
                model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
                input="Say hello",
                metadata=user_metadata,
            )

    assert mock_acompletion.called
    call_kwargs = mock_acompletion.call_args
    # Use explicit `is not None` check — `or` treats {} as falsy (#15466)
    completion_metadata = call_kwargs.kwargs.get("metadata")
    if completion_metadata is None and len(call_kwargs) > 1:
        completion_metadata = call_kwargs[1].get("metadata")

    assert completion_metadata is not None, (
        "metadata was not forwarded to litellm.acompletion"
    )
    assert completion_metadata.get("trace_id") == "abc-123"
    assert completion_metadata.get("user_id") == "u-42"


def test_completion_path_none_metadata_not_injected():
    """
    Completion transformation path: when metadata is None, we must NOT
    inject None into kwargs — let litellm_metadata flow through instead.
    """
    with patch(
        "litellm.completion",
    ) as mock_completion:
        from litellm.types.utils import ModelResponse, Choices, Message, Usage

        mock_resp = ModelResponse(
            id="chatcmpl-none-test",
            model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
            choices=[Choices(index=0, message=Message(role="assistant", content="Hi"), finish_reason="stop")],
            usage=Usage(prompt_tokens=5, completion_tokens=10, total_tokens=15),
        )
        mock_completion.return_value = mock_resp

        with patch(
            "litellm.utils.ProviderConfigManager.get_provider_responses_api_config",
            return_value=None,
        ):
            litellm.responses(
                model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
                input="Say hello",
                # metadata not provided → defaults to None
                litellm_metadata={"proxy_key": "val"},
            )

    assert mock_completion.called
    call_kwargs = mock_completion.call_args
    # metadata should come from transformation.py kwargs.get("metadata")
    # which will be None since we didn't inject it (metadata was None)
    # But litellm_metadata should still be passed through in kwargs
    # Use explicit `is not None` — `if call_kwargs.kwargs` treats {} as falsy
    all_kwargs = call_kwargs.kwargs if call_kwargs.kwargs is not None else call_kwargs[1]
    assert all_kwargs.get("litellm_metadata", {}).get("proxy_key") == "val"
