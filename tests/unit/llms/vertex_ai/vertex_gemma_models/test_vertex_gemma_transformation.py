"""
Mocked tests for Vertex AI Gemma Models

Maps to: litellm/llms/vertex_ai/vertex_gemma_models/transformation.py
"""

import json
from collections.abc import AsyncIterator
from typing import cast
from unittest.mock import AsyncMock, Mock, patch

import pytest

import litellm
from litellm.types.llms.openai import (
    OutputTextDeltaEvent,
    ResponseCompletedEvent,
    ResponsesAPIStreamingResponse,
)


@pytest.fixture(autouse=True)
def _reset_litellm_http_client_cache():
    """Ensure each test gets a fresh async HTTP client mock."""
    from litellm import in_memory_llm_clients_cache

    in_memory_llm_clients_cache.flush_cache()


def _make_gemma_vertex_response(
    content="ok",
    response_id="chatcmpl-test",
    total_tokens=114,
):
    """Build a minimal but valid Vertex Gemma `predictions` response body."""
    return {
        "deployedModelId": "1207280419999999999",
        "model": "projects/993702345710/locations/us-central1/models/gemma-3-12b-it-1222199011122",
        "modelDisplayName": "gemma-3-12b-it-1222199011122",
        "modelVersionId": "1",
        "predictions": {
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "logprobs": None,
                    "message": {
                        "content": content,
                        "reasoning_content": None,
                        "role": "assistant",
                        "tool_calls": [],
                    },
                    "stop_reason": None,
                }
            ],
            "created": 1759863903,
            "id": response_id,
            "model": "google/gemma-3-12b-it",
            "object": "chat.completion",
            "prompt_logprobs": None,
            "usage": {
                "completion_tokens": 100,
                "prompt_tokens": 14,
                "prompt_tokens_details": None,
                "total_tokens": total_tokens,
            },
        },
    }


class TestVertexGemmaCompletion:
    """Test completion flow for Vertex AI Gemma models using litellm.acompletion()"""

    @pytest.mark.asyncio
    async def test_acompletion_basic_request(self):
        """
        Test litellm.acompletion() with Vertex AI Gemma model

        Expected URL:
        https://32277599999999999.us-central1-10582012152.prediction.vertexai.goog/v1/projects/PROJECT_ID/locations/us-central1/endpoints/ENDPOINT_ID:predict

        Expected Request Body (sent to Vertex):
        {
            "instances": [
                {
                    "@requestFormat": "chatCompletions",
                    "messages": [
                        {
                            "role": "user",
                            "content": "What is machine learning?"
                        }
                    ],
                    "max_tokens": 100
                }
            ]
        }

        Expected Vertex Response:
        {
            "deployedModelId": "1207280419999999999",
            "model": "projects/993702345710/locations/us-central1/models/gemma-3-12b-it-1222199011122",
            "modelDisplayName": "gemma-3-12b-it-1222199011122",
            "modelVersionId": "1",
            "predictions": {
                "choices": [
                    {
                        "finish_reason": "length",
                        "index": 0,
                        "logprobs": null,
                        "message": {
                            "content": "Okay, let's break down machine learning...",
                            "reasoning_content": null,
                            "role": "assistant",
                            "tool_calls": []
                        },
                        "stop_reason": null
                    }
                ],
                "created": 1759863903,
                "id": "chatcmpl-aaa4288f-2b8e-4bc0-8b14-4e444decd2c4",
                "model": "google/gemma-3-12b-it",
                "object": "chat.completion",
                "prompt_logprobs": null,
                "usage": {
                    "completion_tokens": 100,
                    "prompt_tokens": 14,
                    "prompt_tokens_details": null,
                    "total_tokens": 114
                }
            }
        }

        Expected LiteLLM Response: Standard OpenAI format
        """
        # Real Vertex response from user's spec
        mock_vertex_response = {
            "deployedModelId": "1207280419999999999",
            "model": "projects/993702345710/locations/us-central1/models/gemma-3-12b-it-1222199011122",
            "modelDisplayName": "gemma-3-12b-it-1222199011122",
            "modelVersionId": "1",
            "predictions": {
                "choices": [
                    {
                        "finish_reason": "length",
                        "index": 0,
                        "logprobs": None,
                        "message": {
                            "content": "Okay, let's break down machine learning. Here's a comprehensive explanation, covering the core concepts, types, and some examples, tailored to different levels of understanding.  I'll structure it into sections: **The Core Idea**, **Types of Machine Learning**, **How It Works (Simplified)**, **Examples**, and **Why It's Useful**.\n\n**1. The Core Idea: Learning from Data**\n\nAt its heart, machine learning (ML) is about enabling computers",
                            "reasoning_content": None,
                            "role": "assistant",
                            "tool_calls": [],
                        },
                        "stop_reason": None,
                    }
                ],
                "created": 1759863903,
                "id": "chatcmpl-aaa4288f-2b8e-4bc0-8b14-4e444decd2c4",
                "model": "google/gemma-3-12b-it",
                "object": "chat.completion",
                "prompt_logprobs": None,
                "usage": {
                    "completion_tokens": 100,
                    "prompt_tokens": 14,
                    "prompt_tokens_details": None,
                    "total_tokens": 114,
                },
            },
        }

        # Mock the async HTTP handler and Vertex authentication
        with (
            patch("litellm.llms.custom_httpx.http_handler.get_async_httpx_client") as mock_get_client,
            patch(
                "litellm.llms.vertex_ai.vertex_gemma_models.main.VertexAIGemmaModels._ensure_access_token",
                return_value=("fake-access-token", "PROJECT_ID"),
            ),
        ):
            mock_client = Mock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_vertex_response
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            # Call litellm.acompletion()
            response = await litellm.acompletion(
                model="vertex_ai/gemma/gemma-3-12b-it-1222199011122",
                messages=[{"role": "user", "content": "What is machine learning?"}],
                max_tokens=100,
                api_base="https://32277599999999999.us-central1-10582012152.prediction.vertexai.goog/v1/projects/PROJECT_ID/locations/us-central1/endpoints/ENDPOINT_ID:predict",
                vertex_project="PROJECT_ID",
                vertex_location="us-central1",
            )

            # Verify the request sent to Vertex
            call_args = mock_client.post.call_args
            assert call_args is not None, "HTTP handler was not called"

            request_data = call_args.kwargs["json"]
            request_url = call_args.kwargs["url"]

            # Validate exact URL matches what we sent
            expected_url = "https://32277599999999999.us-central1-10582012152.prediction.vertexai.goog/v1/projects/PROJECT_ID/locations/us-central1/endpoints/ENDPOINT_ID:predict"
            assert request_url == expected_url, f"Expected URL: {expected_url}\nActual URL: {request_url}"

            # Validate Request Body matches expected format
            assert "instances" in request_data
            assert len(request_data["instances"]) == 1

            instance = request_data["instances"][0]
            assert instance["@requestFormat"] == "chatCompletions"

            # Messages should be directly in the instance, not double-nested
            assert "messages" in instance
            assert instance["messages"][0]["role"] == "user"
            assert instance["messages"][0]["content"] == "What is machine learning?"
            assert instance["max_tokens"] == 100

            # Verify stream parameter is NOT sent to Vertex (will be faked client-side)
            assert "stream" not in instance

            # Validate LiteLLM Response (OpenAI format)
            assert response.id == "chatcmpl-aaa4288f-2b8e-4bc0-8b14-4e444decd2c4"
            assert response.object == "chat.completion"
            assert response.created == 1759863903
            # Model name has the gemma/ prefix stripped during processing
            assert response.model == "gemma-3-12b-it-1222199011122"

            # Validate choices
            assert len(response.choices) == 1
            assert response.choices[0].index == 0
            assert response.choices[0].finish_reason == "length"
            assert response.choices[0].message.role == "assistant"
            assert "machine learning" in response.choices[0].message.content.lower()

            # Validate usage
            assert response.usage.prompt_tokens == 14
            assert response.usage.completion_tokens == 100
            assert response.usage.total_tokens == 114

    @pytest.mark.asyncio
    async def test_acompletion_error_handling(self):
        """
        Test litellm.acompletion() error handling when Vertex returns invalid response

        Expected: Proper error handling when 'predictions' field is missing
        """
        from litellm.exceptions import BadRequestError

        # Invalid response without predictions field
        invalid_response = {
            "deployedModelId": "123",
            "error": {"code": 400, "message": "Invalid request"},
        }

        with (
            patch("litellm.llms.custom_httpx.http_handler.get_async_httpx_client") as mock_get_client,
            patch(
                "litellm.llms.vertex_ai.vertex_gemma_models.main.VertexAIGemmaModels._ensure_access_token",
                return_value=("fake-access-token", "test-project"),
            ),
        ):
            mock_client = Mock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = invalid_response
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            # Should raise exception (wrapped as BadRequestError by LiteLLM)
            with pytest.raises(BadRequestError) as exc_info:
                await litellm.acompletion(
                    model="vertex_ai/gemma/gemma-3-12b-it",
                    messages=[{"role": "user", "content": "Test"}],
                    api_base="https://test.prediction.vertexai.goog/v1/projects/test/locations/us-central1/endpoints/123:predict",
                    vertex_project="test-project",
                    vertex_location="us-central1",
                )

            # Verify the error message contains the original error
            assert "missing 'predictions' field" in str(exc_info.value)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stream", [False, True])
    async def test_acompletion_surfaces_container_error_object_as_its_own_status_and_message(self, stream):
        """
        A serving container can reject the request with its own OpenAI-shaped error object,
        which Vertex still wraps in an HTTP 200 :predict response. The container's status and
        message must reach the caller instead of a 500 "no 'choices'".
        """
        from litellm.exceptions import BadRequestError

        container_message = '"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set'
        vertex_response = {
            "deployedModelId": "123",
            "predictions": {
                "code": 400,
                "message": container_message,
                "object": "error",
                "param": None,
                "type": "BadRequestError",
            },
        }

        with (
            patch("litellm.llms.custom_httpx.http_handler.get_async_httpx_client") as mock_get_client,
            patch(
                "litellm.llms.vertex_ai.vertex_gemma_models.main.VertexAIGemmaModels._ensure_access_token",
                return_value=("fake-access-token", "test-project"),
            ),
        ):
            mock_client = Mock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = vertex_response
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            with pytest.raises(BadRequestError) as exc_info:
                await litellm.acompletion(
                    model="vertex_ai/gemma/gemma-2-2b-it",
                    messages=[{"role": "user", "content": "What is the weather in Paris?"}],
                    tools=[{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
                    stream=stream,
                    api_base="https://test.prediction.vertexai.goog/v1/projects/test/locations/us-central1/endpoints/123:predict",
                    vertex_project="test-project",
                    vertex_location="us-central1",
                )

        assert exc_info.value.status_code == 400
        assert container_message in str(exc_info.value)
        assert "no 'choices'" not in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_acompletion_keeps_container_error_status_beyond_400(self):
        """The container's status is forwarded as is, not collapsed to 400."""
        from litellm.exceptions import RateLimitError

        vertex_response = {
            "deployedModelId": "123",
            "predictions": {"code": 429, "message": "engine overloaded", "object": "error", "type": "RateLimitError"},
        }

        with (
            patch("litellm.llms.custom_httpx.http_handler.get_async_httpx_client") as mock_get_client,
            patch(
                "litellm.llms.vertex_ai.vertex_gemma_models.main.VertexAIGemmaModels._ensure_access_token",
                return_value=("fake-access-token", "test-project"),
            ),
        ):
            mock_client = Mock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = vertex_response
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            with pytest.raises(RateLimitError) as exc_info:
                await litellm.acompletion(
                    model="vertex_ai/gemma/gemma-2-2b-it",
                    messages=[{"role": "user", "content": "Test"}],
                    api_base="https://test.prediction.vertexai.goog/v1/projects/test/locations/us-central1/endpoints/123:predict",
                    vertex_project="test-project",
                    vertex_location="us-central1",
                )

        assert exc_info.value.status_code == 429
        assert "engine overloaded" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_acompletion_error_object_without_http_status_keeps_generic_handling(self):
        """An error-shaped body whose code is not an HTTP error status is not trusted as one."""
        from litellm.exceptions import APIError

        vertex_response = {
            "deployedModelId": "123",
            "predictions": {"code": 0, "message": "unknown failure", "object": "error"},
        }

        with (
            patch("litellm.llms.custom_httpx.http_handler.get_async_httpx_client") as mock_get_client,
            patch(
                "litellm.llms.vertex_ai.vertex_gemma_models.main.VertexAIGemmaModels._ensure_access_token",
                return_value=("fake-access-token", "test-project"),
            ),
        ):
            mock_client = Mock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = vertex_response
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            with pytest.raises(APIError) as exc_info:
                await litellm.acompletion(
                    model="vertex_ai/gemma/gemma-2-2b-it",
                    messages=[{"role": "user", "content": "Test"}],
                    api_base="https://test.prediction.vertexai.goog/v1/projects/test/locations/us-central1/endpoints/123:predict",
                    vertex_project="test-project",
                    vertex_location="us-central1",
                )

        assert exc_info.value.status_code == 500

    def test_sync_completion_surfaces_container_error_object_as_its_own_status_and_message(self):
        """The synchronous path unwraps the same container error object."""
        from litellm.exceptions import BadRequestError

        container_message = '"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set'
        vertex_response = {
            "deployedModelId": "123",
            "predictions": {"code": 400, "message": container_message, "object": "error", "type": "BadRequestError"},
        }

        with (
            patch("litellm.llms.vertex_ai.vertex_gemma_models.transformation._get_httpx_client") as mock_get_client,
            patch(
                "litellm.llms.vertex_ai.vertex_gemma_models.main.VertexAIGemmaModels._ensure_access_token",
                return_value=("fake-access-token", "PROJECT_ID"),
            ),
        ):
            mock_client = Mock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = vertex_response
            mock_client.post = Mock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            with pytest.raises(BadRequestError) as exc_info:
                litellm.completion(
                    model="vertex_ai/gemma/gemma-2-2b-it",
                    messages=[{"role": "user", "content": "What is the weather in Paris?"}],
                    tools=[{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
                    api_base="https://test.prediction.vertexai.goog/v1/projects/PROJECT_ID/locations/us-central1/endpoints/ENDPOINT_ID:predict",
                    vertex_project="PROJECT_ID",
                    vertex_location="us-central1",
                )

        assert exc_info.value.status_code == 400
        assert container_message in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_acompletion_fake_streaming(self):
        """
        Test that streaming requests are faked properly for Vertex AI Gemma models.

        Verifies:
        1. Request body does NOT include 'stream' parameter (model doesn't support it)
        2. Response wraps a MockResponseIterator and yields chunks
        """
        from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
        from litellm.llms.base_llm.base_model_iterator import MockResponseIterator

        # Mock Vertex response
        mock_vertex_response = {
            "deployedModelId": "1207280419999999999",
            "model": "projects/993702345710/locations/us-central1/models/gemma-3-12b-it-1222199011122",
            "modelDisplayName": "gemma-3-12b-it-1222199011122",
            "modelVersionId": "1",
            "predictions": {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "logprobs": None,
                        "message": {
                            "content": "Streaming test response",
                            "reasoning_content": None,
                            "role": "assistant",
                            "tool_calls": [],
                        },
                        "stop_reason": None,
                    }
                ],
                "created": 1759863903,
                "id": "chatcmpl-test-stream",
                "model": "google/gemma-3-12b-it",
                "object": "chat.completion",
                "prompt_logprobs": None,
                "usage": {
                    "completion_tokens": 3,
                    "prompt_tokens": 10,
                    "prompt_tokens_details": None,
                    "total_tokens": 13,
                },
            },
        }

        with (
            patch("litellm.llms.custom_httpx.http_handler.get_async_httpx_client") as mock_get_client,
            patch(
                "litellm.llms.vertex_ai.vertex_gemma_models.main.VertexAIGemmaModels._ensure_access_token",
                return_value=("fake-access-token", "PROJECT_ID"),
            ),
        ):
            mock_client = Mock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_vertex_response
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            # Call litellm.acompletion() with stream=True
            response = await litellm.acompletion(
                model="vertex_ai/gemma/gemma-3-12b-it-1222199011122",
                messages=[{"role": "user", "content": "Test streaming"}],
                stream=True,
                api_base="https://test.us-central1-project.prediction.vertexai.goog/v1/projects/PROJECT_ID/locations/us-central1/endpoints/ENDPOINT_ID:predict",
                vertex_project="PROJECT_ID",
                vertex_location="us-central1",
            )

            assert isinstance(response, CustomStreamWrapper)
            assert isinstance(response.completion_stream, MockResponseIterator)

            # Verify the request sent to Vertex does NOT include 'stream'
            call_args = mock_client.post.call_args
            assert call_args is not None, "HTTP client was not called"

            request_data = call_args.kwargs["json"]
            instance = request_data["instances"][0]

            # Critical: Verify stream parameter is NOT sent to Vertex API
            assert "stream" not in instance, "stream parameter should not be sent to Vertex API"

            # Verify we can iterate the fake stream and get the response
            chunks = []
            async for chunk in response:
                chunks.append(chunk)

            assert len(chunks) == 2
            assert chunks[1].choices[0].finish_reason == "stop"
            assert all(getattr(chunk, "usage", None) is None for chunk in chunks)

            # Verify the chunk has the expected content
            chunk = chunks[0]
            assert hasattr(chunk, "choices")
            assert len(chunk.choices) > 0
            assert chunk.choices[0].delta.content == "Streaming test response"

    @pytest.mark.asyncio
    async def test_aresponses_streams_vertex_gemma_with_llm_tracing(self):
        pytest.importorskip("ddtrace")
        from ddtrace.contrib.internal.litellm.patch import patch as patch_litellm
        from ddtrace.contrib.internal.litellm.patch import unpatch as unpatch_litellm
        from ddtrace.llmobs._integrations.base_stream_handler import TracedAsyncStream

        from litellm.responses.litellm_completion_transformation.streaming_iterator import (
            LiteLLMCompletionStreamingIterator,
        )

        reply = Mock(status_code=200)
        reply.json.return_value = _make_gemma_vertex_response(content="READY")
        client = Mock()
        client.post = AsyncMock(return_value=reply)

        with (
            patch("litellm.llms.custom_httpx.http_handler.get_async_httpx_client", return_value=client),
            patch(
                "litellm.llms.vertex_ai.vertex_gemma_models.main.VertexAIGemmaModels._ensure_access_token",
                return_value=("fake-access-token", "test-project"),
            ),
        ):
            patch_litellm()
            try:
                response = await litellm.aresponses(
                    model="vertex_ai/gemma/test-model",
                    input="Reply exactly READY",
                    stream=True,
                    api_base="https://example.invalid/v1/projects/test-project/locations/us-central1/endpoints/test:predict",
                    vertex_project="test-project",
                    vertex_location="us-central1",
                )
                bridge = cast(LiteLLMCompletionStreamingIterator, response)
                traced_stream = bridge.litellm_custom_stream_wrapper
                assert isinstance(traced_stream, TracedAsyncStream)
                events = [event async for event in cast(AsyncIterator[ResponsesAPIStreamingResponse], response)]
                span = traced_stream.handler.primary_span
                assert span.finished
                assert span.get_tag("_dd.llmobs.span_kind") == "llm"
                assert span.get_metric("_dd.llmobs.total_tokens") == 114
            finally:
                unpatch_litellm()

        assert "stream" not in client.post.call_args.kwargs["json"]["instances"][0]
        assert "READY" in "".join(event.delta for event in events if isinstance(event, OutputTextDeltaEvent))
        assert isinstance(events[-1], ResponseCompletedEvent)
        assert events[-1].response.usage.total_tokens == 114

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stream_options", [None, {"include_usage": False}, {"include_usage": True}])
    async def test_acompletion_stream_respects_usage_option_with_llm_tracing(self, stream_options):
        pytest.importorskip("ddtrace")
        from ddtrace.contrib.internal.litellm.patch import patch as patch_litellm
        from ddtrace.contrib.internal.litellm.patch import unpatch as unpatch_litellm

        reply = Mock(status_code=200)
        reply.json.return_value = _make_gemma_vertex_response(content="READY")
        client = Mock(post=AsyncMock(return_value=reply))
        with (
            patch("litellm.llms.custom_httpx.http_handler.get_async_httpx_client", return_value=client),
            patch(
                "litellm.llms.vertex_ai.vertex_gemma_models.main.VertexAIGemmaModels._ensure_access_token",
                return_value=("fake-access-token", "test-project"),
            ),
        ):
            patch_litellm()
            try:
                stream = await litellm.acompletion(
                    model="vertex_ai/gemma/test-model",
                    messages=[{"role": "user", "content": "Reply exactly READY"}],
                    stream=True,
                    **({"stream_options": stream_options} if stream_options is not None else {}),
                    api_base="https://example.invalid/v1/projects/test-project/locations/us-central1/endpoints/test:predict",
                    vertex_project="test-project",
                    vertex_location="us-central1",
                )
                chunks = [chunk async for chunk in stream]
                span = stream.handler.primary_span
                assert span.finished
                assert span.get_tag("_dd.llmobs.span_kind") == "llm"
            finally:
                unpatch_litellm()

        assert len(chunks) == (3 if stream_options and stream_options["include_usage"] else 2)
        assert chunks[0].choices[0].delta.content == "READY"
        assert chunks[1].choices[0].finish_reason == "stop"
        if stream_options and stream_options["include_usage"]:
            assert chunks[-1].choices[0].delta.content is None
            assert chunks[-1].usage.total_tokens == 114
            assert span.get_metric("_dd.llmobs.total_tokens") == 114
        else:
            assert all(getattr(chunk, "usage", None) is None for chunk in chunks)
            # The wrapper still accumulates the provider's usage for LiteLLM's own
            # accounting; ddtrace only reads emitted chunks, so without an explicit
            # include_usage request its span carries no token metric.
            from litellm.litellm_core_utils.streaming_handler import calculate_total_usage

            assert calculate_total_usage(chunks=stream.chunks).total_tokens == 114
            assert span.get_metric("_dd.llmobs.total_tokens") is None

    @pytest.mark.asyncio
    async def test_acompletion_filters_stream_and_stream_options(self):
        """
        Test that both stream and stream_options are filtered out from the request.

        Verifies that when stream=True and stream_options={'include_usage': True} are passed,
        neither parameter is sent to the Vertex API since Vertex Gemma doesn't support them.
        """
        # Mock Vertex response
        mock_vertex_response = {
            "deployedModelId": "1207280419999999999",
            "model": "projects/993702345710/locations/us-central1/models/gemma-3-12b-it-1222199011122",
            "modelDisplayName": "gemma-3-12b-it-1222199011122",
            "modelVersionId": "1",
            "predictions": {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "logprobs": None,
                        "message": {
                            "content": "Test response",
                            "reasoning_content": None,
                            "role": "assistant",
                            "tool_calls": [],
                        },
                        "stop_reason": None,
                    }
                ],
                "created": 1759863903,
                "id": "chatcmpl-test",
                "model": "google/gemma-3-12b-it",
                "object": "chat.completion",
                "prompt_logprobs": None,
                "usage": {
                    "completion_tokens": 2,
                    "prompt_tokens": 10,
                    "prompt_tokens_details": None,
                    "total_tokens": 12,
                },
            },
        }

        with (
            patch("litellm.llms.custom_httpx.http_handler.get_async_httpx_client") as mock_get_client,
            patch(
                "litellm.llms.vertex_ai.vertex_gemma_models.main.VertexAIGemmaModels._ensure_access_token",
                return_value=("fake-access-token", "PROJECT_ID"),
            ),
        ):
            mock_client = Mock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_vertex_response
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            await litellm.acompletion(
                model="vertex_ai/gemma/gemma-3-12b-it-1222199011122",
                messages=[{"role": "user", "content": "Test"}],
                stream=True,
                stream_options={"include_usage": True},
                api_base="https://test.us-central1-project.prediction.vertexai.goog/v1/projects/PROJECT_ID/locations/us-central1/endpoints/ENDPOINT_ID:predict",
                vertex_project="PROJECT_ID",
                vertex_location="us-central1",
            )

            # Verify the request sent to Vertex
            call_args = mock_client.post.call_args
            assert call_args is not None, "HTTP client was not called"

            request_data = call_args.kwargs["json"]
            instance = request_data["instances"][0]

            # Critical: Verify both stream and stream_options are NOT sent to Vertex API
            assert "stream" not in instance, "stream parameter should not be sent to Vertex API"
            assert "stream_options" not in instance, "stream_options parameter should not be sent to Vertex API"

            # Verify other parameters are present
            assert "messages" in instance
            assert instance["@requestFormat"] == "chatCompletions"

    @pytest.mark.asyncio
    async def test_acompletion_filters_context_management(self):
        """
        Test that context_management is filtered out from the request.

        Vertex AI Gemma's chatCompletions wrapper does not understand
        `context_management` (an Anthropic / OpenAI Responses API concept).
        It must be stripped from the request body so the upstream endpoint
        does not reject the request with an unknown-field error.
        """
        mock_vertex_response = {
            "deployedModelId": "1207280419999999999",
            "model": "projects/993702345710/locations/us-central1/models/gemma-3-12b-it-1222199011122",
            "modelDisplayName": "gemma-3-12b-it-1222199011122",
            "modelVersionId": "1",
            "predictions": {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "index": 0,
                        "logprobs": None,
                        "message": {
                            "content": "ok",
                            "reasoning_content": None,
                            "role": "assistant",
                            "tool_calls": [],
                        },
                        "stop_reason": None,
                    }
                ],
                "created": 1759863903,
                "id": "chatcmpl-test-ctxmgmt",
                "model": "google/gemma-3-12b-it",
                "object": "chat.completion",
                "prompt_logprobs": None,
                "usage": {
                    "completion_tokens": 1,
                    "prompt_tokens": 5,
                    "prompt_tokens_details": None,
                    "total_tokens": 6,
                },
            },
        }

        with (
            patch("litellm.llms.custom_httpx.http_handler.get_async_httpx_client") as mock_get_client,
            patch(
                "litellm.llms.vertex_ai.vertex_gemma_models.main.VertexAIGemmaModels._ensure_access_token",
                return_value=("fake-access-token", "PROJECT_ID"),
            ),
        ):
            mock_client = Mock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_vertex_response
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            # Use `allowed_openai_params` so context_management actually
            # reaches the transformation layer (otherwise the upstream
            # validator drops it before we can prove the transformation
            # strips it). This mirrors the real-world scenario where a
            # caller explicitly opts in to forwarding an arbitrary param.
            await litellm.acompletion(
                model="vertex_ai/gemma/gemma-3-12b-it-1222199011122",
                messages=[{"role": "user", "content": "Test"}],
                context_management=[{"type": "compaction", "compact_threshold": 200000}],
                allowed_openai_params=["context_management"],
                api_base="https://test.us-central1-project.prediction.vertexai.goog/v1/projects/PROJECT_ID/locations/us-central1/endpoints/ENDPOINT_ID:predict",
                vertex_project="PROJECT_ID",
                vertex_location="us-central1",
            )

            call_args = mock_client.post.call_args
            assert call_args is not None, "HTTP client was not called"

            request_data = call_args.kwargs["json"]
            instance = request_data["instances"][0]

            assert "context_management" not in instance, "context_management should not be forwarded to Vertex Gemma"
            assert instance["@requestFormat"] == "chatCompletions"
            assert "messages" in instance

    @pytest.mark.parametrize(
        "param",
        [
            "prompt_cache_key",
            "prompt_cache_retention",
            "safety_identifier",
            "service_tier",
            "store",
            "web_search_options",
            "modalities",
            "prediction",
            "audio",
            "max_retries",
        ],
    )
    def test_get_supported_openai_params_omits_params_the_predict_endpoint_rejects(self, param: str):
        from litellm.llms.vertex_ai.vertex_gemma_models.transformation import (
            VertexGemmaConfig,
        )

        assert param not in VertexGemmaConfig().get_supported_openai_params(model="gemma-2-2b-it")

    @pytest.mark.asyncio
    async def test_acompletion_drops_prompt_cache_key_when_drop_params_is_set(self):
        with (
            patch("litellm.llms.custom_httpx.http_handler.get_async_httpx_client") as mock_get_client,
            patch(
                "litellm.llms.vertex_ai.vertex_gemma_models.main.VertexAIGemmaModels._ensure_access_token",
                return_value=("fake-access-token", "PROJECT_ID"),
            ),
        ):
            mock_client = Mock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = _make_gemma_vertex_response()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            await litellm.acompletion(
                model="vertex_ai/gemma/gemma-2-2b-it",
                messages=[{"role": "user", "content": "Test"}],
                prompt_cache_key="session-lit8592",
                service_tier="default",
                max_completion_tokens=16,
                drop_params=True,
                api_base="https://test.us-central1-project.prediction.vertexai.goog/v1/projects/PROJECT_ID/locations/us-central1/endpoints/ENDPOINT_ID:predict",
                vertex_project="PROJECT_ID",
                vertex_location="us-central1",
            )

            instance = mock_client.post.call_args.kwargs["json"]["instances"][0]
            assert "prompt_cache_key" not in instance
            assert "service_tier" not in instance
            assert instance["max_tokens"] == 16
            assert instance["messages"] == [{"role": "user", "content": "Test"}]

    @pytest.mark.asyncio
    async def test_acompletion_rejects_prompt_cache_key_before_calling_vertex(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(litellm, "drop_params", False)
        with (
            patch("litellm.llms.custom_httpx.http_handler.get_async_httpx_client") as mock_get_client,
            patch(
                "litellm.llms.vertex_ai.vertex_gemma_models.main.VertexAIGemmaModels._ensure_access_token",
                return_value=("fake-access-token", "PROJECT_ID"),
            ),
        ):
            mock_client = Mock()
            mock_client.post = AsyncMock()
            mock_get_client.return_value = mock_client

            with pytest.raises(litellm.UnsupportedParamsError, match="prompt_cache_key"):
                await litellm.acompletion(
                    model="vertex_ai/gemma/gemma-2-2b-it",
                    messages=[{"role": "user", "content": "Test"}],
                    prompt_cache_key="session-lit8592",
                    drop_params=False,
                    api_base="https://test.us-central1-project.prediction.vertexai.goog/v1/projects/PROJECT_ID/locations/us-central1/endpoints/ENDPOINT_ID:predict",
                    vertex_project="PROJECT_ID",
                    vertex_location="us-central1",
                )

            mock_client.post.assert_not_called()

    def test_transform_request_strips_context_management(self):
        """
        Direct unit test for VertexGemmaConfig.transform_request: verify that
        `context_management` is stripped from `optional_params` regardless of
        how it was supplied to the transformation layer.
        """
        from litellm.llms.vertex_ai.vertex_gemma_models.transformation import (
            VertexGemmaConfig,
        )

        config = VertexGemmaConfig()
        result = config.transform_request(
            model="gemma-3-12b-it",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={
                "max_tokens": 32,
                "context_management": [{"type": "compaction", "compact_threshold": 200000}],
            },
            litellm_params={},
            headers={},
        )

        assert "instances" in result
        instance = result["instances"][0]
        assert instance["@requestFormat"] == "chatCompletions"
        assert "context_management" not in instance
        assert instance.get("max_tokens") == 32

    def test_sync_completion_makes_http_call(self):
        """
        Regression test for the synchronous path.

        A refactor once dropped the `response = http_handler.post(...)` line,
        so every sync Vertex Gemma call raised
        `NameError: name 'response' is not defined` before any response
        handling could run. This drives the real sync code path through
        litellm.completion() and asserts a fully parsed response comes back,
        which only happens if the HTTP call is actually issued.
        """
        vertex_response = _make_gemma_vertex_response(
            content="Machine learning is a field of AI.",
            response_id="chatcmpl-sync-regression",
        )

        with (
            patch("litellm.llms.vertex_ai.vertex_gemma_models.transformation._get_httpx_client") as mock_get_client,
            patch(
                "litellm.llms.vertex_ai.vertex_gemma_models.main.VertexAIGemmaModels._ensure_access_token",
                return_value=("fake-access-token", "PROJECT_ID"),
            ),
        ):
            mock_client = Mock()
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = vertex_response
            mock_client.post = Mock(return_value=mock_response)
            mock_get_client.return_value = mock_client

            response = litellm.completion(
                model="vertex_ai/gemma/gemma-3-12b-it-1222199011122",
                messages=[{"role": "user", "content": "What is machine learning?"}],
                max_tokens=100,
                api_base="https://32277599999999999.us-central1-10582012152.prediction.vertexai.goog/v1/projects/PROJECT_ID/locations/us-central1/endpoints/ENDPOINT_ID:predict",
                vertex_project="PROJECT_ID",
                vertex_location="us-central1",
            )

            # The HTTP call must have been made exactly once
            mock_get_client.assert_called_once()
            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            assert call_args.kwargs["url"].endswith(":predict")
            instance = call_args.kwargs["json"]["instances"][0]
            assert instance["@requestFormat"] == "chatCompletions"

            # And the response must be parsed from what the endpoint returned
            assert response.id == "chatcmpl-sync-regression"
            assert response.model == "gemma-3-12b-it-1222199011122"
            assert response.choices[0].message.content == "Machine learning is a field of AI."
            assert response.usage.total_tokens == 114

    def test_sync_completion_uses_provided_client(self):
        """A caller-supplied sync HTTPHandler must be routed through, not replaced."""
        from litellm.llms.custom_httpx.http_handler import HTTPHandler

        vertex_response = _make_gemma_vertex_response(content="hi from sync client")

        custom_client = Mock(spec=HTTPHandler)
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = vertex_response
        custom_client.post = Mock(return_value=mock_response)

        with patch(
            "litellm.llms.vertex_ai.vertex_gemma_models.main.VertexAIGemmaModels._ensure_access_token",
            return_value=("fake-access-token", "PROJECT_ID"),
        ):
            response = litellm.completion(
                model="vertex_ai/gemma/gemma-3-12b-it-1222199011122",
                messages=[{"role": "user", "content": "Test"}],
                api_base="https://test.prediction.vertexai.goog/v1/projects/PROJECT_ID/locations/us-central1/endpoints/ENDPOINT_ID:predict",
                vertex_project="PROJECT_ID",
                vertex_location="us-central1",
                client=custom_client,
            )

        custom_client.post.assert_called_once()
        assert response.choices[0].message.content == "hi from sync client"

    @pytest.mark.asyncio
    async def test_acompletion_uses_provided_async_client(self):
        """
        A caller-supplied AsyncHTTPHandler must flow through the public API and
        be used. This also guards the entry-point `client` type accepting async
        clients, not just sync ones.
        """
        from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

        vertex_response = _make_gemma_vertex_response(content="hi from async client")

        custom_client = Mock(spec=AsyncHTTPHandler)
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = vertex_response
        custom_client.post = AsyncMock(return_value=mock_response)

        with patch(
            "litellm.llms.vertex_ai.vertex_gemma_models.main.VertexAIGemmaModels._ensure_access_token",
            return_value=("fake-access-token", "PROJECT_ID"),
        ):
            response = await litellm.acompletion(
                model="vertex_ai/gemma/gemma-3-12b-it-1222199011122",
                messages=[{"role": "user", "content": "Test"}],
                api_base="https://test.prediction.vertexai.goog/v1/projects/PROJECT_ID/locations/us-central1/endpoints/ENDPOINT_ID:predict",
                vertex_project="PROJECT_ID",
                vertex_location="us-central1",
                client=custom_client,
            )

        custom_client.post.assert_awaited_once()
        assert response.choices[0].message.content == "hi from async client"

    def test_sync_completion_honors_raw_httpx_client_transport(self):
        """
        Regression for the reviewer's concern: a caller-supplied
        httpx.Client(transport=MockTransport(...)) must be honored on the sync
        path. Before the fix the isinstance(client, HTTPHandler) check failed
        for a raw httpx client, so a brand-new default handler was created and
        the caller's transport was silently dropped, sending the request to the
        real Vertex endpoint.
        """
        import httpx

        from litellm.llms.custom_httpx.http_handler import HTTPHandler
        from litellm.llms.vertex_ai.vertex_gemma_models.transformation import (
            VertexGemmaConfig,
        )
        from litellm.types.utils import ModelResponse

        captured = {}

        def transport_handler(request):
            captured["count"] = captured.get("count", 0) + 1
            captured["url"] = str(request.url)
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                status_code=200,
                json=_make_gemma_vertex_response(content="from mock transport"),
            )

        mock_client = httpx.Client(transport=httpx.MockTransport(transport_handler))

        try:
            with patch.object(
                HTTPHandler,
                "__init__",
                side_effect=AssertionError("raw httpx.Client must not be wrapped"),
            ):
                response = VertexGemmaConfig().completion(
                    model="gemma-3-12b-it",
                    messages=[{"role": "user", "content": "hi"}],
                    api_base="https://should-not-be-reached.invalid/v1:predict",
                    api_key="fake-token",
                    custom_prompt_dict={},
                    model_response=ModelResponse(),
                    print_verbose=lambda *args, **kwargs: None,
                    logging_obj=Mock(),
                    optional_params={},
                    acompletion=False,
                    litellm_params={},
                    client=mock_client,
                )

                assert not mock_client.is_closed

                second_response = VertexGemmaConfig().completion(
                    model="gemma-3-12b-it",
                    messages=[{"role": "user", "content": "hi again"}],
                    api_base="https://should-not-be-reached.invalid/v1:predict",
                    api_key="fake-token",
                    custom_prompt_dict={},
                    model_response=ModelResponse(),
                    print_verbose=lambda *args, **kwargs: None,
                    logging_obj=Mock(),
                    optional_params={},
                    acompletion=False,
                    litellm_params={},
                    client=mock_client,
                )
        finally:
            mock_client.close()

        assert captured["count"] == 2
        assert captured.get("url") == "https://should-not-be-reached.invalid/v1:predict"
        assert captured["body"]["instances"][0]["@requestFormat"] == "chatCompletions"
        assert isinstance(response, ModelResponse)
        assert response.choices[0].message.content == "from mock transport"
        assert response.usage.total_tokens == 114
        assert second_response.choices[0].message.content == "from mock transport"

    def test_sync_completion_ignores_async_client_for_backwards_compatibility(self):
        import asyncio
        import httpx

        from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
        from litellm.llms.vertex_ai.vertex_gemma_models.transformation import (
            VertexGemmaConfig,
        )
        from litellm.types.utils import ModelResponse

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = _make_gemma_vertex_response(content="default sync fallback")
        mock_client = httpx.AsyncClient(transport=httpx.MockTransport(Mock()))

        try:
            with patch.object(
                VertexGemmaConfig,
                "_sync_post",
                return_value=mock_response,
            ) as mock_sync_post:
                response = VertexGemmaConfig().completion(
                    model="gemma-3-12b-it",
                    messages=[{"role": "user", "content": "hi"}],
                    api_base="https://should-not-be-reached.invalid/v1:predict",
                    api_key="fake-token",
                    custom_prompt_dict={},
                    model_response=ModelResponse(),
                    print_verbose=lambda *args, **kwargs: None,
                    logging_obj=Mock(),
                    optional_params={},
                    acompletion=False,
                    litellm_params={},
                    client=mock_client,
                )
        finally:
            asyncio.run(mock_client.aclose())

        mock_sync_post.assert_called_once()
        assert mock_sync_post.call_args.kwargs["client"] is None
        assert response.choices[0].message.content == "default sync fallback"

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = _make_gemma_vertex_response(content="default sync handler fallback")
        with patch.object(
            VertexGemmaConfig,
            "_sync_post",
            return_value=mock_response,
        ) as mock_sync_post:
            response = VertexGemmaConfig().completion(
                model="gemma-3-12b-it",
                messages=[{"role": "user", "content": "hi"}],
                api_base="https://should-not-be-reached.invalid/v1:predict",
                api_key="fake-token",
                custom_prompt_dict={},
                model_response=ModelResponse(),
                print_verbose=lambda *args, **kwargs: None,
                logging_obj=Mock(),
                optional_params={},
                acompletion=False,
                litellm_params={},
                client=Mock(spec=AsyncHTTPHandler),
            )

        mock_sync_post.assert_called_once()
        assert mock_sync_post.call_args.kwargs["client"] is None
        assert response.choices[0].message.content == "default sync handler fallback"

    @pytest.mark.asyncio
    async def test_async_completion_honors_raw_httpx_client_transport(self):
        """Async counterpart: a raw httpx.AsyncClient transport must be honored."""
        import httpx

        from litellm.llms.vertex_ai.vertex_gemma_models.transformation import (
            VertexGemmaConfig,
        )
        from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
        from litellm.types.utils import ModelResponse

        captured = {}

        def transport_handler(request):
            captured["url"] = str(request.url)
            captured["body"] = json.loads(request.content)
            captured["timeout"] = request.extensions.get("timeout")
            return httpx.Response(
                status_code=200,
                json=_make_gemma_vertex_response(content="async from mock transport"),
            )

        mock_client = httpx.AsyncClient(
            timeout=5.0,
            transport=httpx.MockTransport(transport_handler),
        )

        try:
            with patch.object(
                AsyncHTTPHandler,
                "__init__",
                side_effect=AssertionError("raw AsyncClient must not be wrapped"),
            ):
                response = await VertexGemmaConfig().completion(
                    model="gemma-3-12b-it",
                    messages=[{"role": "user", "content": "hi"}],
                    api_base="https://should-not-be-reached.invalid/v1:predict",
                    api_key="fake-token",
                    custom_prompt_dict={},
                    model_response=ModelResponse(),
                    print_verbose=lambda *args, **kwargs: None,
                    logging_obj=Mock(),
                    optional_params={},
                    acompletion=True,
                    litellm_params={},
                    client=mock_client,
                )
        finally:
            await mock_client.aclose()

        assert captured.get("url") == "https://should-not-be-reached.invalid/v1:predict"
        assert captured["body"]["instances"][0]["@requestFormat"] == "chatCompletions"
        assert isinstance(response, ModelResponse)
        assert response.choices[0].message.content == "async from mock transport"
        assert response.usage.total_tokens == 114
        assert captured["timeout"] == {
            "connect": 5.0,
            "read": 5.0,
            "write": 5.0,
            "pool": 5.0,
        }

    @pytest.mark.asyncio
    async def test_async_completion_ignores_sync_client_for_backwards_compatibility(self):
        import httpx

        from litellm.llms.custom_httpx.http_handler import HTTPHandler
        from litellm.llms.vertex_ai.vertex_gemma_models.transformation import (
            VertexGemmaConfig,
        )
        from litellm.types.utils import ModelResponse

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = _make_gemma_vertex_response(content="default async fallback")
        mock_client = httpx.Client(transport=httpx.MockTransport(Mock()))

        try:
            with patch.object(
                VertexGemmaConfig,
                "_async_post",
                new=AsyncMock(return_value=mock_response),
            ) as mock_async_post:
                response = await VertexGemmaConfig().completion(
                    model="gemma-3-12b-it",
                    messages=[{"role": "user", "content": "hi"}],
                    api_base="https://should-not-be-reached.invalid/v1:predict",
                    api_key="fake-token",
                    custom_prompt_dict={},
                    model_response=ModelResponse(),
                    print_verbose=lambda *args, **kwargs: None,
                    logging_obj=Mock(),
                    optional_params={},
                    acompletion=True,
                    litellm_params={},
                    client=mock_client,
                )
        finally:
            mock_client.close()

        mock_async_post.assert_awaited_once()
        assert mock_async_post.call_args.kwargs["client"] is None
        assert response.choices[0].message.content == "default async fallback"

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = _make_gemma_vertex_response(content="default async handler fallback")
        with patch.object(
            VertexGemmaConfig,
            "_async_post",
            new=AsyncMock(return_value=mock_response),
        ) as mock_async_post:
            response = await VertexGemmaConfig().completion(
                model="gemma-3-12b-it",
                messages=[{"role": "user", "content": "hi"}],
                api_base="https://should-not-be-reached.invalid/v1:predict",
                api_key="fake-token",
                custom_prompt_dict={},
                model_response=ModelResponse(),
                print_verbose=lambda *args, **kwargs: None,
                logging_obj=Mock(),
                optional_params={},
                acompletion=True,
                litellm_params={},
                client=Mock(spec=HTTPHandler),
            )

        mock_async_post.assert_awaited_once()
        assert mock_async_post.call_args.kwargs["client"] is None
        assert response.choices[0].message.content == "default async handler fallback"
