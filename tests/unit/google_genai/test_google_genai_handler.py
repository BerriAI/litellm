#!/usr/bin/env python3
"""
Test to verify the Google GenAI generate_content handler functionality
"""

import asyncio
from collections.abc import Awaitable, Callable
from types import MappingProxyType
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

import litellm
from litellm import Router
from litellm.google_genai.adapters.handler import GenerateContentToCompletionHandler
from litellm.google_genai.adapters.transformation import GoogleGenAIAdapter
from litellm.types.router import GenericLiteLLMParams


def test_stream_response_when_stream_requested_sync():
    """
    Test that when a stream response is returned and streaming was requested,
    the sync handler correctly transforms it to generate_content streaming format.
    """
    # Mock a stream response
    mock_stream = MagicMock()
    mock_stream.__iter__ = MagicMock(return_value=iter([]))

    # Mock the GoogleGenAIAdapter's translate_completion_output_params_streaming method
    with patch.object(
        GoogleGenAIAdapter,
        "translate_completion_output_params_streaming",
        return_value=mock_stream,
    ) as mock_translate:
        with patch("litellm.completion", return_value=mock_stream):
            # Call the handler with stream=True
            result = GenerateContentToCompletionHandler.generate_content_handler(
                model="gemini-pro",
                contents=[{"role": "user", "parts": [{"text": "Hello"}]}],
                litellm_params={},  # Empty dict for params
                stream=True,
            )

            # Verify that translate_completion_output_params_streaming was called
            mock_translate.assert_called_once_with(mock_stream)
            # Verify the result is the transformed stream
            assert result == mock_stream


@pytest.mark.asyncio
async def test_stream_response_when_stream_requested_async():
    """
    Test that when a stream response is returned and streaming was requested,
    the async handler correctly transforms it to generate_content streaming format.
    """
    # Mock a stream response
    mock_stream = MagicMock()
    mock_stream.__aiter__ = AsyncMock(return_value=iter([]))  # Return an empty async iterator

    # Mock the GoogleGenAIAdapter's translate_completion_output_params_streaming method
    with patch.object(
        GoogleGenAIAdapter,
        "translate_completion_output_params_streaming",
        return_value=mock_stream,
    ) as mock_translate:
        with patch("litellm.acompletion", return_value=mock_stream):
            # Call the handler with stream=True
            result = await GenerateContentToCompletionHandler.async_generate_content_handler(
                model="gemini-pro",
                contents=[{"role": "user", "parts": [{"text": "Hello"}]}],
                litellm_params={},  # Empty dict for params
                stream=True,
            )

            # Verify that translate_completion_output_params_streaming was called
            mock_translate.assert_called_once_with(mock_stream)
            # Verify the result is the transformed stream
            assert result == mock_stream


def test_stream_transformation_error_sync():
    """
    Test that when a stream transformation fails, the sync handler raises a ValueError.
    """
    # Mock a stream response
    mock_stream = MagicMock()
    mock_stream.__iter__ = MagicMock(return_value=iter([]))

    # Mock the GoogleGenAIAdapter's translate_completion_output_params_streaming method to return None
    with patch.object(
        GoogleGenAIAdapter,
        "translate_completion_output_params_streaming",
        return_value=None,
    ):
        # Patch litellm.completion directly to prevent real API calls
        with patch("litellm.completion", return_value=mock_stream):
            # Call the handler with stream=True and expect a ValueError
            with pytest.raises(ValueError, match="Failed to transform streaming response"):
                GenerateContentToCompletionHandler.generate_content_handler(
                    model="gemini-pro",
                    contents=[{"role": "user", "parts": [{"text": "Hello"}]}],
                    litellm_params={},  # Empty dict for params
                    stream=True,
                )


@pytest.mark.asyncio
async def test_stream_transformation_error_async():
    """
    Test that when a stream transformation fails, the async handler raises a ValueError.
    """
    # Mock a stream response
    mock_stream = MagicMock()
    mock_stream.__aiter__ = AsyncMock(return_value=mock_stream)

    # Mock the GoogleGenAIAdapter's translate_completion_output_params_streaming method to return None
    with patch.object(
        GoogleGenAIAdapter,
        "translate_completion_output_params_streaming",
        return_value=None,
    ):
        # Mock litellm.acompletion at the module level where it's imported
        # We need to patch it in the handler module, not in litellm itself
        with patch("litellm.google_genai.adapters.handler.litellm") as mock_litellm:
            # Use AsyncMock for async function
            mock_litellm.acompletion = AsyncMock(return_value=mock_stream)
            # Call the handler with stream=True and expect a ValueError
            with pytest.raises(ValueError, match="Failed to transform streaming response"):
                await GenerateContentToCompletionHandler.async_generate_content_handler(
                    model="gemini-pro",
                    contents=[{"role": "user", "parts": [{"text": "Hello"}]}],
                    litellm_params={},  # Empty dict for params
                    stream=True,
                )


def test_citation_metadata_transformation():
    """
    Test that citationMetadata.citationSources is properly transformed to citationMetadata.citations
    to avoid Pydantic validation errors.
    """
    from unittest.mock import MagicMock

    import httpx

    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.llms.gemini.google_genai.transformation import GoogleGenAIConfig

    # Create a mock response with citationMetadata.citationSources (the problematic format)
    mock_response_data = {
        "candidates": [
            {
                "content": {
                    "parts": [{"text": "This is a video analysis response with citation metadata."}],
                    "role": "model",
                },
                "finishReason": "STOP",
                "index": 0,
                "safetyRatings": [],
                "citationMetadata": {
                    "citationSources": [
                        {
                            "startIndex": 5848,
                            "endIndex": 5900,
                            "uri": "https://example.com/video-source",
                            "license": "MIT",
                            "title": "Video Analysis Source",
                            "publicationDate": "2024-01-15",
                        },
                        {
                            "startIndex": 6200,
                            "endIndex": 6250,
                            "uri": "https://another-source.com/reference",
                            "license": "CC-BY",
                            "title": "Another Reference",
                            "publicationDate": "2024-02-01",
                        },
                    ]
                },
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 150,
            "candidatesTokenCount": 200,
            "totalTokenCount": 350,
        },
        "responseId": "test-response-123",
    }

    # Create mock httpx response
    mock_httpx_response = MagicMock(spec=httpx.Response)
    mock_httpx_response.json.return_value = mock_response_data
    mock_httpx_response.status_code = 200
    mock_httpx_response.headers = {}

    # Create logging object
    logging_obj = LiteLLMLoggingObj(
        model="gemini-2.5-flash",
        messages=[],
        stream=False,
        call_type="generate_content",
        start_time=1234567890,
        litellm_call_id="test-call-123",
        function_id="test-function-123",
    )

    # Create GoogleGenAI config
    config = GoogleGenAIConfig()

    # Test the transformation
    try:
        result = config.transform_generate_content_response(
            model="gemini-2.5-flash",
            raw_response=mock_httpx_response,
            logging_obj=logging_obj,
        )

        # Verify the transformation worked
        assert result is not None

        # Check that citationSources was transformed to citations
        if hasattr(result, "candidates") and result.candidates:
            candidate = result.candidates[0]
            if hasattr(candidate, "citationMetadata") and candidate.citationMetadata:
                # The citationMetadata should now have 'citations' instead of 'citationSources'
                citation_metadata = candidate.citationMetadata

                # Check that citations field exists
                assert hasattr(citation_metadata, "citations"), "citations field should exist after transformation"

                # Verify the citations data is preserved
                if hasattr(citation_metadata, "citations") and citation_metadata.citations:
                    assert len(citation_metadata.citations) == 2, "Should have 2 citations"
                    assert citation_metadata.citations[0]["uri"] == "https://example.com/video-source"
                    assert citation_metadata.citations[1]["uri"] == "https://another-source.com/reference"

        print("✅ Citation metadata transformation test passed!")

    except Exception as e:
        pytest.fail(f"Citation metadata transformation failed: {e}")


@pytest.mark.asyncio
async def test_generate_content_adapter_preserves_proxy_server_request():
    """
    Ensure GenerateContentToCompletionHandler forwards proxy_server_request
    to the downstream completion call so proxy spend logging captures the request body.
    """
    from litellm.types.utils import Choices, Message, ModelResponse

    handler = GenerateContentToCompletionHandler()

    dummy_proxy_request: dict[str, object] = {
        "url": "http://localhost:4000/v1beta/models/gemini-2.0-flash:generateContent",
        "method": "POST",
        "headers": {"content-type": "application/json"},
        "body": {"contents": [{"role": "user", "parts": [{"text": "Hello, world!"}]}]},
    }

    gemini_data: list[dict[str, object]] = [{"role": "user", "parts": [{"text": "Hello, world!"}]}]

    mock_response = ModelResponse(choices=[Choices(message=Message(content="Hi!", role="assistant"))])

    with patch(
        "litellm.google_genai.adapters.handler.litellm.acompletion",
        new_callable=AsyncMock,
    ) as mock_acompletion:
        mock_acompletion.return_value = mock_response

        await handler.async_generate_content_handler(
            model="gemini-2.0-flash",
            contents=gemini_data,
            litellm_params=GenericLiteLLMParams(),
            proxy_server_request=dummy_proxy_request,
            metadata={"source": "unit_test"},
        )

        assert mock_acompletion.called, "Inner acompletion was not called"
        called_kwargs = mock_acompletion.call_args.kwargs

        assert "proxy_server_request" in called_kwargs, "proxy_server_request was dropped from completion_kwargs"
        assert called_kwargs["proxy_server_request"] == dummy_proxy_request


_REJECTED_ADAPTER_REQUEST: Final = MappingProxyType(
    {
        "model": "adapter-model",
        "contents": [{"role": "user", "parts": [{"text": "Say hi"}]}],
        "config": {"maxOutputTokens": -5},
    }
)


@pytest.mark.parametrize(
    "send_rejected_request",
    [
        pytest.param(lambda router: router.agenerate_content(**_REJECTED_ADAPTER_REQUEST), id="async"),
        pytest.param(lambda router: router.agenerate_content_stream(**_REJECTED_ADAPTER_REQUEST), id="async_stream"),
        pytest.param(
            lambda router: asyncio.to_thread(router.generate_content, **_REJECTED_ADAPTER_REQUEST), id="sync"
        ),
        pytest.param(
            lambda router: asyncio.to_thread(router.generate_content_stream, **_REJECTED_ADAPTER_REQUEST),
            id="sync_stream",
        ),
        pytest.param(
            lambda _router: litellm.aadapter_generate_content(
                **{**_REJECTED_ADAPTER_REQUEST, "model": "openai/gpt-5.4-mini"},
                litellm_params=GenericLiteLLMParams(api_key="sk-generate-content-adapter-400"),
            ),
            id="sdk_aadapter_generate_content",
        ),
    ],
)
@pytest.mark.asyncio
async def test_generate_content_adapter_returns_provider_400_without_retrying(
    send_rejected_request: Callable[[Router], Awaitable[object]],
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    route: Final = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            400,
            json={
                "error": {
                    "message": "Invalid 'max_completion_tokens': integer below minimum value.",
                    "type": "invalid_request_error",
                    "param": "max_completion_tokens",
                    "code": "integer_below_min_value",
                }
            },
        )
    )
    router: Final = Router(
        model_list=[
            {
                "model_name": "adapter-model",
                "litellm_params": {"model": "openai/gpt-5.4-mini", "api_key": "sk-generate-content-adapter-400"},
            }
        ],
        num_retries=2,
    )

    with pytest.raises(litellm.BadRequestError) as rejected:
        await send_rejected_request(router)

    assert rejected.value.status_code == 400
    assert "Invalid 'max_completion_tokens'" in rejected.value.message
    assert route.call_count == 1


@pytest.mark.parametrize(
    ("status_code", "expected_error"),
    [
        pytest.param(401, litellm.AuthenticationError, id="401"),
        pytest.param(503, litellm.ServiceUnavailableError, id="503"),
    ],
)
@pytest.mark.asyncio
async def test_generate_content_adapter_keeps_provider_error_status(
    status_code: int,
    expected_error: type[Exception],
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(status_code, json={"error": {"message": "provider error", "type": "error"}})
    )

    with pytest.raises(expected_error) as raised:
        await litellm.agenerate_content(
            **{**_REJECTED_ADAPTER_REQUEST, "model": "openai/gpt-5.4-mini"},
            api_key="sk-generate-content-adapter-status",
            max_retries=0,
        )

    assert getattr(raised.value, "status_code", None) == status_code
