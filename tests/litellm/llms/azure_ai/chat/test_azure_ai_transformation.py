import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path
import litellm


@pytest.mark.parametrize("is_async", [True, False])
@pytest.mark.asyncio
async def test_azure_ai_request_format(is_async):
    """
    Test that Azure AI requests are formatted correctly with the proper endpoint and parameters
    for both synchronous and asynchronous calls
    """
    litellm._turn_on_debug()

    # Set up the test parameters
    api_key = "00xxx"
    api_base = "https://my-endpoint-europe-berri-992.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-08-01-preview"
    model = "azure_ai/gpt-4o-mini"
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello! How can I assist you today?"},
        {"role": "user", "content": "hi"},
    ]

    if is_async:
        # Mock AsyncHTTPHandler.post method for async test
        with patch(
            "litellm.llms.custom_httpx.llm_http_handler.AsyncHTTPHandler.post"
        ) as mock_post:
            # Set up mock response
            mock_post.return_value = AsyncMock()

            # Call the acompletion function
            try:
                await litellm.acompletion(
                    custom_llm_provider="azure_ai",
                    api_key=api_key,
                    api_base=api_base,
                    model=model,
                    messages=messages,
                )
            except Exception as e:
                # We expect an exception since we're mocking the response
                pass

    else:
        # Mock HTTPHandler.post method for sync test
        with patch(
            "litellm.llms.custom_httpx.llm_http_handler.HTTPHandler.post"
        ) as mock_post:
            # Set up mock response
            mock_post.return_value = MagicMock()

            # Call the completion function
            try:
                litellm.completion(
                    custom_llm_provider="azure_ai",
                    api_key=api_key,
                    api_base=api_base,
                    model=model,
                    messages=messages,
                )
            except Exception as e:
                # We expect an exception since we're mocking the response
                pass

    # Verify the request was made with the correct parameters
    mock_post.assert_called_once()
    call_args = mock_post.call_args
    print("sync request call=", json.dumps(call_args.kwargs, indent=4, default=str))

    # Check URL
    assert call_args.kwargs["url"] == api_base

    # Check headers
    assert call_args.kwargs["headers"]["api-key"] == api_key

    # Check request body
    request_body = json.loads(call_args.kwargs["data"])
    assert (
        request_body["model"] == "gpt-4o-mini"
    )  # Model name should be stripped of provider prefix
    assert request_body["messages"] == messages
