"""Tests for LiteLLMCompletionTransformationHandler request normalization."""

from unittest.mock import AsyncMock, patch

import pytest

from litellm.responses.litellm_completion_transformation.handler import (
    LiteLLMCompletionTransformationHandler,
)
from litellm.types.utils import ModelResponse


def test_response_api_handler_drops_client_metadata():
    handler = LiteLLMCompletionTransformationHandler()

    with patch("litellm.completion") as mock_completion:
        mock_completion.return_value = ModelResponse(
            id="id", created=0, model="test", object="chat.completion", choices=[]
        )
        handler.response_api_handler(
            model="test",
            input="hi",
            responses_api_request={},
            client_metadata={"source": "codex"},
        )

        assert mock_completion.call_count == 1
        assert "client_metadata" not in mock_completion.call_args.kwargs


@pytest.mark.asyncio
async def test_async_response_api_handler_drops_client_metadata():
    handler = LiteLLMCompletionTransformationHandler()

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = ModelResponse(
            id="id", created=0, model="test", object="chat.completion", choices=[]
        )
        await handler.async_response_api_handler(
            litellm_completion_request={"model": "test"},
            request_input="hi",
            responses_api_request={},
            client_metadata={"source": "codex"},
        )

        assert mock_acompletion.call_count == 1
        assert "client_metadata" not in mock_acompletion.call_args.kwargs
