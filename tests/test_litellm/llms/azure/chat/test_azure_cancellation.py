"""
Regression coverage: AzureChatCompletion.acompletion must re-raise
asyncio.CancelledError instead of converting it to AzureOpenAIError(500).

Swallowing the cancel breaks client-disconnect propagation — httpx never sees
the cancellation, so the upstream socket stays open until the LLM finishes
generating tokens that nobody is going to read (and that the user is still
billed for).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from openai import AsyncAzureOpenAI

from litellm.llms.azure.azure import AzureChatCompletion
from litellm.types.utils import ModelResponse


@pytest.mark.asyncio
async def test_acompletion_propagates_cancelled_error(monkeypatch):
    handler = AzureChatCompletion()

    fake_client = MagicMock(spec=AsyncAzureOpenAI)
    fake_client.api_key = "sk-test"

    monkeypatch.setattr(
        AzureChatCompletion,
        "get_azure_openai_client",
        lambda self, **kwargs: fake_client,
    )

    async def _raise_cancelled(*args, **kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr(
        AzureChatCompletion,
        "make_azure_openai_chat_completion_request",
        AsyncMock(side_effect=_raise_cancelled),
    )

    logging_obj = MagicMock()

    with pytest.raises(asyncio.CancelledError):
        await handler.acompletion(
            api_key="sk-test",
            api_version="2024-02-01",
            model="gpt-4o",
            api_base="https://example.openai.azure.com",
            data={"messages": [{"role": "user", "content": "hi"}], "model": "gpt-4o"},
            timeout=30,
            dynamic_params=False,
            model_response=ModelResponse(),
            logging_obj=logging_obj,
            max_retries=0,
        )

    # Sanity: the post_call hook still fires for observability before re-raise.
    logging_obj.post_call.assert_called()
