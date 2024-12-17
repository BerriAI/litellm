import json
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path


import httpx
import pytest
from respx import MockRouter

import litellm
from litellm import Choices, Message, ModelResponse


@pytest.mark.asyncio
async def test_o1_handle_system_role():
    """
    Tests that:
    - max_tokens is translated to 'max_completion_tokens'
    - role 'system' is translated to 'user'
    """
    from openai import AsyncOpenAI

    litellm.set_verbose = True

    client = AsyncOpenAI(api_key="fake-api-key")

    with patch.object(
        client.chat.completions.with_raw_response, "create"
    ) as mock_client:
        try:
            await litellm.acompletion(
                model="o1-preview",
                max_tokens=10,
                messages=[{"role": "system", "content": "Hello!"}],
                client=client,
            )
        except Exception as e:
            print(f"Error: {e}")

        mock_client.assert_called_once()
        request_body = mock_client.call_args.kwargs

        print("request_body: ", request_body)

        assert request_body["model"] == "o1-preview"
        assert request_body["max_completion_tokens"] == 10
        assert request_body["messages"] == [{"role": "user", "content": "Hello!"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gpt-4", "gpt-4-0314", "gpt-4-32k", "o1-preview"])
async def test_o1_max_completion_tokens(model: str):
    """
    Tests that:
    - max_completion_tokens is passed directly to OpenAI chat completion models
    """
    from openai import AsyncOpenAI

    litellm.set_verbose = True

    client = AsyncOpenAI(api_key="fake-api-key")

    with patch.object(
        client.chat.completions.with_raw_response, "create"
    ) as mock_client:
        try:
            await litellm.acompletion(
                model=model,
                max_completion_tokens=10,
                messages=[{"role": "user", "content": "Hello!"}],
                client=client,
            )
        except Exception as e:
            print(f"Error: {e}")

        mock_client.assert_called_once()
        request_body = mock_client.call_args.kwargs

        print("request_body: ", request_body)

        assert request_body["model"] == model
        assert request_body["max_completion_tokens"] == 10
        assert request_body["messages"] == [{"role": "user", "content": "Hello!"}]


def test_litellm_responses():
    """
    ensures that type of completion_tokens_details is correctly handled / returned
    """
    from litellm import ModelResponse
    from litellm.types.utils import CompletionTokensDetails

    response = ModelResponse(
        usage={
            "completion_tokens": 436,
            "prompt_tokens": 14,
            "total_tokens": 450,
            "completion_tokens_details": {"reasoning_tokens": 0},
        }
    )

    print("response: ", response)

    assert isinstance(response.usage.completion_tokens_details, CompletionTokensDetails)
