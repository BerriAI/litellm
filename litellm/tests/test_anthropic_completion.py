# What is this?
## Unit tests for Anthropic Adapter

import asyncio
import os
import sys
import traceback

from dotenv import load_dotenv

load_dotenv()
import io
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm import AnthropicConfig, Router, adapter_completion
from litellm.adapters.anthropic_adapter import anthropic_adapter
from litellm.types.llms.anthropic import AnthropicResponse


def test_anthropic_completion_messages_translation():
    messages = [{"role": "user", "content": "Hey, how's it going?"}]

    translated_messages = AnthropicConfig().translate_anthropic_messages_to_openai(messages=messages)  # type: ignore

    assert translated_messages == [{"role": "user", "content": "Hey, how's it going?"}]


def test_anthropic_completion_input_translation():
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hey, how's it going?"}],
    }
    translated_input = anthropic_adapter.translate_completion_input_params(kwargs=data)

    assert translated_input is not None

    assert translated_input["model"] == "gpt-3.5-turbo"
    assert translated_input["messages"] == [
        {"role": "user", "content": "Hey, how's it going?"}
    ]


def test_anthropic_completion_e2e():
    litellm.set_verbose = True

    litellm.adapters = [{"id": "anthropic", "adapter": anthropic_adapter}]

    messages = [{"role": "user", "content": "Hey, how's it going?"}]
    response = adapter_completion(
        model="gpt-3.5-turbo",
        messages=messages,
        adapter_id="anthropic",
        mock_response="This is a fake call",
    )

    print("Response: {}".format(response))

    assert response is not None

    assert isinstance(response, AnthropicResponse)


@pytest.mark.asyncio
async def test_anthropic_router_completion_e2e():
    litellm.set_verbose = True

    litellm.adapters = [{"id": "anthropic", "adapter": anthropic_adapter}]

    router = Router(
        model_list=[
            {
                "model_name": "claude-3-5-sonnet-20240620",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "mock_response": "hi this is macintosh.",
                },
            }
        ]
    )
    messages = [{"role": "user", "content": "Hey, how's it going?"}]

    response = await router.aadapter_completion(
        model="claude-3-5-sonnet-20240620",
        messages=messages,
        adapter_id="anthropic",
        mock_response="This is a fake call",
    )

    print("Response: {}".format(response))

    assert response is not None

    assert isinstance(response, AnthropicResponse)

    assert response.model == "gpt-3.5-turbo"
