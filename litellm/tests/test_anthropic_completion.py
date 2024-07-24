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


def test_anthropic_completion_input_translation_with_metadata():
    """
    Tests that cost tracking works as expected with LiteLLM Proxy

    LiteLLM Proxy will insert litellm_metadata for anthropic endpoints to track user_api_key and user_api_key_team_id

    This test ensures that the `litellm_metadata` is not present in the translated input
    It ensures that `litellm.acompletion()` will receieve metadata which is a litellm specific param
    """
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "Hey, how's it going?"}],
        "litellm_metadata": {
            "user_api_key": "88dc28d0f030c55ed4ab77ed8faf098196cb1c05df778539800c9f1243fe6b4b",
            "user_api_key_alias": None,
            "user_api_end_user_max_budget": None,
            "litellm_api_version": "1.40.19",
            "global_max_parallel_requests": None,
            "user_api_key_user_id": "default_user_id",
            "user_api_key_org_id": None,
            "user_api_key_team_id": None,
            "user_api_key_team_alias": None,
            "user_api_key_team_max_budget": None,
            "user_api_key_team_spend": None,
            "user_api_key_spend": 0.0,
            "user_api_key_max_budget": None,
            "user_api_key_metadata": {},
        },
    }
    translated_input = anthropic_adapter.translate_completion_input_params(kwargs=data)

    assert "litellm_metadata" not in translated_input
    assert "metadata" in translated_input
    assert translated_input["metadata"] == data["litellm_metadata"]


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
