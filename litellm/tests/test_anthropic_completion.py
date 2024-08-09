# What is this?
## Unit tests for Anthropic Adapter

import asyncio
import os
import sys
import traceback

from dotenv import load_dotenv

import litellm.types
import litellm.types.utils

load_dotenv()
import io
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from typing import Optional
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


def streaming_format_tests(chunk: dict, idx: int):
    """
    1st chunk -  chunk.get("type") == "message_start"
    2nd chunk - chunk.get("type") == "content_block_start"
    3rd chunk - chunk.get("type") == "content_block_delta"
    """
    if idx == 0:
        assert chunk.get("type") == "message_start"
    elif idx == 1:
        assert chunk.get("type") == "content_block_start"
    elif idx == 2:
        assert chunk.get("type") == "content_block_delta"


@pytest.mark.parametrize("stream", [True])  # False
def test_anthropic_completion_e2e(stream):
    litellm.set_verbose = True

    litellm.adapters = [{"id": "anthropic", "adapter": anthropic_adapter}]

    messages = [{"role": "user", "content": "Hey, how's it going?"}]
    response = adapter_completion(
        model="gpt-3.5-turbo",
        messages=messages,
        adapter_id="anthropic",
        mock_response="This is a fake call",
        stream=stream,
    )

    print("Response: {}".format(response))

    assert response is not None

    if stream is False:
        assert isinstance(response, AnthropicResponse)
    else:
        """
        - ensure finish reason is returned
        - assert content block is started and stopped
        - ensure last chunk is 'message_stop'
        """
        assert isinstance(response, litellm.types.utils.AdapterCompletionStreamWrapper)
        finish_reason: Optional[str] = None
        message_stop_received = False
        content_block_started = False
        content_block_finished = False
        for idx, chunk in enumerate(response):
            print(chunk)
            streaming_format_tests(chunk=chunk, idx=idx)
            if chunk.get("delta", {}).get("stop_reason") is not None:
                finish_reason = chunk.get("delta", {}).get("stop_reason")
            if chunk.get("type") == "message_stop":
                message_stop_received = True
            if chunk.get("type") == "content_block_stop":
                content_block_finished = True
            if chunk.get("type") == "content_block_start":
                content_block_started = True
        assert content_block_started and content_block_finished
        assert finish_reason is not None
        assert message_stop_received is True


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


def test_anthropic_tool_calling_translation():
    kwargs = {
        "model": "claude-3-5-sonnet-20240620",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Would development of a software platform be under ASC 350-40 or ASC 985?",
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "37d6f703-cbcc-497d-95a1-2aa24a114adc",
                        "name": "TaskPlanningTool",
                        "input": {
                            "completed_steps": [],
                            "next_steps": [
                                {
                                    "tool_name": "AccountingResearchTool",
                                    "description": "Research ASC 350-40 to understand its scope and applicability to software development.",
                                },
                                {
                                    "tool_name": "AccountingResearchTool",
                                    "description": "Research ASC 985 to understand its scope and applicability to software development.",
                                },
                                {
                                    "tool_name": "AccountingResearchTool",
                                    "description": "Compare the scopes of ASC 350-40 and ASC 985 to determine which is more applicable to software platform development.",
                                },
                            ],
                            "learnings": [],
                            "potential_issues": [
                                "The distinction between the two standards might not be clear-cut for all types of software development.",
                                "There might be specific circumstances or details about the software platform that could affect which standard applies.",
                            ],
                            "missing_info": [
                                "Specific details about the type of software platform being developed (e.g., for internal use or for sale).",
                                "Whether the entity developing the software is also the end-user or if it's being developed for external customers.",
                            ],
                            "done": False,
                            "required_formatting": None,
                        },
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "eb7023b1-5ee8-43b8-b90f-ac5a23d37c31",
                        "content": {
                            "completed_steps": [],
                            "next_steps": [
                                {
                                    "tool_name": "AccountingResearchTool",
                                    "description": "Research ASC 350-40 to understand its scope and applicability to software development.",
                                },
                                {
                                    "tool_name": "AccountingResearchTool",
                                    "description": "Research ASC 985 to understand its scope and applicability to software development.",
                                },
                                {
                                    "tool_name": "AccountingResearchTool",
                                    "description": "Compare the scopes of ASC 350-40 and ASC 985 to determine which is more applicable to software platform development.",
                                },
                            ],
                            "formatting_step": None,
                        },
                    }
                ],
            },
        ],
    }

    from litellm.adapters.anthropic_adapter import anthropic_adapter

    translated_params = anthropic_adapter.translate_completion_input_params(
        kwargs=kwargs
    )

    print(translated_params["messages"])

    assert len(translated_params["messages"]) > 0
    assert translated_params["messages"][0]["role"] == "user"
