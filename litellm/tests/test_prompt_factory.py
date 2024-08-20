#### What this tests ####
#    This tests if prompts are being correctly formatted
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from typing import Union

# from litellm.llms.prompt_templates.factory import prompt_factory
import litellm
from litellm import completion
from litellm.llms.prompt_templates.factory import (
    _bedrock_tools_pt,
    anthropic_messages_pt,
    anthropic_pt,
    claude_2_1_pt,
    convert_to_anthropic_image_obj,
    convert_url_to_base64,
    llama_2_chat_pt,
    prompt_factory,
)


def test_llama_3_prompt():
    messages = [
        {"role": "system", "content": "You are a good bot"},
        {"role": "user", "content": "Hey, how's it going?"},
    ]
    received_prompt = prompt_factory(
        model="meta-llama/Meta-Llama-3-8B-Instruct", messages=messages
    )
    print(f"received_prompt: {received_prompt}")

    expected_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a good bot<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nHey, how's it going?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""
    assert received_prompt == expected_prompt


def test_codellama_prompt_format():
    messages = [
        {"role": "system", "content": "You are a good bot"},
        {"role": "user", "content": "Hey, how's it going?"},
    ]
    expected_prompt = "<s>[INST] <<SYS>>\nYou are a good bot\n<</SYS>>\n [/INST]\n[INST] Hey, how's it going? [/INST]\n"
    assert llama_2_chat_pt(messages) == expected_prompt


def test_claude_2_1_pt_formatting():
    # Test case: User only, should add Assistant
    messages = [{"role": "user", "content": "Hello"}]
    expected_prompt = "\n\nHuman: Hello\n\nAssistant: "
    assert claude_2_1_pt(messages) == expected_prompt

    # Test case: System, User, and Assistant "pre-fill" sequence,
    #            Should return pre-fill
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": 'Please return "Hello World" as a JSON object.'},
        {"role": "assistant", "content": "{"},
    ]
    expected_prompt = 'You are a helpful assistant.\n\nHuman: Please return "Hello World" as a JSON object.\n\nAssistant: {'
    assert claude_2_1_pt(messages) == expected_prompt

    # Test case: System, Assistant sequence, should insert blank Human message
    #            before Assistant pre-fill
    messages = [
        {"role": "system", "content": "You are a storyteller."},
        {"role": "assistant", "content": "Once upon a time, there "},
    ]
    expected_prompt = (
        "You are a storyteller.\n\nHuman: \n\nAssistant: Once upon a time, there "
    )
    assert claude_2_1_pt(messages) == expected_prompt

    # Test case: System, User sequence
    messages = [
        {"role": "system", "content": "System reboot"},
        {"role": "user", "content": "Is everything okay?"},
    ]
    expected_prompt = "System reboot\n\nHuman: Is everything okay?\n\nAssistant: "
    assert claude_2_1_pt(messages) == expected_prompt


def test_anthropic_pt_formatting():
    # Test case: User only, should add Assistant
    messages = [{"role": "user", "content": "Hello"}]
    expected_prompt = "\n\nHuman: Hello\n\nAssistant: "
    assert anthropic_pt(messages) == expected_prompt

    # Test case: System, User, and Assistant "pre-fill" sequence,
    #            Should return pre-fill
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": 'Please return "Hello World" as a JSON object.'},
        {"role": "assistant", "content": "{"},
    ]
    expected_prompt = '\n\nHuman: <admin>You are a helpful assistant.</admin>\n\nHuman: Please return "Hello World" as a JSON object.\n\nAssistant: {'
    assert anthropic_pt(messages) == expected_prompt

    # Test case: System, Assistant sequence, should NOT insert blank Human message
    #            before Assistant pre-fill, because "System" messages are Human
    #            messages wrapped with <admin></admin>
    messages = [
        {"role": "system", "content": "You are a storyteller."},
        {"role": "assistant", "content": "Once upon a time, there "},
    ]
    expected_prompt = "\n\nHuman: <admin>You are a storyteller.</admin>\n\nAssistant: Once upon a time, there "
    assert anthropic_pt(messages) == expected_prompt

    # Test case: System, User sequence
    messages = [
        {"role": "system", "content": "System reboot"},
        {"role": "user", "content": "Is everything okay?"},
    ]
    expected_prompt = "\n\nHuman: <admin>System reboot</admin>\n\nHuman: Is everything okay?\n\nAssistant: "
    assert anthropic_pt(messages) == expected_prompt


def test_anthropic_messages_pt():
    # Test case: No messages (filtered system messages only)
    litellm.modify_params = True
    messages = []
    expected_messages = [{"role": "user", "content": [{"type": "text", "text": "."}]}]
    assert (
        anthropic_messages_pt(
            messages, model="claude-3-sonnet-20240229", llm_provider="anthropic"
        )
        == expected_messages
    )

    # Test case: No messages (filtered system messages only) when modify_params is False should raise error
    litellm.modify_params = False
    messages = []
    with pytest.raises(Exception) as err:
        anthropic_messages_pt(
            messages, model="claude-3-sonnet-20240229", llm_provider="anthropic"
        )
    assert "Invalid first message" in str(err.value)


def test_anthropic_messages_nested_pt():
    from litellm.types.llms.anthropic import (
        AnthopicMessagesAssistantMessageParam,
        AnthropicMessagesUserMessageParam,
    )

    messages = [
        {"content": [{"text": "here is a task", "type": "text"}], "role": "user"},
        {
            "content": [{"text": "sure happy to help", "type": "text"}],
            "role": "assistant",
        },
        {
            "content": [
                {
                    "text": "Here is a screenshot of the current desktop with the "
                    "mouse coordinates (500, 350). Please select an action "
                    "from the provided schema.",
                    "type": "text",
                }
            ],
            "role": "user",
        },
    ]

    new_messages = anthropic_messages_pt(
        messages, model="claude-3-sonnet-20240229", llm_provider="anthropic"
    )

    assert isinstance(new_messages[1]["content"][0]["text"], str)


# codellama_prompt_format()
def test_bedrock_tool_calling_pt():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    converted_tools = _bedrock_tools_pt(tools=tools)

    print(converted_tools)


def test_convert_url_to_img():
    response_url = convert_url_to_base64(
        url="https://images.pexels.com/photos/1319515/pexels-photo-1319515.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1"
    )

    assert "image/jpeg" in response_url


@pytest.mark.parametrize(
    "url, expected_media_type",
    [
        ("data:image/jpeg;base64,1234", "image/jpeg"),
        ("data:application/pdf;base64,1234", "application/pdf"),
        (r"data:image\/jpeg;base64,1234", "image/jpeg"),
    ],
)
def test_base64_image_input(url, expected_media_type):
    response = convert_to_anthropic_image_obj(openai_image_url=url)

    assert response["media_type"] == expected_media_type


def test_anthropic_messages_tool_call():
    messages = [
        {
            "role": "user",
            "content": "Would development of a software platform be under ASC 350-40 or ASC 985?",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_call_id": "bc8cb4b6-88c4-4138-8993-3a9d9cd51656",
            "tool_calls": [
                {
                    "id": "bc8cb4b6-88c4-4138-8993-3a9d9cd51656",
                    "function": {
                        "arguments": '{"completed_steps": [], "next_steps": [{"tool_name": "AccountingResearchTool", "description": "Research ASC 350-40 to understand its scope and applicability to software development."}, {"tool_name": "AccountingResearchTool", "description": "Research ASC 985 to understand its scope and applicability to software development."}, {"tool_name": "AccountingResearchTool", "description": "Compare the scopes of ASC 350-40 and ASC 985 to determine which is more applicable to software platform development."}], "learnings": [], "potential_issues": ["The distinction between the two standards might not be clear-cut for all types of software development.", "There might be specific circumstances or details about the software platform that could affect which standard applies."], "missing_info": ["Specific details about the type of software platform being developed (e.g., for internal use or for sale).", "Whether the entity developing the software is also the end-user or if it\'s being developed for external customers."], "done": false, "required_formatting": null}',
                        "name": "TaskPlanningTool",
                    },
                    "type": "function",
                }
            ],
        },
        {
            "role": "function",
            "content": '{"completed_steps":[],"next_steps":[{"tool_name":"AccountingResearchTool","description":"Research ASC 350-40 to understand its scope and applicability to software development."},{"tool_name":"AccountingResearchTool","description":"Research ASC 985 to understand its scope and applicability to software development."},{"tool_name":"AccountingResearchTool","description":"Compare the scopes of ASC 350-40 and ASC 985 to determine which is more applicable to software platform development."}],"formatting_step":null}',
            "name": "TaskPlanningTool",
            "tool_call_id": "bc8cb4b6-88c4-4138-8993-3a9d9cd51656",
        },
    ]

    translated_messages = anthropic_messages_pt(
        messages, model="claude-3-sonnet-20240229", llm_provider="anthropic"
    )

    print(translated_messages)

    assert (
        translated_messages[-1]["content"][0]["tool_use_id"]
        == "bc8cb4b6-88c4-4138-8993-3a9d9cd51656"
    )


def test_anthropic_cache_controls_pt():
    "see anthropic docs for this: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching#continuing-a-multi-turn-conversation"
    messages = [
        # marked for caching with the cache_control parameter, so that this checkpoint can read from the previous cache.
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What are the key terms and conditions in this agreement?",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {
            "role": "assistant",
            "content": "Certainly! the key terms and conditions are the following: the contract is 1 year long for $10/mo",
        },
        # The final turn is marked with cache-control, for continuing in followups.
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What are the key terms and conditions in this agreement?",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {
            "role": "assistant",
            "content": "Certainly! the key terms and conditions are the following: the contract is 1 year long for $10/mo",
            "cache_control": {"type": "ephemeral"},
        },
    ]

    translated_messages = anthropic_messages_pt(
        messages, model="claude-3-5-sonnet-20240620", llm_provider="anthropic"
    )

    for i, msg in enumerate(translated_messages):
        if i == 0:
            assert msg["content"][0]["cache_control"] == {"type": "ephemeral"}
        elif i == 1:
            assert "cache_controls" not in msg["content"][0]
        elif i == 2:
            assert msg["content"][0]["cache_control"] == {"type": "ephemeral"}
        elif i == 3:
            assert msg["content"][0]["cache_control"] == {"type": "ephemeral"}

    print("translated_messages: ", translated_messages)


@pytest.mark.parametrize("provider", ["bedrock", "anthropic"])
def test_bedrock_parallel_tool_calling_pt(provider):
    """
    Make sure parallel tool call blocks are merged correctly - https://github.com/BerriAI/litellm/issues/5277
    """
    from litellm.llms.prompt_templates.factory import _bedrock_converse_messages_pt
    from litellm.types.utils import ChatCompletionMessageToolCall, Function, Message

    messages = [
        {
            "role": "user",
            "content": "What's the weather like in San Francisco, Tokyo, and Paris? - give me 3 responses",
        },
        Message(
            content="Here are the current weather conditions for San Francisco, Tokyo, and Paris:",
            role="assistant",
            tool_calls=[
                ChatCompletionMessageToolCall(
                    index=1,
                    function=Function(
                        arguments='{"city": "New York"}',
                        name="get_current_weather",
                    ),
                    id="tooluse_XcqEBfm8R-2YVaPhDUHsPQ",
                    type="function",
                ),
                ChatCompletionMessageToolCall(
                    index=2,
                    function=Function(
                        arguments='{"city": "London"}',
                        name="get_current_weather",
                    ),
                    id="tooluse_VB9nk7UGRniVzGcaj6xrAQ",
                    type="function",
                ),
            ],
            function_call=None,
        ),
        {
            "tool_call_id": "tooluse_XcqEBfm8R-2YVaPhDUHsPQ",
            "role": "tool",
            "name": "get_current_weather",
            "content": "25 degrees celsius.",
        },
        {
            "tool_call_id": "tooluse_VB9nk7UGRniVzGcaj6xrAQ",
            "role": "tool",
            "name": "get_current_weather",
            "content": "28 degrees celsius.",
        },
    ]

    if provider == "bedrock":
        translated_messages = _bedrock_converse_messages_pt(
            messages=messages,
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            llm_provider="bedrock",
        )
    else:
        translated_messages = anthropic_messages_pt(
            messages=messages,
            model="claude-3-sonnet-20240229-v1:0",
            llm_provider=provider,
        )
    print(translated_messages)

    number_of_messages = len(translated_messages)

    # assert last 2 messages are not the same role
    assert (
        translated_messages[number_of_messages - 1]["role"]
        != translated_messages[number_of_messages - 2]["role"]
    )
