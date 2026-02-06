#### What this tests ####
#    This tests if prompts are being correctly formatted
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from typing import List

# from litellm.litellm_core_utils.prompt_templates.factory import prompt_factory
import litellm
from litellm.litellm_core_utils.prompt_templates.factory import (
    _bedrock_tools_pt,
    anthropic_messages_pt,
    anthropic_pt,
    claude_2_1_pt,
    convert_to_anthropic_image_obj,
    convert_to_anthropic_tool_invoke,
    convert_url_to_base64,
    create_anthropic_image_param,
    llama_2_chat_pt,
    prompt_factory,
)
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    get_completion_messages,
)
from litellm.llms.vertex_ai.gemini.transformation import (
    _gemini_convert_messages_with_history,
)
from litellm.types.llms.openai import AllMessageValues
from unittest.mock import MagicMock, patch


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


def test_anthropic_messages_nested_pt():

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
    response = convert_to_anthropic_image_obj(openai_image_url=url, format=None)

    assert response["media_type"] == expected_media_type


def test_create_anthropic_image_param_with_http_url():
    """Test that HTTP/HTTPS URLs are passed as URL references, not base64."""
    image_param = create_anthropic_image_param(
        "https://example.com/image.jpg", format=None
    )

    assert image_param["type"] == "image"
    assert image_param["source"]["type"] == "url"
    assert image_param["source"]["url"] == "https://example.com/image.jpg"


def test_create_anthropic_image_param_with_https_url():
    """Test that HTTPS URLs are passed as URL references."""
    image_param = create_anthropic_image_param(
        "https://example.com/image.png", format=None
    )

    assert image_param["type"] == "image"
    assert image_param["source"]["type"] == "url"
    assert image_param["source"]["url"] == "https://example.com/image.png"


def test_create_anthropic_image_param_with_dict_input():
    """Test that dict input with URL is handled correctly."""
    image_param = create_anthropic_image_param(
        {"url": "https://example.com/image.jpg", "format": "image/jpeg"}, format=None
    )

    assert image_param["type"] == "image"
    assert image_param["source"]["type"] == "url"
    assert image_param["source"]["url"] == "https://example.com/image.jpg"


def test_create_anthropic_image_param_with_base64_data_uri():
    """Test that data URIs are converted to base64."""
    image_param = create_anthropic_image_param(
        "data:image/jpeg;base64,/9j/4AAQSkZJRg==", format=None
    )

    assert image_param["type"] == "image"
    assert image_param["source"]["type"] == "base64"
    assert image_param["source"]["media_type"] == "image/jpeg"
    assert image_param["source"]["data"] == "/9j/4AAQSkZJRg=="


def test_create_anthropic_image_param_with_format_override():
    """Test that format parameter can override media type."""
    image_param = create_anthropic_image_param(
        "data:image/jpeg;base64,1234", format="image/png"
    )

    assert image_param["type"] == "image"
    assert image_param["source"]["type"] == "base64"
    assert image_param["source"]["media_type"] == "image/png"


def test_anthropic_messages_pt_with_url_image():
    """Test that anthropic_messages_pt correctly handles HTTP/HTTPS URLs as URL references."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": "https://example.com/image.jpg",
                },
            ],
        }
    ]

    result = anthropic_messages_pt(
        messages=messages, model="claude-3-5-sonnet", llm_provider="anthropic"
    )

    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert isinstance(result[0]["content"], list)
    assert len(result[0]["content"]) == 2

    # Check text content
    assert result[0]["content"][0]["type"] == "text"

    # Check image content - should be URL reference, not base64
    assert result[0]["content"][1]["type"] == "image"
    assert result[0]["content"][1]["source"]["type"] == "url"
    assert result[0]["content"][1]["source"]["url"] == "https://example.com/image.jpg"


def test_anthropic_messages_pt_with_base64_image():
    """Test that anthropic_messages_pt correctly handles data URIs as base64."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": "data:image/jpeg;base64,/9j/4AAQSkZJRg==",
                },
            ],
        }
    ]

    result = anthropic_messages_pt(
        messages=messages, model="claude-3-5-sonnet", llm_provider="anthropic"
    )

    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert isinstance(result[0]["content"], list)
    assert len(result[0]["content"]) == 2

    # Check image content - should be base64, not URL
    assert result[0]["content"][1]["type"] == "image"
    assert result[0]["content"][1]["source"]["type"] == "base64"
    assert result[0]["content"][1]["source"]["media_type"] == "image/jpeg"


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


def test_anthropic_cache_controls_tool_calls_pt():
    """
    Tests that cache_control is properly set in tool_calls when converting messages
    for the Anthropic API.
    """
    messages = [
        {
            "role": "user",
            "content": "Can you help me get the weather?",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "weather-tool-id-123",
                    "function": {
                        "arguments": '{"location": "San Francisco"}',
                        "name": "get_weather",
                    },
                    "type": "function",
                }
            ],
            "cache_control": {"type": "ephemeral"},
        },
        {
            "role": "function",
            "content": '{"temperature": 72, "unit": "fahrenheit", "description": "sunny"}',
            "name": "get_weather",
            "tool_call_id": "weather-tool-id-123",
            "cache_control": {"type": "ephemeral"},
        },
    ]

    translated_messages = anthropic_messages_pt(
        messages, model="claude-3-sonnet-20240229", llm_provider="anthropic"
    )

    print("Translated tool call messages:", translated_messages)

    assert translated_messages[0]["role"] == "user"

    assert translated_messages[1]["role"] == "assistant"
    for content_item in translated_messages[1]["content"]:
        if content_item["type"] == "tool_use":
            assert "cache_control" not in content_item
            assert content_item["name"] == "get_weather"

    assert translated_messages[2]["role"] == "user"
    for content_item in translated_messages[2]["content"]:
        if content_item["type"] == "tool_result":
            assert content_item["cache_control"] == {"type": "ephemeral"}


@pytest.mark.parametrize("provider", ["bedrock", "anthropic"])
def test_bedrock_parallel_tool_calling_pt(provider):
    """
    Make sure parallel tool call blocks are merged correctly - https://github.com/BerriAI/litellm/issues/5277
    """
    from litellm.litellm_core_utils.prompt_templates.factory import (
        _bedrock_converse_messages_pt,
    )
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


def test_vertex_only_image_user_message():
    base64_image = "/9j/2wCEAAgGBgcGBQ"

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        },
    ]

    response = _gemini_convert_messages_with_history(
        messages=messages, model="gemini-1.5-pro"
    )

    expected_response = [
        {
            "role": "user",
            "parts": [
                {
                    "inline_data": {
                        "data": "/9j/2wCEAAgGBgcGBQ",
                        "mime_type": "image/jpeg",
                    }
                },
                {"text": " "},
            ],
        }
    ]

    assert len(response) == len(expected_response)
    for idx, content in enumerate(response):
        assert (
            content == expected_response[idx]
        ), "Invalid gemini input. Got={}, Expected={}".format(
            content, expected_response[idx]
        )


def test_no_messages_yields_user_text():
    """
    Test that contents are not empty and have text when called without messages
    This is to support blha blah
    """
    messages: List[AllMessageValues] = []

    contents = _gemini_convert_messages_with_history(messages=messages)

    expected_output = [{"role": "user", "parts": [{"text": " "}]}]

    assert contents == expected_output


def test_convert_url():
    convert_url_to_base64("https://picsum.photos/id/237/200/300")


def test_azure_tool_call_invoke_helper():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the weather in Copenhagen?"},
        {"role": "assistant", "function_call": {"name": "get_weather"}},
    ]

    transformed_messages = litellm.AzureOpenAIConfig().transform_request(
        model="gpt-4o",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )

    assert transformed_messages["messages"] == [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the weather in Copenhagen?"},
        {
            "role": "assistant",
            "function_call": {"name": "get_weather", "arguments": ""},
        },
    ]


@pytest.mark.parametrize(
    "messages, expected_messages, user_continue_message, assistant_continue_message",
    [
        (
            [
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hello! How can I assist you today?"},
                {"role": "user", "content": "What is Databricks?"},
                {"role": "user", "content": "What is Azure?"},
                {"role": "assistant", "content": "I don't know anyything, do you?"},
            ],
            [
                {"role": "user", "content": "Hello!"},
                {
                    "role": "assistant",
                    "content": "Hello! How can I assist you today?",
                },
                {"role": "user", "content": "What is Databricks?"},
                {
                    "role": "assistant",
                    "content": "Please continue.",
                },
                {"role": "user", "content": "What is Azure?"},
                {
                    "role": "assistant",
                    "content": "I don't know anyything, do you?",
                },
                {
                    "role": "user",
                    "content": "Please continue.",
                },
            ],
            None,
            None,
        ),
        (
            [
                {"role": "user", "content": "Hello!"},
            ],
            [
                {"role": "user", "content": "Hello!"},
            ],
            None,
            None,
        ),
        (
            [
                {"role": "user", "content": "Hello!"},
                {"role": "user", "content": "What is Databricks?"},
            ],
            [
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Please continue."},
                {"role": "user", "content": "What is Databricks?"},
            ],
            None,
            None,
        ),
        (
            [
                {"role": "user", "content": "Hello!"},
                {"role": "user", "content": "What is Databricks?"},
                {"role": "user", "content": "What is Azure?"},
            ],
            [
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Please continue."},
                {"role": "user", "content": "What is Databricks?"},
                {
                    "role": "assistant",
                    "content": "Please continue.",
                },
                {"role": "user", "content": "What is Azure?"},
            ],
            None,
            None,
        ),
        (
            [
                {"role": "user", "content": "Hello!"},
                {
                    "role": "assistant",
                    "content": "Hello! How can I assist you today?",
                },
                {"role": "user", "content": "What is Databricks?"},
                {"role": "user", "content": "What is Azure?"},
                {"role": "assistant", "content": "I don't know anyything, do you?"},
                {"role": "assistant", "content": "I can't repeat sentences."},
            ],
            [
                {"role": "user", "content": "Hello!"},
                {
                    "role": "assistant",
                    "content": "Hello! How can I assist you today?",
                },
                {"role": "user", "content": "What is Databricks?"},
                {
                    "role": "assistant",
                    "content": "Please continue",
                },
                {"role": "user", "content": "What is Azure?"},
                {
                    "role": "assistant",
                    "content": "I don't know anyything, do you?",
                },
                {
                    "role": "user",
                    "content": "Ok",
                },
                {
                    "role": "assistant",
                    "content": "I can't repeat sentences.",
                },
                {"role": "user", "content": "Ok"},
            ],
            {
                "role": "user",
                "content": "Ok",
            },
            {
                "role": "assistant",
                "content": "Please continue",
            },
        ),
    ],
)
def test_ensure_alternating_roles(
    messages, expected_messages, user_continue_message, assistant_continue_message
):
    messages = get_completion_messages(
        messages=messages,
        assistant_continue_message=assistant_continue_message,
        user_continue_message=user_continue_message,
        ensure_alternating_roles=True,
    )

    print(messages)

    assert messages == expected_messages


def test_alternating_roles_e2e():
    from litellm.llms.custom_httpx.http_handler import HTTPHandler
    import json

    litellm.set_verbose = True
    http_handler = HTTPHandler()

    with patch.object(http_handler, "post", new=MagicMock()) as mock_post:
        try:
            response = litellm.completion(
                **{
                    "model": "databricks/databricks-meta-llama-3-1-70b-instruct",
                    "messages": [
                        {"role": "user", "content": "Hello!"},
                        {
                            "role": "assistant",
                            "content": "Hello! How can I assist you today?",
                        },
                        {"role": "user", "content": "What is Databricks?"},
                        {"role": "user", "content": "What is Azure?"},
                        {
                            "role": "assistant",
                            "content": "I don't know anyything, do you?",
                        },
                        {"role": "assistant", "content": "I can't repeat sentences."},
                    ],
                    "user_continue_message": {
                        "role": "user",
                        "content": "Ok",
                    },
                    "assistant_continue_message": {
                        "role": "assistant",
                        "content": "Please continue",
                    },
                    "ensure_alternating_roles": True,
                },
                client=http_handler,
            )
        except Exception as e:
            print(f"error: {e}")

        assert mock_post.call_args.kwargs["data"] == json.dumps(
            {
                "model": "databricks-meta-llama-3-1-70b-instruct",
                "messages": [
                    {"role": "user", "content": "Hello!"},
                    {
                        "role": "assistant",
                        "content": "Hello! How can I assist you today?",
                    },
                    {"role": "user", "content": "What is Databricks?"},
                    {
                        "role": "assistant",
                        "content": "Please continue",
                    },
                    {"role": "user", "content": "What is Azure?"},
                    {
                        "role": "assistant",
                        "content": "I don't know anyything, do you?",
                    },
                    {
                        "role": "user",
                        "content": "Ok",
                    },
                    {
                        "role": "assistant",
                        "content": "I can't repeat sentences.",
                    },
                    {
                        "role": "user",
                        "content": "Ok",
                    },
                ],
            }
        )


def test_just_system_message():
    from litellm.litellm_core_utils.prompt_templates.factory import (
        _bedrock_converse_messages_pt,
    )

    with pytest.raises(litellm.BadRequestError) as e:
        _bedrock_converse_messages_pt(
            messages=[],
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            llm_provider="bedrock",
        )
        assert "bedrock requires at least one non-system message" in str(e.value)


def test_convert_generic_image_chunk_to_openai_image_obj():
    from litellm.litellm_core_utils.prompt_templates.factory import (
        convert_generic_image_chunk_to_openai_image_obj,
        convert_to_anthropic_image_obj,
    )

    url = "https://i.pinimg.com/736x/b4/b1/be/b4b1becad04d03a9071db2817fc9fe77.jpg"
    image_obj = convert_to_anthropic_image_obj(url, format=None)
    url_str = convert_generic_image_chunk_to_openai_image_obj(image_obj)
    image_obj = convert_to_anthropic_image_obj(url_str, format=None)
    print(image_obj)


def test_hf_chat_template():
    from litellm.litellm_core_utils.prompt_templates.factory import (
        hf_chat_template,
    )

    model = "llama/arn:aws:bedrock:us-east-1:1234:imported-model/45d34re"
    litellm.register_prompt_template(
        model=model,
        tokenizer_config={
            "add_bos_token": True,
            "add_eos_token": False,
            "bos_token": {
                "__type": "AddedToken",
                "content": "",
                "lstrip": False,
                "normalized": True,
                "rstrip": False,
                "single_word": False,
            },
            "clean_up_tokenization_spaces": False,
            "eos_token": {
                "__type": "AddedToken",
                "content": "",
                "lstrip": False,
                "normalized": True,
                "rstrip": False,
                "single_word": False,
            },
            "legacy": True,
            "model_max_length": 16384,
            "pad_token": {
                "__type": "AddedToken",
                "content": "",
                "lstrip": False,
                "normalized": True,
                "rstrip": False,
                "single_word": False,
            },
            "sp_model_kwargs": {},
            "unk_token": None,
            "tokenizer_class": "LlamaTokenizerFast",
            "chat_template": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{' ' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{' ' + tool['type'] + ' ' + tool['function']['name'] + '\n' + '```json' + '\n' + tool['function']['arguments'] + '\n' + '```' + ' '}}{%- set ns.is_first = true -%}{%- else %}{{' ' + tool['type'] + ' ' + tool['function']['name'] + '\n' + '```json' + '\n' + tool['function']['arguments'] + '\n' + '```' + ' '}}{{' '}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{' ' + message['content'] + ' '}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}{{' ' + content + ' '}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{' ' + message['content'] + ' '}}{%- set ns.is_output_first = false %}{%- else %}{{' ' + message['content'] + ' '}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{' '}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{' '}}{% endif %}",
        },
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the weather in Copenhagen?"},
    ]
    chat_template = hf_chat_template(model=model, messages=messages)
    print(chat_template)
    assert (
        chat_template.rstrip()
        == "You are a helpful assistant. What is the weather in Copenhagen?"
    )


def test_ollama_pt():
    from litellm.litellm_core_utils.prompt_templates.factory import ollama_pt

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    prompt = ollama_pt(model="ollama/llama3.1", messages=messages)
    print(prompt)


# ============ Server Tool Use Reconstruction Tests ============
# Fixes: https://github.com/BerriAI/litellm/issues/17737


def test_convert_to_anthropic_tool_invoke_regular_tool():
    """Test that regular tool_use is converted correctly."""
    tool_calls = [
        {
            "id": "toolu_01ABC123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "San Francisco"}',
            },
        }
    ]

    result = convert_to_anthropic_tool_invoke(tool_calls)

    assert len(result) == 1
    assert result[0]["type"] == "tool_use"
    assert result[0]["id"] == "toolu_01ABC123"
    assert result[0]["name"] == "get_weather"
    assert result[0]["input"] == {"location": "San Francisco"}


def test_convert_to_anthropic_tool_invoke_server_tool():
    """
    Test that server_tool_use (srvtoolu_) is reconstructed as server_tool_use.

    Fixes: https://github.com/BerriAI/litellm/issues/17737
    """
    tool_calls = [
        {
            "id": "srvtoolu_01ABC123",
            "type": "function",
            "function": {
                "name": "web_search",
                "arguments": '{"query": "elephant weight"}',
            },
        }
    ]

    result = convert_to_anthropic_tool_invoke(tool_calls)

    assert len(result) == 1
    assert result[0]["type"] == "server_tool_use"  # NOT tool_use
    assert result[0]["id"] == "srvtoolu_01ABC123"
    assert result[0]["name"] == "web_search"
    assert result[0]["input"] == {"query": "elephant weight"}


def test_convert_to_anthropic_tool_invoke_with_web_search_results():
    """
    Test that web_search_tool_result is included after server_tool_use.

    Fixes: https://github.com/BerriAI/litellm/issues/17737
    """
    tool_calls = [
        {
            "id": "srvtoolu_01ABC123",
            "type": "function",
            "function": {
                "name": "web_search",
                "arguments": '{"query": "elephant weight"}',
            },
        }
    ]

    web_search_results = [
        {
            "type": "web_search_tool_result",
            "tool_use_id": "srvtoolu_01ABC123",
            "content": [
                {
                    "type": "web_search_result",
                    "url": "https://example.com",
                    "title": "Elephant Facts",
                    "snippet": "Elephants weigh 5000 kg",
                }
            ],
        }
    ]

    result = convert_to_anthropic_tool_invoke(
        tool_calls, web_search_results=web_search_results
    )

    assert len(result) == 2
    # First: server_tool_use
    assert result[0]["type"] == "server_tool_use"
    assert result[0]["id"] == "srvtoolu_01ABC123"
    # Second: web_search_tool_result
    assert result[1]["type"] == "web_search_tool_result"
    assert result[1]["tool_use_id"] == "srvtoolu_01ABC123"


def test_convert_to_anthropic_tool_invoke_mixed_tools():
    """
    Test that mixed server and regular tools are reconstructed correctly.

    Fixes: https://github.com/BerriAI/litellm/issues/17737
    """
    tool_calls = [
        {
            "id": "srvtoolu_01ABC123",
            "type": "function",
            "function": {
                "name": "web_search",
                "arguments": '{"query": "elephant weight"}',
            },
        },
        {
            "id": "toolu_01XYZ789",
            "type": "function",
            "function": {"name": "add_numbers", "arguments": '{"a": 5000, "b": 100}'},
        },
    ]

    web_search_results = [
        {
            "type": "web_search_tool_result",
            "tool_use_id": "srvtoolu_01ABC123",
            "content": [{"url": "https://example.com", "title": "Test"}],
        }
    ]

    result = convert_to_anthropic_tool_invoke(
        tool_calls, web_search_results=web_search_results
    )

    assert len(result) == 3
    # First: server_tool_use
    assert result[0]["type"] == "server_tool_use"
    assert result[0]["id"] == "srvtoolu_01ABC123"
    # Second: web_search_tool_result
    assert result[1]["type"] == "web_search_tool_result"
    # Third: regular tool_use
    assert result[2]["type"] == "tool_use"
    assert result[2]["id"] == "toolu_01XYZ789"


def test_anthropic_messages_pt_with_server_tool_use():
    """
    Test that anthropic_messages_pt correctly reconstructs server_tool_use from provider_specific_fields.

    Fixes: https://github.com/BerriAI/litellm/issues/17737
    """
    messages = [
        {"role": "user", "content": "Search for elephant weight and add 100"},
        {
            "role": "assistant",
            "content": "Let me search for that.",
            "tool_calls": [
                {
                    "id": "srvtoolu_01ABC123",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": '{"query": "elephant weight"}',
                    },
                },
                {
                    "id": "toolu_01XYZ789",
                    "type": "function",
                    "function": {
                        "name": "add_numbers",
                        "arguments": '{"a": 5000, "b": 100}',
                    },
                },
            ],
            "provider_specific_fields": {
                "web_search_results": [
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": "srvtoolu_01ABC123",
                        "content": [
                            {
                                "url": "https://example.com",
                                "title": "Test",
                                "snippet": "5000 kg",
                            }
                        ],
                    }
                ]
            },
        },
        {"role": "tool", "tool_call_id": "toolu_01XYZ789", "content": "5100"},
    ]

    result = anthropic_messages_pt(
        messages, model="claude-sonnet-4-5", llm_provider="anthropic"
    )

    # Find the assistant message
    assistant_msg = next(m for m in result if m["role"] == "assistant")
    content = assistant_msg["content"]

    # Should have: text, server_tool_use, web_search_tool_result, tool_use
    types = [c.get("type") for c in content]
    assert "text" in types
    assert "server_tool_use" in types
    assert "web_search_tool_result" in types
    assert "tool_use" in types

    # Verify server_tool_use
    server_tool = next(c for c in content if c.get("type") == "server_tool_use")
    assert server_tool["id"] == "srvtoolu_01ABC123"

    # Verify web_search_tool_result comes after server_tool_use
    server_idx = types.index("server_tool_use")
    web_result_idx = types.index("web_search_tool_result")
    assert web_result_idx == server_idx + 1

    # Verify regular tool_use
    tool_use = next(c for c in content if c.get("type") == "tool_use")
    assert tool_use["id"] == "toolu_01XYZ789"


# ============ parse_tool_call_arguments Tests ============
# Tests for the shared utility that parses tool call JSON arguments


def test_parse_tool_call_arguments_valid_json():
    """Test that valid JSON is parsed correctly."""
    from litellm.litellm_core_utils.prompt_templates.common_utils import (
        parse_tool_call_arguments,
    )

    result = parse_tool_call_arguments('{"city": "Paris", "units": "celsius"}')
    assert result == {"city": "Paris", "units": "celsius"}


def test_parse_tool_call_arguments_empty_input():
    """Test that None/empty input returns empty dict."""
    from litellm.litellm_core_utils.prompt_templates.common_utils import (
        parse_tool_call_arguments,
    )

    assert parse_tool_call_arguments(None) == {}
    assert parse_tool_call_arguments("") == {}


def test_parse_tool_call_arguments_malformed_json():
    """Test that malformed JSON raises ValueError with context."""
    from litellm.litellm_core_utils.prompt_templates.common_utils import (
        parse_tool_call_arguments,
    )

    with pytest.raises(ValueError) as exc_info:
        parse_tool_call_arguments(
            '{"skill_name": "pptx',
            tool_name="load_skill",
            context="Anthropic tool invoke",
        )

    error_msg = str(exc_info.value)
    assert "load_skill" in error_msg
    assert "Anthropic tool invoke" in error_msg
    assert '{"skill_name": "pptx' in error_msg
    assert "Unterminated string" in error_msg


def test_convert_to_anthropic_tool_invoke_malformed_json():
    """
    Test that convert_to_anthropic_tool_invoke raises ValueError with context
    when tool arguments contain malformed JSON.

    Fixes: https://github.com/BerriAI/litellm/issues/18920
    """
    tool_calls = [
        {
            "id": "toolu_01_invalid",
            "type": "function",
            "function": {
                "name": "bad_tool",
                "arguments": '{"truncated',  # Malformed JSON
            },
        }
    ]

    with pytest.raises(ValueError) as exc_info:
        convert_to_anthropic_tool_invoke(tool_calls)

    error_msg = str(exc_info.value)
    assert "bad_tool" in error_msg
    assert '{"truncated' in error_msg
