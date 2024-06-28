#### What this tests ####
#    This tests if prompts are being correctly formatted
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../.."))

# from litellm.llms.prompt_templates.factory import prompt_factory
import litellm
from litellm import completion
from litellm.llms.prompt_templates.factory import (
    _bedrock_tools_pt,
    anthropic_messages_pt,
    anthropic_pt,
    claude_2_1_pt,
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
    assert anthropic_messages_pt(messages) == expected_messages

    # Test case: No messages (filtered system messages only) when modify_params is False should raise error
    litellm.modify_params = False
    messages = []
    with pytest.raises(Exception) as err:
        anthropic_messages_pt(messages)
    assert "Invalid first message." in str(err.value)


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
