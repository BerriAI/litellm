#### What this tests ####
#    This tests if prompts are being correctly formatted
import sys
import os

sys.path.insert(0, os.path.abspath("../.."))

# from litellm.llms.prompt_templates.factory import prompt_factory
from litellm import completion
from litellm.llms.prompt_templates.factory import (
    anthropic_pt,
    claude_2_1_pt,
    llama_2_chat_pt,
)


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


# codellama_prompt_format()
