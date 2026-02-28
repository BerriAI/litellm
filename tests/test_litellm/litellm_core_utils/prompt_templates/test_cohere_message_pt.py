import pytest

from litellm.litellm_core_utils.prompt_templates.factory import cohere_message_pt


def test_cohere_message_pt_with_string_content():
    messages = [{"role": "user", "content": "Hi"}]

    prompt, tool_results = cohere_message_pt(messages)

    assert prompt == "Hi"
    assert tool_results == []


def test_cohere_message_pt_with_list_content_text_part():
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Hi",
                }
            ],
        }
    ]

    prompt, tool_results = cohere_message_pt(messages)

    assert prompt == "Hi"
    assert tool_results == []

