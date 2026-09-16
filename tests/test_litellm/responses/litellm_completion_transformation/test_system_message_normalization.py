"""
Tests for system message normalization in Responses API -> Chat Completion transformation.
Regression tests for issue #40693: Anthropic /v1/messages -> Responses -> Chat Completions can emit non-leading system messages.
"""

from typing import Any
import pytest
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)


def test_reproduce_issue_40693_non_leading_system_message() -> None:
    """
    Reproduces issue #40693:
    When instructions are provided and the Responses input contains a system message
    (e.g., Claude Code harness injecting skills/agent metadata after user prompt),
    the resulting message sequence must NOT emit non-leading system messages.
    All system messages must be normalized into a single leading system message.
    """
    responses_api_request = {
        "instructions": "You are Claude Code, an AI assistant.",
    }
    input_items = [
        {"role": "user", "content": "Hello, please help with this repo."},
        {
            "role": "system",
            "content": "Available skills: [git, bash, edit]\nAvailable tools: [search]",
        },
    ]

    messages = LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
        input=input_items,
        responses_api_request=responses_api_request,
    )

    # 1. Exactly one leading system message at index 0
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"

    # 2. No non-leading system messages
    assert all((m.get("role") if isinstance(m, dict) else getattr(m, "role", None)) != "system" for m in messages[1:])

    # 3. Content from both instructions and subsequent system message are preserved
    system_content = messages[0]["content"]
    assert "You are Claude Code, an AI assistant." in system_content
    assert "Available skills: [git, bash, edit]" in system_content


def test_single_non_leading_system_message_moved_to_start() -> None:
    """
    When a single system message appears after a user message without instructions,
    it should be moved to the beginning of the message list.
    """
    responses_api_request: dict[str, Any] = {}
    input_items = [
        {"role": "user", "content": "What is the weather?"},
        {"role": "system", "content": "Respond only in metric units."},
    ]

    messages = LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
        input=input_items,
        responses_api_request=responses_api_request,
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "Respond only in metric units."
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "What is the weather?"


def test_already_leading_system_message_unchanged() -> None:
    """
    When a single system message is already at the beginning, it should remain untouched.
    """
    responses_api_request: dict[str, Any] = {}
    input_items = [
        {"role": "system", "content": "System prompt."},
        {"role": "user", "content": "User prompt."},
    ]

    messages = LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
        input=input_items,
        responses_api_request=responses_api_request,
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "System prompt."
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "User prompt."


def test_no_system_message() -> None:
    """
    When no system message is provided, messages should remain unchanged.
    """
    responses_api_request: dict[str, Any] = {}
    input_items = [
        {"role": "user", "content": "Hello!"},
    ]

    messages = LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
        input=input_items,
        responses_api_request=responses_api_request,
    )

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello!"


def test_multiple_system_messages_with_structured_blocks() -> None:
    """
    Handles system messages with list content blocks (e.g. text/input_text blocks).
    """
    responses_api_request = {
        "instructions": "Instruction text.",
    }
    input_items = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "Structured system block 1."},
                {"type": "text", "text": "Structured system block 2."},
            ],
        },
        {"role": "user", "content": "Run tests."},
    ]

    messages = LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
        input=input_items,
        responses_api_request=responses_api_request,
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"

    system_content = messages[0]["content"]
    assert "Instruction text." in system_content
    assert "Structured system block 1." in system_content
    assert "Structured system block 2." in system_content


def test_transform_responses_api_request_to_chat_completion_request_normalizes_system() -> None:
    """
    Verifies end-to-end transformation via transform_responses_api_request_to_chat_completion_request.
    """
    request = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
        model="openai/qwen3.8-flash-next",
        input=[
            {"role": "user", "content": "Hello"},
            {"role": "system", "content": "Follow instructions"},
        ],
        responses_api_request={"instructions": "Be helpful"},
    )

    messages = request["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "Be helpful" in messages[0]["content"]
    assert "Follow instructions" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Hello"


def test_system_message_with_list_of_strings_and_empty_content() -> None:
    """
    Ensures list of strings and empty strings are handled properly in content extraction.
    """
    input_items = [
        {"role": "system", "content": ["Line 1", "", "Line 2"]},
        {"role": "system", "content": ""},
        {"role": "system", "content": None},
        {"role": "user", "content": "Query"},
    ]
    messages = LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
        input=input_items,
        responses_api_request={},
    )
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "Line 1\n\nLine 2"
    assert messages[1]["role"] == "user"


def test_system_message_object_with_attributes() -> None:
    """
    Ensures messages that are objects with .role and .content attributes (not dicts) are handled.
    """

    class ObjMessage:
        def __init__(self, role: str, content: Any) -> None:
            self.role = role
            self.content = content

    input_items = [
        ObjMessage(role="user", content="Hello from user"),
        ObjMessage(role="system", content="System instruction from obj"),
    ]
    normalized = LiteLLMCompletionResponsesConfig._normalize_system_messages(input_items)  # type: ignore[arg-type]
    assert len(normalized) == 2
    assert normalized[0].role == "system"  # type: ignore[union-attr]
    assert normalized[0].content == "System instruction from obj"  # type: ignore[union-attr]
    assert normalized[1].role == "user"  # type: ignore[union-attr]


def test_multiple_system_message_objects_merged() -> None:
    """
    Ensures multiple object-based system messages are extracted and merged into a single system message.
    """

    class ObjMessage:
        def __init__(self, role: str, content: Any) -> None:
            self.role = role
            self.content = content

    input_items = [
        ObjMessage(role="system", content="System part A"),
        ObjMessage(role="user", content="User prompt"),
        ObjMessage(role="system", content="System part B"),
    ]
    normalized = LiteLLMCompletionResponsesConfig._normalize_system_messages(input_items)  # type: ignore[arg-type]
    assert len(normalized) == 2
    assert normalized[0]["role"] == "system"
    assert normalized[0]["content"] == "System part A\n\nSystem part B"
    assert normalized[1].role == "user"  # type: ignore[union-attr]


def test_extract_system_content_edge_cases() -> None:
    """
    Directly tests _extract_system_content edge cases including non-string/non-list content.
    """
    assert LiteLLMCompletionResponsesConfig._extract_system_content({"content": None}) == ()
    assert LiteLLMCompletionResponsesConfig._extract_system_content({"content": 12345}) == ()
    assert LiteLLMCompletionResponsesConfig._extract_system_content({"content": "hello"}) == ("hello",)
    assert LiteLLMCompletionResponsesConfig._extract_system_content({"content": ["a", "b"]}) == ("a", "b")
    assert LiteLLMCompletionResponsesConfig._extract_system_content(
        {"content": [{"text": "t1"}, {"other": "none"}]}
    ) == ("t1",)
