from collections.abc import Mapping
from typing import Final

import pytest
from pydantic import JsonValue, TypeAdapter

from litellm.llms.anthropic.pass_through.adapters.transformation import LiteLLMAnthropicMessagesAdapter
from litellm.llms.anthropic.pass_through.convertible_blocks import unconvertible_anthropic_blocks
from litellm.llms.anthropic.pass_through.responses_adapters.transformation import (
    LiteLLMAnthropicToResponsesAPIAdapter,
)
from litellm.types.llms.anthropic import AllAnthropicPassThroughMessageValues

_JSON_OBJECTS: Final = TypeAdapter(list[dict[str, JsonValue]])

_PDF: Final = {"type": "base64", "media_type": "application/pdf", "data": "JVBERi0xLjQK"}
_PNG: Final = {"type": "base64", "media_type": "image/png", "data": "aW1hZ2U="}

_CONVERTED_BLOCKS: Final[tuple[tuple[str, Mapping[str, object]], ...]] = (
    ("user", {"type": "text", "text": "hi"}),
    ("user", {"type": "image", "source": _PNG}),
    ("user", {"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}}),
    ("user", {"type": "document", "source": _PDF}),
    ("user", {"type": "document", "source": {"type": "url", "url": "https://example.com/a.pdf"}}),
    ("user", {"type": "tool_result", "tool_use_id": "toolu_01", "content": "42"}),
    ("assistant", {"type": "text", "text": "hi"}),
    ("assistant", {"type": "tool_use", "id": "toolu_01", "name": "lookup", "input": {}}),
    ("assistant", {"type": "thinking", "thinking": "Let me think.", "signature": "sig"}),
)

_DROPPED_BLOCKS: Final[tuple[tuple[str, Mapping[str, object], str], ...]] = (
    ("user", {"type": "search_result", "source": "https://example.com", "title": "t", "content": []}, "search_result"),
    ("user", {"type": "document", "source": {"type": "text", "data": "notes"}}, "document with a text source"),
    ("user", {"type": "image", "source": {"type": "file", "file_id": "file_01"}}, "image with a file source"),
    ("user", {"type": "image", "source": {"type": "base64", "data": ""}}, "image with an empty base64 source"),
    ("user", {"type": "document"}, "document without a source"),
    ("user", {"text": "untyped"}, "a block without a type"),
    ("user", {"type": "tool_use", "id": "toolu_01", "name": "lookup", "input": {}}, "tool_use"),
    (
        "assistant",
        {"type": "server_tool_use", "id": "srvtoolu_01", "name": "web_search", "input": {}},
        "server_tool_use",
    ),
)


def _adapter_messages(role: str, block: Mapping[str, object]) -> list[AllAnthropicPassThroughMessageValues]:
    return [{"role": role, "content": [dict(block)]}]  # pyright: ignore[reportReturnType]  # sends block shapes the TypedDicts leave out on purpose


def _converted_by_both_adapters(role: str, block: Mapping[str, object]) -> tuple[bool, bool]:
    messages: Final = _adapter_messages(role, block)
    chat: Final = _JSON_OBJECTS.validate_python(
        LiteLLMAnthropicMessagesAdapter().translate_anthropic_messages_to_openai(messages=messages)
    )
    responses: Final = LiteLLMAnthropicToResponsesAPIAdapter().translate_messages_to_responses_input(messages)
    chat_kept: Final = any(
        message.get("content") or message.get("tool_calls") or message.get("thinking_blocks") for message in chat
    )
    return chat_kept, bool(responses)


@pytest.mark.parametrize(("role", "block"), _CONVERTED_BLOCKS)
def test_blocks_both_conversions_keep_are_accepted(role: str, block: Mapping[str, object]) -> None:
    assert _converted_by_both_adapters(role, block) == (True, True)
    assert unconvertible_anthropic_blocks([{"role": role, "content": [block]}]) == ()


@pytest.mark.parametrize(("role", "block", "label"), _DROPPED_BLOCKS)
def test_blocks_both_conversions_drop_are_named(role: str, block: Mapping[str, object], label: str) -> None:
    assert _converted_by_both_adapters(role, block) == (False, False)
    assert unconvertible_anthropic_blocks([{"role": role, "content": [block]}]) == (label,)


def test_labels_are_listed_once_in_order_and_string_content_is_accepted() -> None:
    messages: Final = [
        {"role": "user", "content": "plain text"},
        {"role": "user", "content": [{"type": "search_result"}, {"type": "container_upload"}]},
        {"role": "assistant", "content": [{"type": "search_result"}]},
    ]
    assert unconvertible_anthropic_blocks(messages) == ("container_upload", "search_result")


def test_redacted_thinking_is_accepted_like_other_replayed_thinking() -> None:
    messages: Final = [{"role": "assistant", "content": [{"type": "redacted_thinking", "data": "opaque"}]}]
    assert unconvertible_anthropic_blocks(messages) == ()


def test_messages_that_are_not_json_are_left_to_the_count() -> None:
    messages: Final = [{"role": "user", "content": [{"type": "search_result", "source": object()}]}]
    assert unconvertible_anthropic_blocks(messages) == ()


def _tool_result(*inner: Mapping[str, object]) -> Mapping[str, object]:
    return {"type": "tool_result", "tool_use_id": "toolu_01", "content": list(inner)}


_KEPT_INSIDE_TOOL_RESULT: Final[tuple[Mapping[str, object], ...]] = (
    {"type": "text", "text": "42"},
    {"type": "image", "source": _PNG},
    {"type": "document", "source": _PDF},
    {"type": "tool_reference", "tool_name": "lookup"},
)

_DROPPED_INSIDE_TOOL_RESULT: Final[tuple[tuple[Mapping[str, object], str], ...]] = (
    (
        {"type": "search_result", "source": "https://example.com", "title": "t", "content": [{"type": "text", "text": "x"}]},
        "search_result inside a tool_result",
    ),
    ({"type": "document", "source": {"type": "text", "data": "notes"}}, "document with a text source inside a tool_result"),
)


@pytest.mark.parametrize("inner", _KEPT_INSIDE_TOOL_RESULT)
def test_tool_result_content_the_conversions_keep_is_accepted(inner: Mapping[str, object]) -> None:
    assert unconvertible_anthropic_blocks([{"role": "user", "content": [_tool_result(inner)]}]) == ()


@pytest.mark.parametrize(("inner", "label"), _DROPPED_INSIDE_TOOL_RESULT)
def test_tool_result_content_both_conversions_drop_is_named(inner: Mapping[str, object], label: str) -> None:
    block: Final = _tool_result(inner)
    chat: Final = _JSON_OBJECTS.validate_python(
        LiteLLMAnthropicMessagesAdapter().translate_anthropic_messages_to_openai(
            messages=_adapter_messages("user", block)
        )
    )
    responses: Final = _JSON_OBJECTS.validate_python(
        LiteLLMAnthropicToResponsesAPIAdapter().translate_messages_to_responses_input(_adapter_messages("user", block))
    )
    assert [message.get("content") for message in chat] == [""]
    assert [item.get("output") for item in responses] == [""]
    assert unconvertible_anthropic_blocks([{"role": "user", "content": [block]}]) == (label,)
