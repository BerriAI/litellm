import asyncio
import json
from collections.abc import Mapping
from typing import Final

import pytest
from pydantic import JsonValue, TypeAdapter

from litellm.llms.anthropic.messages.adapters.transformation import LiteLLMAnthropicMessagesAdapter
from litellm.llms.anthropic.messages.convertible_blocks import (
    malformed_tool_result_text_error,
    unconvertible_anthropic_blocks,
    unconvertible_blocks_error,
)
from litellm.llms.anthropic.messages.responses_adapters.transformation import (
    LiteLLMAnthropicToResponsesAPIAdapter,
)
from litellm.llms.vertex_ai.count_tokens.context import gemini_count_contents
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


def _tool_turn(inner: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    return (
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_01", "name": "lookup", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_01", "content": [dict(inner)]}]},
    )


def _sent_by_each_counter(inner: Mapping[str, object]) -> tuple[str, str]:
    messages: Final = _tool_turn(inner)
    gemini: Final = asyncio.run(gemini_count_contents(messages, "anthropic", "gemini-2.5-flash", "gemini", {}))
    openai: Final = LiteLLMAnthropicToResponsesAPIAdapter().translate_messages_to_responses_input(
        [dict(message) for message in messages]  # pyright: ignore[reportArgumentType]  # sends block shapes the TypedDicts leave out on purpose
    )
    return json.dumps(gemini, default=str), json.dumps(openai, default=str)


_KEPT_INSIDE_TOOL_RESULT: Final[tuple[tuple[Mapping[str, object], str], ...]] = (
    ({"type": "text", "text": "forty-two"}, "forty-two"),
    ({"type": "image", "source": _PNG}, _PNG["data"]),
    ({"type": "document", "source": _PDF}, _PDF["data"]),
)

_DROPPED_INSIDE_TOOL_RESULT: Final[tuple[tuple[Mapping[str, object], str, str], ...]] = (
    (
        {
            "type": "search_result",
            "source": "https://example.com",
            "title": "t",
            "content": [{"type": "text", "text": "search-snippet"}],
        },
        "search-snippet",
        "search_result inside a tool_result",
    ),
    (
        {"type": "document", "source": {"type": "text", "data": "text-notes"}},
        "text-notes",
        "document with a text source inside a tool_result",
    ),
    ({"type": "tool_reference", "tool_name": "find_docs"}, "find_docs", "tool_reference inside a tool_result"),
)


@pytest.mark.parametrize(("inner", "marker"), _KEPT_INSIDE_TOOL_RESULT)
def test_tool_result_content_both_counters_send_is_accepted(inner: Mapping[str, object], marker: str) -> None:
    gemini, openai = _sent_by_each_counter(inner)

    assert marker in gemini
    assert marker in openai
    assert unconvertible_anthropic_blocks(_tool_turn(inner)) == ()


@pytest.mark.parametrize(("inner", "marker", "label"), _DROPPED_INSIDE_TOOL_RESULT)
def test_tool_result_content_both_counters_drop_is_named(inner: Mapping[str, object], marker: str, label: str) -> None:
    gemini, openai = _sent_by_each_counter(inner)

    assert marker not in gemini
    assert marker not in openai
    assert unconvertible_anthropic_blocks(_tool_turn(inner)) == (label,)


def _replayed_web_search(encrypted_content: str) -> tuple[Mapping[str, object], ...]:
    return (
        {"role": "user", "content": "What is new?"},
        {
            "role": "assistant",
            "content": [
                {"type": "server_tool_use", "id": "srvtoolu_01", "name": "web_search", "input": {"query": "news"}},
                {
                    "type": "web_search_tool_result",
                    "tool_use_id": "srvtoolu_01",
                    "content": [
                        {
                            "type": "web_search_result",
                            "url": "https://docs.litellm.ai/release_notes",
                            "title": "Release notes",
                            "encrypted_content": encrypted_content,
                            "snippet": "LiteLLM ships a weekly release candidate.",
                        }
                    ],
                },
            ],
        },
    )


def test_a_replayed_web_search_inference_sends_as_text_is_accepted() -> None:
    assert unconvertible_blocks_error(_replayed_web_search(""), "anthropic") is None


def test_an_encrypted_web_search_inference_cannot_send_is_named() -> None:
    error: Final = unconvertible_blocks_error(_replayed_web_search("opaque"), "anthropic")
    assert error is not None
    assert error.endswith("server_tool_use, web_search_tool_result")


@pytest.mark.parametrize(
    ("tool_result", "error"),
    (
        (
            {"type": "tool_result", "tool_use_id": "toolu_01", "content": [{"type": "text", "text": None}]},
            "tool_result toolu_01 content.0.text: Input should be a valid string",
        ),
        (
            {"type": "tool_result", "content": [{"type": "text", "text": "ok"}, {"type": "text", "text": 5}]},
            "tool_result content.1.text: Input should be a valid string",
        ),
    ),
)
def test_a_non_string_tool_result_text_is_named_with_or_without_a_tool_use_id(
    tool_result: Mapping[str, object], error: str
) -> None:
    assert malformed_tool_result_text_error([{"role": "user", "content": [tool_result]}]) == error
