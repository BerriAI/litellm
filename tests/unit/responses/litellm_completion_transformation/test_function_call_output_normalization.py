"""
Tests for normalizing Responses API function_call_output into chat tool messages.

This is important for Gemini/Vertex, which expects tool results to be represented
as tool/function response parts; if the tool output is passed as a list of input_* parts,
we normalize it to text/image blocks or a string.
"""

import json
from copy import deepcopy
from typing import Final, cast

import pytest

from litellm.exceptions import BadRequestError
from litellm.litellm_core_utils.prompt_templates.factory import convert_to_anthropic_tool_result
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.types.llms.openai import ChatCompletionToolMessage


def _normalize_media_content(kind: str, parts: list[dict[str, object]]) -> object:
    if kind == "direct":
        return LiteLLMCompletionResponsesConfig._transform_responses_api_content_to_chat_completion_content(parts)
    return LiteLLMCompletionResponsesConfig._transform_responses_api_tool_call_output_to_chat_completion_message(
        {
            "type": kind,
            "call_id": "call_media",
            "output": parts,
        }
    )[0]["content"]


@pytest.mark.parametrize(
    ("kind", "detail", "nested_url"),
    [
        ("function_call_output", "low", True),
        ("function_call_output", "high", True),
        ("function_call_output", "original", True),
        ("function_call_output", "low", False),
        ("custom_tool_call_output", "high", True),
        ("direct", "high", True),
    ],
)
def test_image_detail_mime_and_adjacent_text_survive_normalization(kind: str, detail: str, nested_url: bool) -> None:
    metadata: Final = {"detail": detail, "format": "image/png", "video_metadata": {"fps": 2}}
    image: Final = {
        "type": "input_image",
        "image_url": {"url": "data:application/octet-stream;base64,AAAA", **metadata}
        if nested_url
        else "data:application/octet-stream;base64,AAAA",
        **({} if nested_url else metadata),
    }
    parts: Final = [{"type": "input_text", "text": ""}, {"type": "input_text", "text": "Screenshot follows"}, image]
    original: Final = deepcopy(parts)
    assert _normalize_media_content(kind, parts) == [
        *([{"type": "text", "text": ""}] if kind == "direct" else []),
        {"type": "text", "text": "Screenshot follows"},
        {"type": "image_url", "image_url": {"url": "data:application/octet-stream;base64,AAAA", **metadata}},
    ]
    assert parts == original


@pytest.mark.parametrize("kind", ["direct", "function_call_output", "custom_tool_call_output"])
def test_media_normalization_preserves_formats_references_and_metadata(kind: str) -> None:
    audio: Final = [
        {
            "type": "input_audio",
            "input_audio": {"data": "AAAA", "format": audio_format},
            "cache_control": {"type": "ephemeral"},
        }
        for audio_format in ("wav", "flac", "audio/ogg")
    ]
    files: Final = [
        {"type": "input_file", "file_data": "data:audio/flac;base64,BBBB", field: "audio/flac"}
        for field in ("format", "mime_type", "content_type")
    ]
    parts: Final = [
        {"type": "input_text", "text": "Inspect this"},
        *audio,
        *files,
        {"type": "input_file", "file_url": "https://example.test/a.pdf"},
        {"type": "input_file", "file_id": "file-test", "detail": "low", "cache_control": {"type": "ephemeral"}},
        {"type": "input_file", "file_data": "PDF", "file": {"filename": "report.pdf", "format": "application/pdf"}},
    ]
    original: Final = deepcopy(parts)
    assert _normalize_media_content(kind, parts) == [
        {"type": "text", "text": "Inspect this"},
        *audio,
        *[{"type": "file", "file": {"file_data": "data:audio/flac;base64,BBBB", "format": "audio/flac"}}] * 3,
        {"type": "file", "file": {"file_id": "https://example.test/a.pdf"}},
        {"type": "file", "file": {"file_id": "file-test", "detail": "low"}, "cache_control": {"type": "ephemeral"}},
        {"type": "file", "file": {"file_data": "PDF", "filename": "report.pdf", "format": "application/pdf"}},
    ]
    assert parts == original


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        (None, ""),
        ("", ""),
        ([], ""),
        ([{"type": "input_text", "text": ""}], ""),
        ([{"type": kind, "text": ""} for kind in ("input_text", "output_text", "text")], ""),
        ("[]", "[]"),
        ([{"type": "input_text", "text": " \n"}], " \n"),
        ('{"ok":true}', '{"ok":true}'),
        ([{"type": "input_text", "text": "hello"}, {"type": "input_text", "text": " world"}], "hello world"),
        ({"type": "file", "status": "ready"}, '{"type": "file", "status": "ready"}'),
        (
            {"type": "file", "file": {"filename": "report.pdf"}, "status": "ready"},
            '{"type": "file", "file": {"filename": "report.pdf"}, "status": "ready"}',
        ),
        ({"type": "input_image", "format": 17}, '{"type": "input_image", "format": 17}'),
        ([{"type": "unknown", "value": 17}], '[{"type": "unknown", "value": 17}]'),
        ([{"type": "input_text", "text": 17}], '[{"type": "input_text", "text": 17}]'),
        ([{"type": "input_text"}], '[{"type": "input_text"}]'),
        ([{"type": "input_text", "text": None}], '[{"type": "input_text", "text": null}]'),
        (
            [{"type": "input_text", "text": ""}, {"type": "unknown", "value": 17}],
            '[{"type": "input_text", "text": ""}, {"type": "unknown", "value": 17}]',
        ),
        ([{"type": "input_text", "text": ""}, 17], '[{"type": "input_text", "text": ""}, 17]'),
        (
            [{"type": "input_text", "text": "keep"}, {"type": "input_image", "file_id": "file-only"}],
            "keep",
        ),
        ([{"type": "input_image", "file_id": "file-only"}], '[{"type": "input_image", "file_id": "file-only"}]'),
        ([{"type": "tool_reference"}], '[{"type": "tool_reference"}]'),
        (
            [{"type": "tool_reference", "tool_name": ""}, {"type": "tool_reference", "tool_name": 17}],
            '[{"type": "tool_reference", "tool_name": ""}, {"type": "tool_reference", "tool_name": 17}]',
        ),
        ([{"type": "input_text", "text": "keep"}, {"type": "tool_reference", "tool_name": None}], "keep"),
        (
            {"type": "tool_reference", "tool_name": "get_weather"},
            '{"type": "tool_reference", "tool_name": "get_weather"}',
        ),
        ({"ids": {7}}, "{'ids': {7}}"),
        ([{"type": "unknown", "ids": {7}}], "[{'type': 'unknown', 'ids': {7}}]"),
    ],
)
def test_tool_output_without_usable_media_keeps_its_existing_fallback(output: object, expected: str) -> None:
    messages: Final = (
        LiteLLMCompletionResponsesConfig._transform_responses_api_tool_call_output_to_chat_completion_message(
            {"type": "function_call_output", "call_id": "call_fallback", "output": output}
        )
    )
    assert messages == [{"role": "tool", "tool_call_id": "call_fallback", "content": expected}]


@pytest.mark.parametrize(
    ("kind", "part"),
    [
        ("direct", {"type": "input_audio"}),
        ("direct", {"type": "input_audio", "input_audio": None}),
        ("function_call_output", {"type": "input_audio", "input_audio": {"format": "wav"}}),
        ("custom_tool_call_output", {"type": "input_audio", "input_audio": {"data": "AAAA"}}),
        ("direct", {"type": "input_audio", "input_audio": {"data": 42, "format": "wav"}}),
        ("function_call_output", {"type": "input_audio", "input_audio": {"data": "AAAA", "format": 42}}),
        (
            "custom_tool_call_output",
            {"type": "input_file", "file_data": {"private_payload": "must-not-appear-in-error"}},
        ),
        ("direct", {"type": "input_file", "file_id": "file-test", "filename": 42}),
        ("function_call_output", {"type": "input_file", "file_id": "file-test", "mime_type": ["audio/flac"]}),
        ("direct", {"type": "input_image", "image_url": {42: "must-not-appear-in-error"}}),
        ("custom_tool_call_output", {"type": "file", "file": None}),
        ("function_call_output", {"type": "file", "file": {"file_data": 42}}),
    ],
)
def test_malformed_media_returns_sanitized_client_error_without_mutating_input(
    kind: str,
    part: dict[str, object],
) -> None:
    parts: Final = [{"type": "input_text", "text": "Keep adjacent content"}, part]
    original: Final = deepcopy(parts)
    with pytest.raises(BadRequestError, match="Invalid Responses media content") as error:
        _normalize_media_content(kind, parts)
    assert error.value.status_code == 400
    assert "must-not-appear-in-error" not in str(error.value)
    assert parts == original


def test_computer_screenshot_output_remains_structured_media() -> None:
    messages: Final = (
        LiteLLMCompletionResponsesConfig._transform_responses_api_tool_call_output_to_chat_completion_message(
            {
                "type": "computer_call_output",
                "call_id": "call_screenshot",
                "output": {"type": "computer_screenshot", "image_url": "data:image/png;base64,AAAA"},
            }
        )
    )
    assert messages[0]["content"] == [
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,AAAA", "detail": "auto"},
        }
    ]


@pytest.mark.parametrize(
    ("kind", "output", "expected_content"),
    [
        (
            "function_call_output",
            [
                {"type": "tool_reference", "tool_name": "get_weather"},
                {"type": "tool_reference", "tool_name": "get_forecast"},
            ],
            [
                {"type": "tool_reference", "tool_name": "get_weather"},
                {"type": "tool_reference", "tool_name": "get_forecast"},
            ],
        ),
        (
            "custom_tool_call_output",
            [
                {"type": "input_text", "text": "Loaded "},
                {"type": "tool_reference", "tool_name": "get_weather"},
                {"type": "output_text", "text": ""},
                {"type": "text", "text": " and "},
                {"type": "tool_reference", "tool_name": "get_forecast"},
            ],
            [
                {"type": "text", "text": "Loaded "},
                {"type": "tool_reference", "tool_name": "get_weather"},
                {"type": "text", "text": " and "},
                {"type": "tool_reference", "tool_name": "get_forecast"},
            ],
        ),
        (
            "tool_result",
            [
                {"type": "tool_reference", "tool_name": ""},
                {"type": "tool_reference", "tool_name": "get_weather", "extra": 1},
                {"type": "unknown", "value": 17},
            ],
            [{"type": "tool_reference", "tool_name": "get_weather"}],
        ),
    ],
    ids=["reference-only", "interleaved-with-text", "malformed-and-unknown-beside-a-valid-reference"],
)
def test_tool_references_reach_anthropic_in_order(
    kind: str, output: list[dict[str, object]], expected_content: list[dict[str, str]]
) -> None:
    original: Final = deepcopy(output)
    messages: Final = (
        LiteLLMCompletionResponsesConfig._transform_responses_api_tool_call_output_to_chat_completion_message(
            {"type": kind, "call_id": "call_load_tools", "output": output}, keep_tool_references=True
        )
    )

    assert messages == [{"role": "tool", "tool_call_id": "call_load_tools", "content": expected_content}]
    assert convert_to_anthropic_tool_result(
        cast(ChatCompletionToolMessage, messages[0])  # cast-ok: the whole tool message is asserted above
    ) == {
        "type": "tool_result",
        "tool_use_id": "call_load_tools",
        "content": expected_content,
    }
    assert output == original


_REFERENCE_ONLY: Final = [{"type": "tool_reference", "tool_name": "get_weather"}]
_TEXT_THEN_REFERENCE: Final = [
    {"type": "input_text", "text": "Loaded "},
    {"type": "tool_reference", "tool_name": "get_weather"},
]


@pytest.mark.parametrize(
    ("model", "custom_llm_provider", "output", "expected_content"),
    [
        ("claude-opus-5-5", "anthropic", _REFERENCE_ONLY, _REFERENCE_ONLY),
        ("claude-opus-5-5", "anthropic", _TEXT_THEN_REFERENCE, [{"type": "text", "text": "Loaded "}, *_REFERENCE_ONLY]),
        ("gpt-5.6", "openai", _REFERENCE_ONLY, json.dumps(_REFERENCE_ONLY)),
        ("gemini-3.1-pro-preview", "vertex_ai", _REFERENCE_ONLY, json.dumps(_REFERENCE_ONLY)),
        ("gpt-5.6", "openai", _TEXT_THEN_REFERENCE, "Loaded "),
        ("my-batch-alias", None, _REFERENCE_ONLY, json.dumps(_REFERENCE_ONLY)),
        ("my-batch-alias", None, _TEXT_THEN_REFERENCE, "Loaded "),
    ],
    ids=[
        "anthropic-reference-only",
        "anthropic-text-then-reference",
        "openai-keeps-json",
        "vertex-gemini-keeps-json",
        "openai-keeps-text",
        "unresolved-alias-keeps-json",
        "unresolved-alias-keeps-text",
    ],
)
def test_only_anthropic_routes_receive_tool_references(
    model: str, custom_llm_provider: str | None, output: list[dict[str, str]], expected_content: object
) -> None:
    request: Final = LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
        model=model,
        input=[
            {"type": "function_call", "call_id": "call_route_refs", "name": "load_tools", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_route_refs", "output": output},
        ],
        responses_api_request={},
        custom_llm_provider=custom_llm_provider,
    )

    assert [message["content"] for message in request["messages"] if message["role"] == "tool"] == [expected_content]
