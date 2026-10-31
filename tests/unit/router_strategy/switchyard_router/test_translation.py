from typing import Final

import pytest

from litellm.router_strategy.switchyard_router.translation import chat_request, normalize_request, normalize_response
from litellm.types.utils import ModelResponse


def test_chat_roundtrip_preserves_tools_history_and_provider_options() -> None:
    request: Final[dict[str, object]] = {
        "messages": [
            {"role": "system", "content": "Be precise"},
            {"role": "user", "content": "Read the file", "name": "reader"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call-1", "type": "function", "function": {"name": "read", "arguments": '{"path":"a"}'}}
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "contents"},
        ],
        "tools": [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}],
        "tool_choice": {"type": "function", "function": {"name": "read"}},
        "temperature": 0.2,
        "max_completion_tokens": 50,
        "parallel_tool_calls": False,
        "stop": ["END"],
        "metadata": {"trace": "abc"},
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    actual: Final = chat_request(normalize_request(request), request, model="fast", stream=True, judge=False)
    assert actual == {
        **request,
        "model": "fast",
        "tools": ({"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}},),
    }


def test_text_response_with_null_tool_calls_normalizes() -> None:
    response: Final = ModelResponse(
        model="fast",
        choices=[
            {
                "message": {"role": "assistant", "content": "answer", "tool_calls": None},
                "finish_reason": "stop",
            }
        ],
    )
    assert normalize_response(response, "internal-id") == {
        "id": "internal-id",
        "model": "fast",
        "outputs": (
            {"role": "assistant", "content": ({"type": "text", "text": "answer"},), "stop_reason": "end_turn"},
        ),
    }


def test_null_tools_are_normalized_to_empty_catalog() -> None:
    request: Final = normalize_request({"messages": [{"role": "user", "content": "hello"}], "tools": None})
    assert request["tools"] == ()


def test_judge_rewrite_removes_user_tools_limits_and_stop_sequences() -> None:
    request: Final[dict[str, object]] = {
        "messages": [{"role": "user", "content": "task"}],
        "stop": ["}"],
        "max_completion_tokens": 1,
        "tools": [{"type": "function", "function": {"name": "read", "parameters": {}}}],
    }
    rewritten: Final[dict[str, object]] = {
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Judge task"}]}],
        "instructions": [{"role": "system", "content": [{"type": "text", "text": "Return JSON"}]}],
        "output": {"max_output_tokens": 200, "response_format": {"type": "json_object"}},
    }
    assert chat_request(rewritten, request, model="judge", stream=False, judge=True) == {
        "model": "judge",
        "stream": False,
        "messages": [{"role": "system", "content": "Return JSON"}, {"role": "user", "content": "Judge task"}],
        "max_tokens": 200,
        "response_format": {"type": "json_object"},
    }


@pytest.mark.parametrize(
    "content",
    [
        [{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}],
        [{"type": "input_audio", "input_audio": {"data": "abc"}}],
    ],
)
def test_unsupported_content_is_rejected_before_execution(content: object) -> None:
    with pytest.raises(ValueError, match="text and function-tool"):
        normalize_request({"messages": [{"role": "user", "content": content}]})


@pytest.mark.parametrize("choice", ["auto", "none", "required"])
def test_standard_tool_choice_and_text_blocks_survive_roundtrip(choice: str) -> None:
    request: Final = {
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        "tool_choice": choice,
    }
    assert chat_request(normalize_request(request), request, model="fast", stream=False, judge=False) == {
        **request,
        "model": "fast",
        "stream": False,
    }


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"messages": []}, "at least one message"),
        ({"messages": [{"role": "user", "content": "hello"}], "n": 2}, "n=1"),
        ({"messages": [{"role": "function", "content": "hello"}]}, "unsupported chat role"),
        (
            {"messages": [{"role": "user", "content": "hello"}], "tools": [{"type": "web_search"}]},
            "function tools only",
        ),
        (
            {"messages": [{"role": "user", "content": "hello"}], "tool_choice": {"type": "web_search"}},
            "function tool_choice",
        ),
        (
            {
                "messages": [
                    {
                        "role": "assistant",
                        "tool_calls": [{"id": "t", "type": "function", "function": {"name": "read", "arguments": {}}}],
                    }
                ]
            },
            "JSON arguments",
        ),
    ],
)
def test_unsupported_chat_inputs_fail_explicitly(payload: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_request(payload)


def test_rewritten_tool_calls_results_and_refusals_are_preserved() -> None:
    normalized: Final = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_call", "id": "t", "name": "read", "arguments": {"path": "file"}},
                    {"type": "refusal", "text": "Cannot modify"},
                ],
            },
            {
                "role": "tool",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_call_id": "t",
                        "content": [{"type": "text", "text": "result"}],
                        "is_error": None,
                    }
                ],
            },
        ]
    }
    assert chat_request(normalized, {}, model="fast", stream=False, judge=False) == {
        "model": "fast",
        "stream": False,
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "refusal": "Cannot modify",
                "tool_calls": (
                    {"id": "t", "type": "function", "function": {"name": "read", "arguments": '{"path": "file"}'}},
                ),
            },
            {"role": "tool", "tool_call_id": "t", "content": ({"type": "text", "text": "result"},)},
        ],
    }


@pytest.mark.parametrize(
    ("normalized", "message"),
    [
        ({"messages": [{"role": "tool", "content": []}]}, "unsupported tool result"),
        ({"messages": [{"role": "assistant", "content": [{"type": "image", "source": {}}]}]}, "unsupported normalized"),
        ({"messages": [], "tool_choice": {"type": "raw", "data": {}}}, "unsupported tool_choice"),
    ],
)
def test_unsupported_native_rewrites_are_not_silently_dropped(normalized: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        chat_request(normalized, {}, model="fast", stream=False, judge=False)


@pytest.mark.parametrize(
    ("finish", "expected"), [("length", "max_tokens"), ("tool_calls", "tool_use"), ("content_filter", "content_filter")]
)
def test_response_finish_reason_and_refusal_are_normalized(finish: str, expected: str) -> None:
    response: Final = ModelResponse(
        model="fast",
        choices=[
            {
                "message": {"role": "assistant", "content": None, "refusal": "Cannot comply"},
                "finish_reason": finish,
            }
        ],
    )
    assert normalize_response(response, "internal-id")["outputs"] == (
        {
            "role": "assistant",
            "content": ({"type": "refusal", "text": "Cannot comply"},),
            "stop_reason": expected,
        },
    )
