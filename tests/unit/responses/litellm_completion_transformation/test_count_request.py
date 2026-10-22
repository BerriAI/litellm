import json
from typing import Final

import pytest
from pydantic import JsonValue

from litellm.litellm_core_utils.tool_fields import DropPolicy
from litellm.responses.litellm_completion_transformation.count_request import (
    ChatCountBridgeError,
    ChatCountRequest,
    chat_count_request,
)
from litellm.responses.litellm_completion_transformation.transformation import ChatCompletionSession

_MARK: Final[dict[str, JsonValue]] = {"type": "ephemeral"}
_MARKED_RESPONSES_REQUEST: Final[dict[str, JsonValue]] = {
    "model": "claude-opus-5-5",
    "instructions": "Be terse.",
    "input": [
        {"type": "message", "role": "user", "content": "Weather in SF?", "cache_control": _MARK},
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "And in LA?", "cache_control": _MARK}],
        },
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": '{"city": "SF"}',
            "cache_control": _MARK,
        },
        {"type": "function_call_output", "call_id": "call_1", "output": "sunny", "cache_control": _MARK},
    ],
    "tools": [
        {
            "type": "function",
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            "cache_control": _MARK,
        }
    ],
}


@pytest.mark.asyncio
@pytest.mark.parametrize("drop_params", (False, True))
async def test_claude_count_request_leaves_out_every_cache_control_mark(drop_params: bool) -> None:
    request: Final = await chat_count_request(
        model="claude-opus-5-5",
        custom_llm_provider="anthropic",
        api_base=None,
        responses_request=_MARKED_RESPONSES_REQUEST,
        policy=DropPolicy(drop_params=drop_params, additional_drop_params=frozenset()),
    )

    assert isinstance(request, ChatCountRequest)
    counted: Final = json.dumps({"messages": request.messages, "tools": request.tools}, default=str)
    assert "cache_control" not in counted
    for kept in ("Weather in SF?", "And in LA?", "call_1", "sunny", "get_weather"):
        assert kept in counted


_WEATHER_TOOL: Final[dict[str, JsonValue]] = {
    "type": "function",
    "name": "get_weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
}
_QUESTION: Final[dict[str, JsonValue]] = {"role": "user", "content": "Weather in SF?"}


async def _claude_count(
    responses_request: dict[str, JsonValue], policy: DropPolicy
) -> ChatCountRequest | ChatCountBridgeError | None:
    return await chat_count_request(
        model="claude-opus-5-5",
        custom_llm_provider="anthropic",
        api_base=None,
        responses_request={"model": "claude-opus-5-5", **responses_request},
        policy=policy,
    )


@pytest.mark.asyncio
async def test_claude_count_request_counts_additional_tools_input_items_as_tools_like_inference() -> None:
    policy: Final = DropPolicy(drop_params=False, additional_drop_params=frozenset())

    hoisted: Final = await _claude_count(
        {"input": [{"type": "additional_tools", "tools": [_WEATHER_TOOL]}, _QUESTION]}, policy
    )

    assert isinstance(hoisted, ChatCountRequest)
    assert hoisted.tools is not None
    assert hoisted == await _claude_count({"input": [_QUESTION], "tools": [_WEATHER_TOOL]}, policy)


@pytest.mark.asyncio
async def test_claude_count_request_drops_a_malformed_tool_flag_under_drop_params_like_inference() -> None:
    policy: Final = DropPolicy(drop_params=True, additional_drop_params=frozenset())

    flagged: Final = await _claude_count(
        {"input": [_QUESTION], "tools": [{**_WEATHER_TOOL, "defer_loading": "yes"}]}, policy
    )

    assert isinstance(flagged, ChatCountRequest)
    assert flagged == await _claude_count({"input": [_QUESTION], "tools": [_WEATHER_TOOL]}, policy)


@pytest.mark.asyncio
async def test_claude_count_request_rejects_a_malformed_tool_flag_without_drop_params() -> None:
    flagged: Final = await _claude_count(
        {"input": [_QUESTION], "tools": [{**_WEATHER_TOOL, "defer_loading": "yes"}]},
        DropPolicy(drop_params=False, additional_drop_params=frozenset()),
    )

    assert isinstance(flagged, ChatCountBridgeError)
    assert "defer_loading" in flagged.message


@pytest.mark.asyncio
async def test_claude_count_request_leaves_out_tools_the_deployment_drops() -> None:
    policy: Final = DropPolicy(drop_params=False, additional_drop_params=frozenset({"tools"}))

    dropped: Final = await _claude_count({"input": [_QUESTION], "tools": [_WEATHER_TOOL]}, policy)

    assert isinstance(dropped, ChatCountRequest)
    assert dropped.tools is None
    assert dropped == await _claude_count({"input": [_QUESTION]}, policy)


_SUMMARY: Final = "Earlier turns covered the Tokyo trip."
_COMPACTED_INPUT: Final[list[JsonValue]] = [
    {"type": "compaction", "id": "cmp_1", "encrypted_content": json.dumps({"type": "compaction", "content": _SUMMARY})},
    {"type": "message", "role": "user", "content": "And Osaka?"},
]
_COMPACTION_REQUESTS: Final = (
    pytest.param({}, id="summary_route"),
    pytest.param({"context_management": [{"type": "compaction", "compact_threshold": 50000}]}, id="native_route"),
)


async def _stored_turns_the_compaction_replaces(previous_response_id: str) -> ChatCompletionSession:
    return ChatCompletionSession(
        messages=[
            {"role": "user", "content": "Weather in SF?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_0", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
                ],
            },
        ],
        litellm_session_id=None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("compaction_request", _COMPACTION_REQUESTS)
async def test_claude_count_request_counts_a_compaction_item_as_its_summary_text(
    compaction_request: dict[str, JsonValue],
) -> None:
    request: Final = await chat_count_request(
        model="claude-opus-5-5",
        custom_llm_provider="anthropic",
        api_base=None,
        responses_request={"model": "claude-opus-5-5", "input": _COMPACTED_INPUT, **compaction_request},
        policy=DropPolicy(drop_params=False, additional_drop_params=frozenset()),
    )

    assert isinstance(request, ChatCountRequest)
    assert [(message.get("role"), message.get("content")) for message in request.messages] == [
        ("user", _SUMMARY),
        ("user", "And Osaka?"),
    ]
    assert "compaction_blocks" not in json.dumps(request.messages, default=str)


@pytest.mark.asyncio
@pytest.mark.parametrize("compaction_request", _COMPACTION_REQUESTS)
async def test_claude_count_request_leaves_out_the_stored_turns_a_compaction_item_replaces(
    compaction_request: dict[str, JsonValue],
) -> None:
    request: Final = await chat_count_request(
        model="claude-opus-5-5",
        custom_llm_provider="anthropic",
        api_base=None,
        responses_request={
            "model": "claude-opus-5-5",
            "previous_response_id": "resp_before_the_compaction",
            "input": _COMPACTED_INPUT,
            **compaction_request,
        },
        policy=DropPolicy(drop_params=False, additional_drop_params=frozenset()),
        session_lookup=_stored_turns_the_compaction_replaces,
    )

    assert isinstance(request, ChatCountRequest)
    counted: Final = json.dumps(request.messages, default=str)
    assert _SUMMARY in counted and "And Osaka?" in counted
    assert "Weather in SF?" not in counted
    assert "call_0" not in counted
