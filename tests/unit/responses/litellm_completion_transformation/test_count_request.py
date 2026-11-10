import json
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Final
from unittest.mock import patch

import pytest
from pydantic import JsonValue

from litellm.litellm_core_utils.tool_fields import DropPolicy
from litellm.responses.litellm_completion_transformation.count_request import (
    ChatCountBridgeError,
    ChatCountRequest,
    chat_count_request,
)
from litellm.responses.litellm_completion_transformation.transformation import ChatCompletionSession
from litellm.responses.utils import ResponsesAPIRequestUtils

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
        {
            "input": [
                {"type": "additional_tools", "role": "developer", "content": "Reply PINEAPPLE.", "tools": [_WEATHER_TOOL]},
                _QUESTION,
            ]
        },
        policy,
    )

    assert isinstance(hoisted, ChatCountRequest)
    assert hoisted.tools is not None
    assert hoisted == await _claude_count({"input": [_QUESTION], "tools": [_WEATHER_TOOL]}, policy)


@pytest.mark.asyncio
async def test_claude_count_request_counts_a_tool_list_repeated_on_every_turn_once_under_drop_params() -> None:
    policy: Final = DropPolicy(drop_params=True, additional_drop_params=frozenset())
    tool_list: Final[dict[str, JsonValue]] = {"type": "additional_tools", "tools": [_WEATHER_TOOL]}

    repeated: Final = await _claude_count(
        {"input": [tool_list, _QUESTION, tool_list, _QUESTION], "tools": [_WEATHER_TOOL]}, policy
    )

    assert repeated == await _claude_count({"input": [_QUESTION, _QUESTION], "tools": [_WEATHER_TOOL]}, policy)


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
async def test_claude_count_request_leaves_a_non_string_role_for_the_count_endpoint_to_judge() -> None:
    counted: Final = await _claude_count(
        {"input": [{"type": "additional_tools", "tools": [_WEATHER_TOOL]}, _QUESTION, {"role": ["user"], "content": "?"}]},
        DropPolicy(drop_params=False, additional_drop_params=frozenset()),
    )

    assert isinstance(counted, ChatCountRequest)
    assert [message.get("role") for message in counted.messages] == ["user", ["user"]]


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


async def _a_stored_question_and_answer(previous_response_id: str) -> ChatCompletionSession:
    return ChatCompletionSession(
        messages=[{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "Four."}],
        litellm_session_id=None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "follow_up_input",
    (
        [{"role": "user", "content": None}],
        [{"role": "user", "content": None}, {"type": "additional_tools", "tools": [_WEATHER_TOOL]}],
    ),
    ids=("empty_user_turn", "empty_user_turn_then_additional_tools"),
)
async def test_claude_count_request_counts_the_empty_user_turn_a_follow_up_sends(
    follow_up_input: list[JsonValue],
) -> None:
    request: Final = await chat_count_request(
        model="claude-opus-5-5",
        custom_llm_provider="anthropic",
        api_base=None,
        responses_request={
            "model": "claude-opus-5-5",
            "previous_response_id": "resp_stored",
            "input": follow_up_input,
        },
        policy=DropPolicy(drop_params=False, additional_drop_params=frozenset()),
        session_lookup=_a_stored_question_and_answer,
    )

    assert isinstance(request, ChatCountRequest)
    assert [(message.get("role"), message.get("content")) for message in request.messages] == [
        ("user", "What is 2+2?"),
        ("assistant", "Four."),
        ("user", ""),
    ]


_APPLICATION_PROFILE_ARN: Final = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc"
_ANSWERED_THEN_EMPTY: Final[list[JsonValue]] = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "Four."},
    {"role": "user", "content": None},
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "responses_request",
    (
        {"model": _APPLICATION_PROFILE_ARN, "input": _ANSWERED_THEN_EMPTY},
        {"model": _APPLICATION_PROFILE_ARN, "previous_response_id": "resp_stored", "input": [_ANSWERED_THEN_EMPTY[-1]]},
    ),
    ids=("in_request", "follow_up"),
)
@pytest.mark.parametrize(
    ("base_model", "fills"), (("anthropic.claude-opus-5-5", True), (None, False)), ids=("base_model", "no_base_model")
)
async def test_converse_count_request_reads_an_arn_deployment_model_from_base_model_like_inference(
    responses_request: dict[str, JsonValue], base_model: str | None, fills: bool
) -> None:
    request: Final = await chat_count_request(
        model=_APPLICATION_PROFILE_ARN,
        custom_llm_provider="bedrock",
        api_base=None,
        responses_request=responses_request,
        policy=DropPolicy(drop_params=False, additional_drop_params=frozenset()),
        session_lookup=_a_stored_question_and_answer,
        base_model=base_model,
    )

    assert isinstance(request, ChatCountRequest)
    assert [(message.get("role"), message.get("content")) for message in request.messages] == [
        ("user", "What is 2+2?"),
        ("assistant", "Four."),
        *((("user", ""),) if fills else ()),
    ]


@pytest.mark.asyncio
async def test_claude_count_request_reads_the_deployment_model_when_the_request_names_an_alias() -> None:
    request: Final = await chat_count_request(
        model="claude-opus-5-5",
        custom_llm_provider="anthropic",
        api_base=None,
        responses_request={"model": "my-claude-alias", "input": _ANSWERED_THEN_EMPTY},
        policy=DropPolicy(drop_params=False, additional_drop_params=frozenset()),
    )

    assert isinstance(request, ChatCountRequest)
    assert [(message.get("role"), message.get("content")) for message in request.messages] == [
        ("user", "What is 2+2?"),
        ("assistant", "Four."),
        ("user", ""),
    ]


class _StoredTurns:
    def __init__(self, rows: tuple[Mapping[str, object], ...]) -> None:
        self.rows: Final = rows

    async def query_raw(self, query: str, request_ids: Sequence[str]) -> list[dict[str, object]]:
        sessions: Final = {row["session_id"] for row in self.rows if row["request_id"] in request_ids}
        return [dict(row) for row in self.rows if row["session_id"] in sessions]


def _stored_converse_turn(request_id: str, input_items: object, reply: str) -> Mapping[str, object]:
    return {
        "request_id": request_id,
        "call_type": "aresponses",
        "session_id": "converse-session",
        "model": "us.anthropic.claude-opus-5-5",
        "custom_llm_provider": "bedrock",
        "proxy_server_request": {"input": input_items, "model": "bedrock/us.anthropic.claude-opus-5-5"},
        "response": {
            "id": request_id,
            "object": "response",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": reply, "annotations": []}],
                }
            ],
        },
    }


def _response_id(chat_id: str) -> str:
    return ResponsesAPIRequestUtils._build_responses_api_response_id(
        custom_llm_provider="bedrock", model_id="deployment-a", response_id=chat_id
    )


@pytest.mark.asyncio
async def test_converse_count_request_replays_a_stored_empty_user_turn_for_a_string_input_like_a_list_input() -> None:
    first_id: Final = _response_id("chatcmpl-turn-one")
    follow_up_id: Final = _response_id("chatcmpl-turn-two")
    stored: Final = _StoredTurns(
        (
            _stored_converse_turn(first_id, "What is 2+2?", "Four."),
            _stored_converse_turn(follow_up_id, [{"role": "user", "content": None}], "Eight."),
        )
    )

    async def counted_turns(follow_up_input: JsonValue) -> list[object]:
        request: Final = await chat_count_request(
            model="us.anthropic.claude-opus-5-5",
            custom_llm_provider="bedrock",
            api_base=None,
            responses_request={
                "model": "us.anthropic.claude-opus-5-5",
                "previous_response_id": follow_up_id,
                "input": follow_up_input,
            },
            policy=DropPolicy(drop_params=False, additional_drop_params=frozenset()),
        )
        assert isinstance(request, ChatCountRequest)
        return [message.get("role") for message in request.messages]

    with patch(  # test-quality-ok: no DI seam for the stored-turn read; test_handler.py stubs this same boundary
        "litellm.proxy.proxy_server.prisma_client", SimpleNamespace(db=stored)
    ):
        string_input_turns: Final = await counted_turns("Double it.")
        list_input_turns: Final = await counted_turns([{"role": "user", "content": "Double it."}])

    assert string_input_turns == list_input_turns == ["user", "assistant", "user", "assistant", "user"]
