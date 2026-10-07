import json
import os
import uuid
from collections.abc import Callable
from typing import Final

import pytest
from integration._support.anthropic_sse import message_json
from integration._support.client import Gateway, eventually, object_value, string_value
from integration._support.openai_wire import chat_reply, responses_reply
from integration._support.provider import SharedProvider
from integration._support.wire import Reply
from pydantic import JsonValue
from redis import Redis

_Turns = Callable[[str], tuple[list[JsonValue], list[JsonValue]]]


def _claude_code_turns(task: str) -> tuple[list[JsonValue], list[JsonValue]]:
    """Turns 1 and 3 of a Claude Code session on /v1/messages: 1 and 5 messages"""
    first: Final[list[JsonValue]] = [{"role": "user", "content": task}]
    third: Final[list[JsonValue]] = [
        *first,
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "calc.py test_calc.py"}],
        },
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "toolu_2", "name": "Read", "input": {"file_path": "calc.py"}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_2", "content": "def add(a, b): return a - b"}],
        },
    ]
    return first, third


def _agent_turns(task: str) -> tuple[list[JsonValue], list[JsonValue]]:
    """Turns 1 and 3 of an OpenAI tool loop on /v1/chat/completions: 2 and 6 messages"""

    def call(call_id: str, path: str) -> list[JsonValue]:
        function: Final[JsonValue] = {"name": "write_file", "arguments": json.dumps({"path": path})}
        return [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": call_id, "type": "function", "function": function}],
            },
            {"role": "tool", "tool_call_id": call_id, "content": f"wrote {path}"},
        ]

    first: Final[list[JsonValue]] = [
        {"role": "system", "content": "You are a coding agent"},
        {"role": "user", "content": task},
    ]
    return first, [*first, *call("call_1", "a.yaml"), *call("call_2", "b.yaml")]


def _responses_turns(task: str) -> tuple[list[JsonValue], list[JsonValue]]:
    """Turns 1 and 3 of an agent on /v1/responses: 1 and 5 input items"""

    def call(call_id: str, path: str) -> list[JsonValue]:
        return [
            {
                "type": "function_call",
                "call_id": call_id,
                "name": "write_file",
                "arguments": json.dumps({"path": path}),
            },
            {"type": "function_call_output", "call_id": call_id, "output": f"wrote {path}"},
        ]

    first: Final[list[JsonValue]] = [{"role": "user", "content": task}]
    return first, [*first, *call("call_1", "a.yaml"), *call("call_2", "b.yaml")]


def _body(path: str, model: str, conversation: list[JsonValue]) -> dict[str, JsonValue]:
    if path == "/v1/responses":
        return {"model": model, "input": conversation}
    return {"model": model, "max_tokens": 16, "messages": conversation}


def _reply(path: str, text: str) -> Reply:
    identity: Final = f"id_{uuid.uuid4().hex}"
    if path == "/v1/messages":
        return Reply(body=message_json(identity, "claude-sonnet-5-5", text))
    if path == "/v1/responses":
        return responses_reply(identity, "gpt-5.6-sol", text, stream=False)
    return chat_reply(identity, "gpt-5.4", text, stream=False)


def _answer(path: str, payload: dict[str, JsonValue]) -> str:
    if path == "/v1/messages":
        return string_value(_first(payload["content"])["text"])
    if path == "/v1/responses":
        return string_value(_first(_first(payload["output"])["content"])["text"])
    return string_value(object_value(_first(payload["choices"])["message"])["content"])


def _first(value: JsonValue) -> dict[str, JsonValue]:
    assert isinstance(value, list), value
    return object_value(value[0])


def _cached_responses(redis: Redis) -> frozenset[bytes]:
    digests: Final = tuple(key for key in redis.scan_iter() if len(key) == 64)
    return frozenset(key for key in digests if b'"response"' in (redis.get(key) or b""))


@pytest.mark.parametrize(
    ("path", "model", "turns"),
    [
        pytest.param("/v1/messages", "anthropic/claude-sonnet-5-5", _claude_code_turns, id="messages"),
        pytest.param("/v1/chat/completions", "openai/gpt-5.4", _agent_turns, id="chat-completions"),
        pytest.param("/v1/responses", "openai/responses/gpt-5.6-sol", _responses_turns, id="responses"),
    ],
)
def test_cache_serves_a_turn_under_max_messages_and_skips_one_past_it(
    gateway: Gateway, provider: SharedProvider, path: str, model: str, turns: _Turns
) -> None:
    under_cap, past_cap = turns(f"update the config {uuid.uuid4().hex}")
    provider.expect(_reply(path, "first answer"), _reply(path, "second answer"), _reply(path, "third answer"))

    with Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"])) as redis:
        cached_before: Final = _cached_responses(redis)
        gateway.post(path, _body(path, model, under_cap))
        eventually(lambda: _cached_responses(redis) - cached_before, lambda written: len(written) == 1)
    repeated: Final = gateway.post(path, _body(path, model, under_cap))
    past_cap_twice: Final = (
        gateway.post(path, _body(path, model, past_cap)),
        gateway.post(path, _body(path, model, past_cap)),
    )

    assert _answer(path, repeated) == "first answer", "a repeated turn under max_messages was not served from the cache"
    assert tuple(_answer(path, answer) for answer in past_cap_twice) == ("second answer", "third answer"), (
        "a turn past max_messages was served from the cache"
    )
    assert len(provider.received()) == 3
