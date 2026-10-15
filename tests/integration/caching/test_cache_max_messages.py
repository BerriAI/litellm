from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final, cast

import pytest
import yaml
from integration._support.anthropic_sse import message_json
from integration._support.client import JSON_OBJECT, Gateway, eventually, object_value, string_value
from integration._support.openai_wire import chat_reply, responses_reply
from integration._support.process import owned_proxy
from integration._support.provider import SharedProvider
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter
from redis import Redis
from redis.exceptions import ResponseError

_Turns = Callable[[str], tuple[list[JsonValue], list[JsonValue]]]
_INDEXES: Final = TypeAdapter(tuple[bytes, ...])
_COUNT: Final = TypeAdapter(tuple[int])


def _redis_command(redis: Redis[bytes], *arguments: str | int) -> object:
    execute: Final = cast(
        Callable[..., object], redis.execute_command
    )  # cast-ok: Redis stubs omit command return types
    return execute(*arguments)


def _search_indexes(redis: Redis[bytes]) -> tuple[bytes, ...]:
    return _INDEXES.validate_python(_redis_command(redis, "FT._LIST"))


def _index_size(redis: Redis[bytes], index: str) -> int:
    return _COUNT.validate_python(_redis_command(redis, "FT.SEARCH", index, "*", "LIMIT", 0, 0))[0]


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


def _cached_responses(redis: Redis[bytes]) -> frozenset[bytes]:
    digests: Final = tuple(key for key in redis.scan_iter() if len(key) == 64)
    return frozenset(key for key in digests if b'"response"' in (redis.get(key) or b""))


_SCENARIOS: Final = (
    pytest.param("/v1/messages", "anthropic/claude-sonnet-5-5", _claude_code_turns, id="messages"),
    pytest.param("/v1/chat/completions", "openai/gpt-5.4", _agent_turns, id="chat-completions"),
    pytest.param("/v1/responses", "openai/responses/gpt-5.6-sol", _responses_turns, id="responses"),
)


@pytest.mark.parametrize(("path", "model", "turns"), _SCENARIOS)
def test_cache_serves_a_turn_under_max_messages_and_skips_one_past_it(
    gateway: Gateway, provider: SharedProvider, path: str, model: str, turns: _Turns
) -> None:
    under_cap, past_cap = turns(f"update the config {uuid.uuid4().hex}")
    provider.expect(_reply(path, "first answer"), _reply(path, "second answer"), _reply(path, "third answer"))

    redis_client: Final[Redis[bytes]] = Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]))
    with redis_client as redis:
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


def _tool_reply(path: str, assistant: Mapping[str, JsonValue]) -> Reply:
    base: Final = JSON_OBJECT.validate_json(_reply(path, "unused").body)
    if path == "/v1/messages":
        return Reply(body=json.dumps({**base, "content": assistant["content"], "stop_reason": "tool_use"}).encode())
    if path == "/v1/responses":
        return Reply(body=json.dumps({**base, "output": [{**assistant, "id": "fc_cached"}]}).encode())
    return Reply(
        body=json.dumps(
            {
                **base,
                "choices": [
                    {
                        "index": 0,
                        "message": assistant,
                        "finish_reason": "tool_calls",
                    }
                ],
            }
        ).encode()
    )


def _stable_payload(path: str, payload: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    if path == "/v1/responses":
        assert string_value(payload["id"]).startswith("resp_")
        return {key: value for key, value in payload.items() if key != "id"}
    return payload


@pytest.mark.parametrize(("path", "model", "turns"), _SCENARIOS)
def test_semantic_cache_embeds_tool_results_and_does_not_replay_the_previous_turn(
    gateway: Gateway,
    provider: SharedProvider,
    tmp_path: Path,
    path: str,
    model: str,
    turns: _Turns,
) -> None:
    redis_client: Final[Redis[bytes]] = Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]))
    with redis_client as redis:
        try:
            _search_indexes(redis)
        except ResponseError:
            pytest.skip("Redis Search is required for semantic caching")

        task: Final = f"update the config {uuid.uuid4().hex}"
        first_turn, past_cap = turns(task)
        equivalent_first, equivalent_past_cap = turns(task + " please")
        tool_turn: Final = past_cap[:-2]
        tool_result: Final = "calc.py test_calc.py" if path == "/v1/messages" else "wrote a.yaml"
        index: Final = f"semantic-tool-turn-{uuid.uuid4().hex}"

        def embed(request: Request) -> Reply:
            assert (request.method, request.target) == ("POST", "/v1/embeddings")
            body: Final = JSON_OBJECT.validate_json(request.body)
            assert body["model"] == "text-embedding-3-small"
            prompt: Final = string_value(body["input"])
            vector: Final = ([0.0, 1.0] if tool_result in prompt else [1.0, 0.0]) + [0.0] * 1534
            return Reply(
                body=json.dumps(
                    {
                        "object": "list",
                        "model": "text-embedding-3-small",
                        "data": [{"object": "embedding", "index": 0, "embedding": vector}],
                        "usage": {"prompt_tokens": 1, "total_tokens": 1},
                    }
                ).encode()
            )

        with wire_server(embed) as embeddings:
            config: Final = JSON_OBJECT.validate_python(
                yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
            )
            models: Final = config["model_list"]
            assert isinstance(models, list)
            settings: Final = object_value(config["litellm_settings"])
            config_path: Final = tmp_path / "semantic-cache.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        **config,
                        "model_list": [
                            *models,
                            {
                                "model_name": "semantic-embedding",
                                "litellm_params": {
                                    "model": "openai/text-embedding-3-small",
                                    "api_base": embeddings.url + "/v1",
                                    "api_key": "scripted-embedding-key",
                                },
                            },
                        ],
                        "litellm_settings": {
                            **settings,
                            "cache_params": {
                                "type": "redis-semantic",
                                "redis_url": f"redis://{os.environ['REDIS_HOST']}:{os.environ['REDIS_PORT']}",
                                "similarity_threshold": 0.8,
                                "redis_semantic_cache_embedding_model": "semantic-embedding",
                                "redis_semantic_cache_index_name": index,
                            },
                        },
                    }
                )
            )
            first_reply: Final = _tool_reply(path, object_value(tool_turn[len(first_turn)]))
            provider.expect(
                first_reply,
                _reply(path, "tool result used"),
                _reply(path, "second answer"),
                _reply(path, "third answer"),
            )
            try:
                with owned_proxy(gateway, tmp_path, {}, config=config_path) as proxy:
                    initial: Final = proxy.post(path, _body(path, model, first_turn))
                    eventually(lambda: _index_size(redis, index), lambda count: count == 1)
                    cached: Final = proxy.post(path, _body(path, model, equivalent_first))
                    assert _stable_payload(path, cached) == _stable_payload(path, initial), (
                        "semantic cache lost the provider-returned tool call"
                    )
                    assert len(provider.received()) == 1

                    follow_up: Final = proxy.post(path, _body(path, model, tool_turn))
                    assert _answer(path, follow_up) == "tool result used", "semantic cache replayed the prior tool call"
                    eventually(lambda: _index_size(redis, index), lambda count: count == 2)
                    assert (
                        _answer(path, proxy.post(path, _body(path, model, equivalent_past_cap[:-2])))
                        == "tool result used"
                    )
                    assert len(provider.received()) == 1

                    past_cap_twice: Final = tuple(proxy.post(path, _body(path, model, past_cap)) for _ in range(2))
                    assert tuple(_answer(path, answer) for answer in past_cap_twice) == (
                        "second answer",
                        "third answer",
                    )
                    assert len(provider.received()) == 2
                    assert any(
                        tool_result in string_value(JSON_OBJECT.validate_json(request.body)["input"])
                        for request in embeddings.drain()
                    )
            finally:
                if index.encode() in _search_indexes(redis):
                    _redis_command(redis, "FT.DROPINDEX", index, "DD")
