from __future__ import annotations

import json
import os
import random
import string
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.budget import case_timeout_seconds
from integration._support.client import Gateway
from integration._support.database import scratch_database
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.timing import assert_not_delayed_by, assert_within_factor_of_control, timed
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

SECRET: Final = "AKIAIOSFODNN7EXAMPLE"
TEXT: Final = f"AWS_ACCESS_KEY_ID={SECRET}"
MASTER_KEY: Final = "sk-integration-hide-secrets-7426-synthetic-master"


def _provider(request: Request) -> Reply:
    if request.method != "POST":
        return Reply()
    body: Final = json.loads(request.body)
    identity: Final = uuid.uuid4().hex
    if request.target.endswith("/messages"):
        response: Final = {
            "id": f"msg_{identity}",
            "type": "message",
            "role": "assistant",
            "model": body["model"],
            "content": [{"type": "text", "text": "accepted"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 1},
        }
        if body.get("stream"):
            events: Final = (
                ("message_start", {"type": "message_start", "message": {**response, "content": []}}),
                (
                    "content_block_start",
                    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                ),
                (
                    "content_block_delta",
                    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "accepted"}},
                ),
                ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                (
                    "message_delta",
                    {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
                ),
                ("message_stop", {"type": "message_stop"}),
            )
            return Reply(
                content_type="text/event-stream",
                chunks=tuple(f"event: {name}\ndata: {json.dumps(value)}\n\n".encode() for name, value in events),
            )
        return Reply(body=json.dumps(response).encode())
    return Reply(
        body=json.dumps(
            {
                "id": f"chatcmpl_{identity}",
                "object": "chat.completion",
                "created": 1,
                "model": body["model"],
                "choices": [
                    {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "accepted"}}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
            }
        ).encode()
    )


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: Gateway
    wire: Wire
    database_url: str
    owned: OwnedProxy
    environment: Mapping[str, str]


@contextmanager
def tool_rig(
    directory: Path,
    respond: Callable[[Request], Reply] = _provider,
    *,
    configure: Callable[[dict[str, JsonValue]], None] | None = None,
    environment: Mapping[str, str] | None = None,
    workers: int = 2,
) -> Iterator[Rig]:
    assert os.environ.get("LITELLM_LICENSE"), "LITELLM_LICENSE must be set: hide-secrets is an enterprise feature"
    product_root: Final = Path(os.environ.get("INTEGRATION_PROXY_ROOT") or Path(__file__).resolve().parents[3])
    with (
        wire_server(respond) as wire,
        scratch_database("hide_secrets") as database,
    ):
        config: Final = directory / "config.yaml"
        configuration: Final = {
            "model_list": [
                {
                    "model_name": "claude",
                    "litellm_params": {
                        "model": "anthropic/claude-sonnet-5",
                        "api_key": "synthetic-provider-key",
                        "api_base": wire.url,
                        "num_retries": 0,
                    },
                },
                {
                    "model_name": "gpt",
                    "litellm_params": {
                        "model": "openai/gpt-5",
                        "api_key": "synthetic-provider-key",
                        "api_base": wire.url,
                        "num_retries": 0,
                    },
                },
            ],
            "general_settings": {
                "master_key": MASTER_KEY,
                "store_prompts_in_spend_logs": True,
                "disable_spend_logs": False,
                "proxy_batch_write_at": 1,
                "proxy_batch_polling_interval": 1,
            },
            "router_settings": {"num_retries": 0},
            "litellm_settings": {
                "cache": True,
                "cache_params": {
                    "type": "redis",
                    "host": "os.environ/REDIS_HOST",
                    "port": "os.environ/REDIS_PORT",
                },
            },
            "guardrails": [
                {
                    "guardrail_name": "hide-secrets",
                    "litellm_params": {"guardrail": "hide-secrets", "mode": "pre_call", "default_on": True},
                }
            ],
        }
        if configure is not None:
            configure(configuration)
        config.write_text(yaml.safe_dump(configuration))
        child_environment: Final = {
            "DATABASE_URL": database,
            "LITELLM_DISABLE_NO_REDIS_WARNING": "true",
            "ANTHROPIC_API_BASE": wire.url,
            "ANTHROPIC_API_KEY": "synthetic-provider-key",
            "LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS": "True",
            "PYTHONPATH": os.pathsep.join(
                (str(product_root / "enterprise"), str(product_root), os.environ.get("PYTHONPATH", ""))
            ),
            **(environment or {}),
        }
        with httpx.Client() as client:
            seed: Final = Gateway(client, MASTER_KEY, wire.url)
            with owned_proxy_process(
                seed,
                directory,
                child_environment,
                config=config,
                workers=workers,
                client_timeout=case_timeout_seconds(),
            ) as owned:
                yield Rig(owned.gateway, wire, database, owned, child_environment)


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with tool_rig(tmp_path_factory.mktemp("hide-secrets-tools")) as value:
        yield value


def _input() -> dict[str, JsonValue]:
    return {
        "env": TEXT,
        "path": 'C:\\tmp\\"config"',
        "count": 3,
        "enabled": False,
        "optional": None,
        "nested": ["safe", {"value": TEXT}],
    }


def _messages(shape: str) -> dict[str, JsonValue]:
    user: Final[dict[str, JsonValue]] = {"role": "user", "content": "read .env"}
    tool: Final[dict[str, JsonValue]] = {
        "type": "tool_use",
        "id": "tool_1",
        "name": "read_file",
        "input": {"path": ".env"},
    }
    prefix: Final[list[JsonValue]] = [user, {"role": "assistant", "content": [tool]}]
    if shape == "plain":
        return {"model": "claude", "max_tokens": 32, "messages": [{"role": "user", "content": TEXT}]}
    if shape.startswith("result"):
        content: Final[JsonValue] = (
            TEXT
            if shape == "result-string"
            else [
                {"type": "text", "text": TEXT},
                {"type": "text", "text": "keep me"},
            ]
        )
        return {
            "model": "claude",
            "max_tokens": 32,
            "messages": [
                *prefix,
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool_1",
                            "content": content,
                            "cache_control": {"type": "ephemeral"},
                        },
                        {"type": "text", "text": "summarize"},
                    ],
                },
            ],
        }
    if shape.startswith("input"):
        arguments: Final = (
            _input()
            if shape == "input-dict"
            else {"command": "export API_TOKEN=0123456789abcdef0123456789abcdef01234567", "enabled": False}
        )
        return {
            "model": "claude",
            "max_tokens": 32,
            "messages": [
                user,
                {"role": "assistant", "content": [{**tool, "input": arguments}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tool_1", "content": "done"}]},
            ],
        }
    return {
        "model": "claude",
        "max_tokens": 32,
        "messages": [user],
        "system": TEXT
        if shape == "system-string"
        else [{"type": "text", "text": TEXT, "cache_control": {"type": "ephemeral"}}],
    }


@pytest.mark.parametrize(
    "shape",
    ("plain", "result-string", "result-nested", "input-dict", "input-assignment", "system-string", "system-blocks"),
)
@pytest.mark.parametrize("stream", (False, True))
def test_messages_secret_never_reaches_provider(rig: Rig, shape: str, stream: bool) -> None:
    rig.wire.drain()
    request: Final = {**_messages(shape), "stream": stream, "cache": {"no-cache": True}}
    response: Final = rig.proxy.request("POST", "/v1/messages", request)
    assert response.status_code == 200, response.text
    assert "accepted" in response.text
    sent: Final = rig.wire.drain()
    assert len(sent) == 1
    assert SECRET.encode() not in sent[0].body
    assert b"0123456789abcdef0123456789abcdef01234567" not in sent[0].body
    assert b"REDACTED" in sent[0].body
    observed: Final = json.loads(sent[0].body)
    if shape == "input-dict":
        value: Final = observed["messages"][1]["content"][0]["input"]
        assert value.keys() == _input().keys()
        assert value["path"] == _input()["path"]
        assert value["count"] == 3
        assert value["enabled"] is False
        assert value["optional"] is None


def test_chat_tool_arguments_remain_valid_json(rig: Rig) -> None:
    rig.wire.drain()
    response: Final = rig.proxy.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": "gpt",
            "cache": {"no-cache": True},
            "messages": [
                {"role": "user", "content": "read .env"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": json.dumps(_input())},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "done"},
            ],
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == "accepted"
    sent: Final = rig.wire.drain()
    assert len(sent) == 1
    assert SECRET.encode() not in sent[0].body
    arguments: Final = json.loads(json.loads(sent[0].body)["messages"][1]["tool_calls"][0]["function"]["arguments"])
    assert arguments.keys() == _input().keys()
    assert arguments["count"] == 3
    assert arguments["path"] == _input()["path"]
    assert arguments["enabled"] is False
    assert arguments["optional"] is None


def test_deeply_nested_tool_arguments_do_not_stall_the_worker(rig: Rig) -> None:
    def call(depth: int) -> float:
        rig.wire.drain()
        arguments: Final = '{"k":' * depth + json.dumps(TEXT) + "}" * depth
        sent: Final = timed(
            lambda: rig.proxy.request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": "gpt",
                    "cache": {"no-cache": True},
                    "messages": [
                        {"role": "user", "content": "read .env"},
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "read", "arguments": arguments},
                                }
                            ],
                        },
                        {"role": "tool", "tool_call_id": "call_1", "content": "done"},
                    ],
                },
            )
        )
        assert sent.value.status_code == 200, sent.value.text
        upstream: Final = rig.wire.drain()
        assert len(upstream) == 1
        assert SECRET.encode() not in upstream[0].body
        return sent.seconds

    control: Final = tuple(call(1) for _ in range(3))
    assert_not_delayed_by((call(300),), control=control, injected_delay=5.0)


@pytest.fixture(scope="module")
def single_worker(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with tool_rig(tmp_path_factory.mktemp("hide-secrets-one-worker"), workers=1) as value:
        yield value


def _tool_history(turns: int, seed: int, shape: str = "anthropic") -> tuple[str, dict[str, JsonValue]]:
    generator: Final = random.Random(seed)
    alphabet: Final = string.ascii_letters + string.digits
    messages: Final[list[JsonValue]] = [{"role": "user", "content": "read the project"}]
    for turn in range(turns):
        result = "\n".join("".join(generator.choices(alphabet, k=60)) for _ in range(64))
        if shape == "anthropic":
            call: JsonValue = {
                "type": "tool_use",
                "id": f"tool_{turn}",
                "name": "read",
                "input": {"path": f"{turn}.py"},
            }
            messages.append({"role": "assistant", "content": [call]})
            messages.append(
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"tool_{turn}", "content": result}]}
            )
        else:
            function: JsonValue = {"name": "read", "arguments": json.dumps({"path": f"{turn}.py"})}
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": f"call_{turn}", "type": "function", "function": function}],
                }
            )
            messages.append({"role": "tool", "tool_call_id": f"call_{turn}", "content": result})
    if shape == "anthropic":
        return "/v1/messages", {"model": "claude", "max_tokens": 32, "messages": messages, "cache": {"no-cache": True}}
    return "/v1/chat/completions", {"model": "gpt", "messages": messages, "cache": {"no-cache": True}}


@pytest.mark.parametrize("shape", ("anthropic", "chat"))
def test_resent_tool_history_is_scanned_once(single_worker: Rig, shape: str) -> None:
    path, body = _tool_history(150, seed=2 if shape == "anthropic" else 4, shape=shape)
    turns: Final = body["messages"]
    assert isinstance(turns, list)
    first: Final = timed(lambda: single_worker.proxy.request("POST", path, body))
    following: Final = {
        **body,
        "messages": [*turns, {"role": "assistant", "content": "ok"}, {"role": "user", "content": "next"}],
    }
    second: Final = timed(lambda: single_worker.proxy.request("POST", path, following))
    assert first.value.status_code == 200 and second.value.status_code == 200, (first.value.text, second.value.text)
    assert_within_factor_of_control(second.seconds, control=first.seconds, factor=0.5)
