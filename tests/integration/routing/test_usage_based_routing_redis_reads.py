from __future__ import annotations

import json
import shlex
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway, eventually, object_value, string_value
from integration._support.process import owned_proxy
from integration._support.redis_process import owned_redis
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter
from redis import Redis
from redis.exceptions import TimeoutError as RedisTimeoutError

JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
MODEL_INFO_ENTRIES: Final = TypeAdapter(tuple[dict[str, JsonValue], ...])
MONITOR_COMMAND: Final = TypeAdapter(dict[str, JsonValue])
OPENAI_MODEL: Final = "gpt-4o-mini"
MASTER_KEY: Final = "sk-integration-usage-routing-redis-reads"
API_KEY: Final = "synthetic-usage-routing-key"
ENDPOINT_PATHS: Final = MappingProxyType(
    {
        "/v1/chat/completions": ("/v1/chat/completions", "/v1/chat/completions"),
        "/v1/messages": ("/v1/responses", "/v1/responses"),
        "/v1/responses": ("/v1/responses", "/v1/responses"),
    }
)
CHAT_RESPONSE: Final = json.dumps(
    {
        "id": "chatcmpl_usage_routing_redis_reads",
        "object": "chat.completion",
        "created": 1700000000,
        "model": OPENAI_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "redis read contract"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11},
    }
).encode()
RESPONSES_RESPONSE: Final = json.dumps(
    {
        "id": "resp_usage_routing_redis_reads",
        "object": "response",
        "created_at": 1700000000,
        "status": "completed",
        "model": OPENAI_MODEL,
        "output": [
            {
                "id": "msg_usage_routing_redis_reads",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "redis read contract", "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 7, "output_tokens": 4, "total_tokens": 11},
    }
).encode()


def _request_object(body: bytes) -> dict[str, JsonValue]:
    return JSON_OBJECT.validate_json(body)


def _deployment_list(
    model_name: str, api_base: str, deployment_ids: tuple[str, str]
) -> list[dict[str, JsonValue]]:
    return [
        {
            "model_name": model_name,
            "litellm_params": {
                "model": f"openai/{OPENAI_MODEL}",
                "api_base": api_base,
                "api_key": API_KEY,
                "rpm": 1,
            },
            "model_info": {"id": deployment_id},
        }
        for deployment_id in deployment_ids
    ]


def _request_payload(endpoint: str, model_name: str, marker: str) -> dict[str, JsonValue]:
    if endpoint == "/v1/responses":
        return {"model": model_name, "input": marker, "max_output_tokens": 16, "store": False}
    return {"model": model_name, "messages": [{"role": "user", "content": marker}], "max_tokens": 16}


def _expected_wire_body(endpoint: str, marker: str) -> dict[str, JsonValue]:
    if endpoint == "/v1/messages":
        return {
            "model": OPENAI_MODEL,
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": marker}]}],
            "include": ["reasoning.encrypted_content"],
            "max_output_tokens": 16,
        }
    if endpoint == "/v1/responses":
        return {"model": OPENAI_MODEL, "input": marker, "max_output_tokens": 16, "store": False}
    return {"model": OPENAI_MODEL, "messages": [{"role": "user", "content": marker}], "max_tokens": 16}


def _reply(request: Request) -> Reply:
    if request.target == "/v1/models":
        return Reply(body=json.dumps({"object": "list", "data": [{"id": OPENAI_MODEL, "object": "model"}]}).encode())
    return Reply(body=RESPONSES_RESPONSE if request.target == "/v1/responses" else CHAT_RESPONSE)


@contextmanager
def _capture_redis_commands(host: str, port: int) -> Iterator[SimpleQueue[str]]:
    commands: Final = SimpleQueue[str]()
    started: Final = threading.Event()
    armed: Final = threading.Event()
    stopped: Final = threading.Event()
    ready_marker: Final = f"monitor-ready-{uuid.uuid4().hex}"
    stop_marker: Final = f"monitor-stop-{uuid.uuid4().hex}"

    def capture() -> None:
        with Redis(host=host, port=port, socket_timeout=1, decode_responses=True) as client:
            with client.monitor() as monitor:
                started.set()
                stream: Final = iter(monitor.listen())
                while not stopped.is_set():
                    try:
                        record: Final = MONITOR_COMMAND.validate_python(next(stream))
                    except RedisTimeoutError:
                        continue
                    command: Final = record.get("command")
                    if not isinstance(command, str):
                        continue
                    commands.put(command)
                    if ready_marker in command:
                        armed.set()

    thread: Final = threading.Thread(target=capture, daemon=True)
    thread.start()
    try:
        assert started.wait(timeout=5), "Redis MONITOR did not start"
        with Redis(host=host, port=port, socket_timeout=1, decode_responses=True) as client:
            client.set(ready_marker, "ready", ex=1)
        assert armed.wait(timeout=5), "Redis MONITOR did not capture its readiness command"
        yield commands
    finally:
        stopped.set()
        with Redis(host=host, port=port, socket_timeout=1, decode_responses=True) as client:
            client.set(stop_marker, "stop", ex=1)
        thread.join(timeout=5)
        assert not thread.is_alive(), "Redis MONITOR thread survived cleanup"


def _registered_deployment_ids(candidate: Gateway, model_name: str) -> frozenset[str]:
    entries: Final = MODEL_INFO_ENTRIES.validate_python(candidate.get("/model/info")["data"])
    group: Final = tuple(entry for entry in entries if entry["model_name"] == model_name)
    return frozenset(string_value(object_value(entry["model_info"])["id"]) for entry in group)


def _drain_mgets(commands: SimpleQueue[str]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    captured: Final = tuple(commands.get_nowait() for _ in range(commands.qsize()))
    parsed: Final = tuple((line, tuple(shlex.split(line))) for line in captured)
    return tuple((line, arguments) for line, arguments in parsed if arguments and arguments[0] == "MGET")


@pytest.mark.parametrize(
    "endpoint",
    ("/v1/chat/completions", "/v1/messages", "/v1/responses"),
    ids=("chat-completions", "messages", "responses"),
)
def test_proxy_usage_routing_reads_cooldown_tpm_then_rpm_from_redis(
    endpoint: str, tmp_path: Path
) -> None:
    with owned_redis(tmp_path) as cache, wire_server(_reply) as wire:
        run_id: Final = uuid.uuid4().hex
        model_name: Final = f"usage-redis-{run_id}"
        deployment_ids: Final = (f"dep-a-{run_id[:8]}", f"dep-b-{run_id[:8]}")
        configuration: Final = JSON_OBJECT.validate_python(
            yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        )
        config: Final = {
            **configuration,
            "general_settings": {
                **JSON_OBJECT.validate_python(configuration["general_settings"]),
                "store_model_in_db": False,
            },
            "model_list": _deployment_list(model_name, f"{wire.url}/v1", deployment_ids),
            "router_settings": {
                "routing_strategy": "usage-based-routing-v2",
                "redis_host": cache.host,
                "redis_port": cache.port,
            },
        }
        config_path: Final = tmp_path / "usage-routing.yaml"
        config_path.write_text(yaml.safe_dump(config))
        with httpx.Client(base_url=wire.url, timeout=15, trust_env=False) as bootstrap_client:
            bootstrap: Final = Gateway(bootstrap_client, MASTER_KEY, wire.url)
            with owned_proxy(
                bootstrap,
                tmp_path,
                {"REDIS_HOST": cache.host, "REDIS_PORT": str(cache.port), "STORE_MODEL_IN_DB": "False"},
                config=config_path,
            ) as candidate:
                eventually(
                    lambda: _registered_deployment_ids(candidate, model_name),
                    lambda registered: registered == frozenset(deployment_ids),
                    seconds=15,
                )
                wire.drain()
                eventually(
                    lambda: datetime.now(UTC),
                    lambda current: current.second < 40,
                    seconds=65,
                )
                minute: Final = datetime.now(UTC).strftime("%H-%M")
                markers: Final = tuple(f"{run_id}-{index}" for index in range(3))
                payloads: Final = tuple(_request_payload(endpoint, model_name, marker) for marker in markers)
                request_headers: Final = (
                    {"anthropic-version": "2023-06-01"} if endpoint == "/v1/messages" else {}
                )
                with _capture_redis_commands(cache.host, cache.port) as commands:
                    responses: Final = tuple(
                        candidate.request("POST", endpoint, payload, headers=request_headers) for payload in payloads
                    )
                    assert tuple(response.status_code for response in responses) == (200, 200, 429), [
                        response.text for response in responses
                    ]
                    assert "No deployments available" in responses[2].text
                    served_ids: Final = tuple(response.headers["x-litellm-model-id"] for response in responses[:2])
                    assert set(served_ids) == set(deployment_ids), served_ids
                    rpm_keys: Final = tuple(
                        f"{deployment_id}:openai/{OPENAI_MODEL}:rpm:{minute}" for deployment_id in deployment_ids
                    )
                    with Redis(host=cache.host, port=cache.port, decode_responses=True) as redis_client:
                        rpm_values: Final = eventually(
                            lambda: tuple(redis_client.get(key) for key in rpm_keys),
                            lambda values: values == ("1", "1"),
                            seconds=15,
                        )
                    assert rpm_values == ("1", "1")
                received: Final = wire.drain()
                assert len(received) == 2
                assert tuple(request.method for request in received) == ("POST", "POST")
                assert tuple(request.target for request in received) == ENDPOINT_PATHS[endpoint]
                observed_bodies: Final = tuple(_request_object(request.body) for request in received)
                expected_bodies: Final = tuple(_expected_wire_body(endpoint, marker) for marker in markers[:2])
                assert observed_bodies == expected_bodies, observed_bodies
                expected_mget: Final = (
                    "MGET",
                    f"deployment:{deployment_ids[0]}:cooldown",
                    f"deployment:{deployment_ids[1]}:cooldown",
                    f"{deployment_ids[0]}:openai/{OPENAI_MODEL}:tpm:{minute}",
                    f"{deployment_ids[1]}:openai/{OPENAI_MODEL}:tpm:{minute}",
                    *(
                        f"{deployment_id}:openai/{OPENAI_MODEL}:rpm:{minute}"
                        for deployment_id in deployment_ids
                    ),
                )
                mgets: Final = _drain_mgets(commands)
                assert any(arguments == expected_mget for _, arguments in mgets), mgets
                raw_mgets: Final = tuple(line for line, _ in mgets)
                print(f"proxy {endpoint} MGETs: {raw_mgets}")
