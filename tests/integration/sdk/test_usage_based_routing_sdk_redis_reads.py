from __future__ import annotations

import asyncio
import json
import shlex
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from queue import SimpleQueue
from typing import Final

import litellm
import pytest
from integration._support.client import eventually
from integration._support.redis_process import owned_redis
from integration._support.wire import Reply, Request, wire_server
from litellm import Router
from pydantic import JsonValue, TypeAdapter
from redis import Redis
from redis.exceptions import TimeoutError as RedisTimeoutError

JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
MONITOR_COMMAND: Final = TypeAdapter(dict[str, JsonValue])
OPENAI_MODEL: Final = "gpt-4o-mini"
API_KEY: Final = "synthetic-usage-routing-key"
CHAT_RESPONSE: Final = json.dumps(
    {
        "id": "chatcmpl_usage_routing_sdk_redis_reads",
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


def _reply(request: Request) -> Reply:
    return Reply(body=CHAT_RESPONSE)


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


def _drain_mgets(commands: SimpleQueue[str]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    captured: Final = tuple(commands.get_nowait() for _ in range(commands.qsize()))
    parsed: Final = tuple((line, tuple(shlex.split(line))) for line in captured)
    return tuple((line, arguments) for line, arguments in parsed if arguments and arguments[0] == "MGET")


def _model_id(response: object) -> str:
    response_params: Final = getattr(response, "_hidden_params")
    hidden_params: Final = JSON_OBJECT.validate_python(response_params)
    model_id: Final = hidden_params.get("model_id")
    assert isinstance(model_id, str), hidden_params
    return model_id


async def _exercise_router(router: Router, model_name: str, markers: tuple[str, str, str]) -> tuple[str, str]:
    first: Final = await router.acompletion(
        model=model_name, messages=[{"role": "user", "content": markers[0]}], max_tokens=8
    )
    second: Final = await router.acompletion(
        model=model_name, messages=[{"role": "user", "content": markers[1]}], max_tokens=8
    )
    with pytest.raises(litellm.RateLimitError, match="No deployments available"):
        await router.acompletion(
            model=model_name, messages=[{"role": "user", "content": markers[2]}], max_tokens=8
        )
    return _model_id(first), _model_id(second)


def test_sdk_usage_routing_reads_tpm_then_rpm_from_redis(tmp_path: Path) -> None:
    with owned_redis(tmp_path) as cache, wire_server(_reply) as wire:
        run_id: Final = uuid.uuid4().hex
        model_name: Final = f"usage-redis-{run_id}"
        deployment_ids: Final = (f"dep-a-{run_id[:8]}", f"dep-b-{run_id[:8]}")
        router: Final = Router(
            model_list=_deployment_list(model_name, f"{wire.url}/v1", deployment_ids),
            routing_strategy="usage-based-routing-v2",
            redis_host=cache.host,
            redis_port=cache.port,
        )
        try:
            eventually(
                lambda: datetime.now(UTC),
                lambda current: current.second < 40,
                seconds=65,
            )
            minute: Final = datetime.now(UTC).strftime("%H-%M")
            markers: Final = tuple(f"{run_id}-{index}" for index in range(3))
            with _capture_redis_commands(cache.host, cache.port) as commands:
                served_ids: Final = asyncio.run(_exercise_router(router, model_name, markers))
                assert set(served_ids) == set(deployment_ids), served_ids
            received: Final = wire.drain()
            assert len(received) == 2
            assert tuple(request.method for request in received) == ("POST", "POST")
            assert tuple(request.target for request in received) == ("/v1/chat/completions",) * 2
            observed_bodies: Final = tuple(_request_object(request.body) for request in received)
            expected_bodies: Final = tuple(
                {
                    "model": OPENAI_MODEL,
                    "messages": [{"role": "user", "content": marker}],
                    "max_tokens": 8,
                }
                for marker in markers[:2]
            )
            assert observed_bodies == expected_bodies, observed_bodies
            expected_mget: Final = (
                "MGET",
                f"deployment:{deployment_ids[0]}:cooldown",
                f"deployment:{deployment_ids[1]}:cooldown",
                *(f"{deployment_id}:openai/{OPENAI_MODEL}:tpm:{minute}" for deployment_id in deployment_ids),
                *(f"{deployment_id}:openai/{OPENAI_MODEL}:rpm:{minute}" for deployment_id in deployment_ids),
            )
            mgets: Final = _drain_mgets(commands)
            assert any(arguments == expected_mget for _, arguments in mgets), mgets
            raw_mgets: Final = tuple(line for line, _ in mgets)
            print(f"sdk MGETs: {raw_mgets}")
        finally:
            router.reset()
