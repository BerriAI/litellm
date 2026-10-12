from __future__ import annotations

import asyncio
import json
import re
import uuid
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal, cast

import anthropic
import httpx
import openai
import pytest
import yaml
from anthropic.types import MessageParam, RawMessageStartEvent
from anthropic.types import ToolParam as AnthropicToolParam
from integration._support.client import Gateway, eventually, gateway_from_environment, string_value
from integration._support.database import read_rows
from integration._support.openai_wire import chat_reply, responses_reply
from integration._support.process import OwnedProxy, WorkerHealthcheck, graceful_stop_seconds, owned_proxy_process
from integration._support.redis_edge import (
    Interception,
    RedisRelay,
    commands_mentioning,
    monitored_commands,
    redis_relay,
)
from integration._support.redis_process import OwnedRedis, owned_redis
from integration._support.responses_vendor import response_identities
from integration._support.wire import Reply, Request, Wire, wire_server
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from openai.types.responses import ResponseInputParam
from openai.types.responses import ToolParam as ResponsesToolParam
from pydantic import JsonValue, TypeAdapter
from redis import Redis

JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
SIMPLE_MODEL: Final = "gpt-4o-mini"
COMPLEX_MODEL: Final = "gpt-4o"
COMPLEX_ASK: Final = "COMPLEX-ASK"
PIN_TTL_SECONDS: Final = 600
SESSION_HEADER: Final = "x-litellm-session-id"
STEP_TOKEN: Final = re.compile(r"step-[0-9a-f]{12}")
NO_KEEPALIVE: Final = httpx.Limits(max_keepalive_connections=0)
Endpoint = Literal["chat", "responses", "messages"]
Turn = Literal["simple", "complex", "continue"]


@dataclass(frozen=True, slots=True)
class Outcome:
    identity: str
    model: str


@dataclass(frozen=True, slots=True)
class Observed:
    outcome: Outcome
    judge_posts: int
    tier: str


@dataclass(frozen=True, slots=True)
class Rig:
    owned: OwnedProxy
    cache: OwnedRedis
    relay: RedisRelay
    judge: Wire
    simple: Wire
    complex_tier: Wire
    router: str
    pin_prefix: str


def _token() -> str:
    return "step-" + uuid.uuid4().hex[:12]


def _session() -> str:
    return "session-" + uuid.uuid4().hex


def _step_token(body: bytes) -> str:
    found: Final = STEP_TOKEN.findall(body.decode(errors="replace"))
    assert found, body
    return found[-1]


def _judge(request: Request) -> Reply:
    if request.method == "GET":
        return Reply(body=json.dumps({"object": "list", "data": []}).encode())
    tier: Final = "COMPLEX" if COMPLEX_ASK.encode() in request.body else "SIMPLE"
    return chat_reply("chatcmpl-judge-" + uuid.uuid4().hex[:8], SIMPLE_MODEL, json.dumps({"tier": tier}), stream=False)


def _tier(model: str) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if request.method == "GET":
            return Reply(body=json.dumps({"object": "list", "data": []}).encode())
        stream: Final = JSON_OBJECT.validate_json(request.body).get("stream") is True
        token: Final = _step_token(request.body)
        if request.target == "/responses":
            return responses_reply(f"resp_{token}", model, f"answer {token}", stream=stream)
        assert request.target == "/chat/completions", request.target
        return chat_reply(f"chatcmpl-{token}", model, f"answer {token}", stream=stream)

    return respond


def _config(directory: Path, wires: tuple[str, str, str], router: str, redis: tuple[str, int] | None) -> Path:
    shared: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    judge, simple, complex_tier = wires
    config: Final = {
        **shared,
        "model_list": [
            {
                "model_name": name,
                "litellm_params": {"model": f"openai/{model}", "api_base": url, "api_key": "synthetic"},
            }
            for name, model, url in (
                ("judge", SIMPLE_MODEL, judge),
                (SIMPLE_MODEL, SIMPLE_MODEL, simple),
                (COMPLEX_MODEL, COMPLEX_MODEL, complex_tier),
            )
        ]
        + [
            {
                "model_name": router,
                "litellm_params": {
                    "model": "auto_router/complexity_router",
                    "complexity_router_config": {
                        "classifier_type": "llm",
                        "classifier_llm_config": {"model": "judge", "timeout_ms": 20000},
                        "classification_mode": "user_turn",
                        "return_raw_model_name": True,
                        "session_affinity_ttl_seconds": PIN_TTL_SECONDS,
                        "tiers": {
                            "SIMPLE": SIMPLE_MODEL,
                            "MEDIUM": SIMPLE_MODEL,
                            "COMPLEX": COMPLEX_MODEL,
                            "REASONING": COMPLEX_MODEL,
                        },
                    },
                },
            }
        ],
        "litellm_settings": {"drop_params": True},
        "router_settings": {
            "num_retries": 0,
            **({"redis_host": redis[0], "redis_port": redis[1]} if redis is not None else {}),
        },
    }
    path: Final = directory / f"{router}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _ask(turn: Turn, token: str) -> str:
    return f"{COMPLEX_ASK} prove the protocol {token}" if turn == "complex" else f"hello {token}"


def _call(endpoint: Endpoint, token: str) -> JsonValue:
    if endpoint == "chat":
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": f"call_{token}", "type": "function", "function": {"name": "get_time", "arguments": "{}"}}
            ],
        }
    if endpoint == "responses":
        return {"type": "function_call", "call_id": f"call_{token}", "name": "get_time", "arguments": "{}"}
    return {
        "role": "assistant",
        "content": [{"type": "tool_use", "id": f"toolu_{token}", "name": "get_time", "input": {}}],
    }


def _result(endpoint: Endpoint, token: str) -> JsonValue:
    if endpoint == "chat":
        return {"role": "tool", "tool_call_id": f"call_{token}", "content": f"10:00 UTC {token}"}
    if endpoint == "responses":
        return {"type": "function_call_output", "call_id": f"call_{token}", "output": f"10:00 UTC {token}"}
    return {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": f"toolu_{token}", "content": f"10:00 UTC {token}"}],
    }


def _turns(endpoint: Endpoint, script: Sequence[tuple[Turn, str]]) -> tuple[JsonValue, ...]:
    def pieces(index: int, turn: Turn, token: str) -> tuple[JsonValue, ...]:
        if turn == "continue":
            return _call(endpoint, token), _result(endpoint, token)
        ask: Final[JsonValue] = {"role": "user", "content": _ask(turn, token)}
        return (ask,) if index == 0 else ({"role": "assistant", "content": "ok"}, ask)

    return tuple(chain.from_iterable(pieces(index, turn, token) for index, (turn, token) in enumerate(script)))


def _tools(endpoint: Endpoint) -> list[JsonValue]:
    if endpoint == "chat":
        return [
            {
                "type": "function",
                "function": {"name": "get_time", "description": "Current time", "parameters": {"type": "object"}},
            }
        ]
    if endpoint == "responses":
        return [
            {"type": "function", "name": "get_time", "description": "Current time", "parameters": {"type": "object"}}
        ]
    return [{"name": "get_time", "description": "Current time", "input_schema": {"type": "object"}}]


def _origin(gateway: Gateway) -> str:
    return str(gateway.client.base_url).rstrip("/")


def _headers(gateway: Gateway, session: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {gateway.key}", SESSION_HEADER: session}


def _chat_httpx(gateway: Gateway, router: str, session: str, turns: tuple[JsonValue, ...]) -> Outcome:
    body: Final = {"model": router, "messages": list(turns), "tools": _tools("chat")}
    with httpx.Client(timeout=30, trust_env=False, limits=NO_KEEPALIVE) as client:
        response: Final = client.post(
            f"{_origin(gateway)}/v1/chat/completions", json=body, headers=_headers(gateway, session)
        )
    assert response.status_code == 200, response.text
    payload: Final = JSON_OBJECT.validate_json(response.content)
    return Outcome(string_value(payload["id"]), string_value(payload["model"]))


def _chat_openai_sync(gateway: Gateway, router: str, session: str, turns: tuple[JsonValue, ...]) -> Outcome:
    with (
        httpx.Client(timeout=30, trust_env=False, limits=NO_KEEPALIVE) as transport,
        openai.OpenAI(
            api_key=gateway.key,
            base_url=f"{_origin(gateway)}/v1",
            max_retries=0,
            http_client=transport,
            default_headers={SESSION_HEADER: session},
        ) as client,
    ):
        completion: Final = client.chat.completions.create(
            model=router,
            messages=cast(list[ChatCompletionMessageParam], list(turns)),
            tools=cast(list[ChatCompletionToolParam], _tools("chat")),
        )
        return Outcome(completion.id, completion.model)


def _chat_openai_async_stream(gateway: Gateway, router: str, session: str, turns: tuple[JsonValue, ...]) -> Outcome:
    async def run() -> Outcome:
        async with (
            httpx.AsyncClient(timeout=30, trust_env=False, limits=NO_KEEPALIVE) as transport,
            openai.AsyncOpenAI(
                api_key=gateway.key,
                base_url=f"{_origin(gateway)}/v1",
                max_retries=0,
                http_client=transport,
                default_headers={SESSION_HEADER: session},
            ) as client,
        ):
            stream: Final = await client.chat.completions.create(
                model=router,
                messages=cast(list[ChatCompletionMessageParam], list(turns)),
                tools=cast(list[ChatCompletionToolParam], _tools("chat")),
                stream=True,
            )
            chunks: Final = [chunk async for chunk in stream]
            assert chunks and len({chunk.id for chunk in chunks}) == 1, chunks
            return Outcome(chunks[-1].id, chunks[-1].model)

    return asyncio.run(run())


def _responses_httpx(gateway: Gateway, router: str, session: str, turns: tuple[JsonValue, ...]) -> Outcome:
    body: Final = {"model": router, "input": list(turns), "tools": _tools("responses"), "max_output_tokens": 32}
    with httpx.Client(timeout=30, trust_env=False, limits=NO_KEEPALIVE) as client:
        response: Final = client.post(f"{_origin(gateway)}/v1/responses", json=body, headers=_headers(gateway, session))
    assert response.status_code == 200, response.text
    payload: Final = JSON_OBJECT.validate_json(response.content)
    return Outcome(string_value(payload["id"]), string_value(payload["model"]))


def _responses_openai_async_stream(
    gateway: Gateway, router: str, session: str, turns: tuple[JsonValue, ...]
) -> Outcome:
    async def run() -> Outcome:
        async with (
            httpx.AsyncClient(timeout=30, trust_env=False, limits=NO_KEEPALIVE) as transport,
            openai.AsyncOpenAI(
                api_key=gateway.key,
                base_url=f"{_origin(gateway)}/v1",
                max_retries=0,
                http_client=transport,
                default_headers={SESSION_HEADER: session},
            ) as client,
        ):
            stream: Final = await client.responses.create(
                model=router,
                input=cast(ResponseInputParam, list(turns)),
                tools=cast(list[ResponsesToolParam], _tools("responses")),
                max_output_tokens=32,
                stream=True,
            )
            events: Final = [event async for event in stream]
            completed: Final = [event for event in events if event.type == "response.completed"]
            assert len(completed) == 1, [event.type for event in events]
            return Outcome(completed[0].response.id, str(completed[0].response.model))

    return asyncio.run(run())


def _messages_anthropic_sync(gateway: Gateway, router: str, session: str, turns: tuple[JsonValue, ...]) -> Outcome:
    with (
        httpx.Client(timeout=30, trust_env=False, limits=NO_KEEPALIVE) as transport,
        anthropic.Anthropic(
            api_key=gateway.key,
            base_url=_origin(gateway),
            max_retries=0,
            http_client=transport,
            default_headers={SESSION_HEADER: session},
        ) as client,
    ):
        message: Final = client.messages.create(
            model=router,
            max_tokens=32,
            messages=cast(list[MessageParam], list(turns)),
            tools=cast(list[AnthropicToolParam], _tools("messages")),
        )
        return Outcome(message.id, str(message.model))


def _messages_anthropic_async_stream(
    gateway: Gateway, router: str, session: str, turns: tuple[JsonValue, ...]
) -> Outcome:
    async def run() -> Outcome:
        async with (
            httpx.AsyncClient(timeout=30, trust_env=False, limits=NO_KEEPALIVE) as transport,
            anthropic.AsyncAnthropic(
                api_key=gateway.key,
                base_url=_origin(gateway),
                max_retries=0,
                http_client=transport,
                default_headers={SESSION_HEADER: session},
            ) as client,
        ):
            stream: Final = await client.messages.create(
                model=router,
                max_tokens=32,
                messages=cast(list[MessageParam], list(turns)),
                tools=cast(list[AnthropicToolParam], _tools("messages")),
                stream=True,
            )
            events: Final = [event async for event in stream]
            starts: Final = [event for event in events if isinstance(event, RawMessageStartEvent)]
            assert len(starts) == 1, [event.type for event in events]
            return Outcome(starts[0].message.id, str(starts[0].message.model))

    return asyncio.run(run())


Recorder = Callable[[str, object], None]


def _record(record_property: Recorder, *observed: Observed) -> None:
    record_property("response_ids", " ".join(item.outcome.identity for item in observed))


Caller = Callable[[Gateway, str, str, tuple[JsonValue, ...]], Outcome]
CLIENTS: Final[MappingProxyType[str, tuple[Endpoint, Caller]]] = MappingProxyType(
    {
        "chat-openai-sync": ("chat", _chat_openai_sync),
        "chat-openai-async-stream": ("chat", _chat_openai_async_stream),
        "responses-httpx-sync": ("responses", _responses_httpx),
        "responses-openai-async-stream": ("responses", _responses_openai_async_stream),
        "messages-anthropic-sync": ("messages", _messages_anthropic_sync),
        "messages-anthropic-async-stream": ("messages", _messages_anthropic_async_stream),
    }
)


def _serving_pid(owned: OwnedProxy) -> int | None:
    with httpx.Client(timeout=5, trust_env=False, limits=NO_KEEPALIVE) as client:
        try:
            response: Final = client.get(f"{_origin(owned.gateway)}/health/integration-readiness")
        except httpx.TransportError:
            return None
    identity: Final = response.headers.get("x-integration-worker-pid", "")
    return int(identity) if response.status_code == 200 and identity.isdecimal() else None


def _workers(owned: OwnedProxy) -> tuple[int, int]:
    pids: Final = sorted(eventually(owned.worker_pids, lambda pids: len(pids) == 2, seconds=30))
    return pids[0], pids[1]


@contextmanager
def _only(owned: OwnedProxy, worker: int) -> Iterator[None]:
    others: Final = owned.worker_pids() - {worker}
    with ExitStack() as stack:
        for other in others:
            stack.enter_context(owned.suspended(other))
        eventually(lambda: _serving_pid(owned), lambda pid: pid == worker, seconds=15)
        yield


def _posts(wire: Wire) -> tuple[Request, ...]:
    return tuple(request for request in wire.drain() if request.method == "POST")


def _drain_all(rig_wires: tuple[Wire, ...]) -> None:
    for wire in rig_wires:
        wire.drain()


def _tier_that_received(token: str, simple: Wire, complex_tier: Wire) -> str:
    hits: Final = tuple(
        name
        for name, wire in (("simple", simple), ("complex", complex_tier))
        if any(token.encode() in request.body for request in _posts(wire))
    )
    assert len(hits) == 1, hits
    return hits[0]


def _step(
    rig: Rig, endpoint: Endpoint, call: Caller, session: str, script: Sequence[tuple[Turn, str]], worker: int | None
) -> Observed:
    _drain_all((rig.judge, rig.simple, rig.complex_tier))
    turns: Final = _turns(endpoint, script)
    with _only(rig.owned, worker) if worker is not None else ExitStack():
        outcome: Final = call(rig.owned.gateway, rig.router, session, turns)
    return Observed(outcome, len(_posts(rig.judge)), _tier_that_received(script[-1][1], rig.simple, rig.complex_tier))


def _pin(cache: OwnedRedis, key: str) -> JsonValue | None:
    with Redis(host=cache.host, port=cache.port, socket_timeout=1, decode_responses=True) as client:
        raw: Final = client.get(key)
    return None if raw is None else JSON_OBJECT.validate_json(str(raw))


def _write_raw_pin(cache: OwnedRedis, key: str, raw: str) -> None:
    with Redis(host=cache.host, port=cache.port, socket_timeout=1, decode_responses=True) as client:
        client.set(key, raw, ex=PIN_TTL_SECONDS)


def _delete_pin(cache: OwnedRedis, key: str) -> None:
    with Redis(host=cache.host, port=cache.port, socket_timeout=1, decode_responses=True) as client:
        client.delete(key)


def _pin_key_for(cache: OwnedRedis, session: str) -> str:
    with Redis(host=cache.host, port=cache.port, socket_timeout=1, decode_responses=True) as client:
        keys: Final = tuple(
            str(key) for key in client.scan_iter(match=f"complexity_router_session_affinity:*:{session}")
        )
    assert len(keys) == 1, keys
    return keys[0]


def _pinned(model: str, tier: str) -> dict[str, JsonValue]:
    return {"model": model, "tier": tier}


def _spend_model(candidates: frozenset[str]) -> str:
    keys: Final = tuple(sorted(candidates))
    placeholders: Final = ", ".join("%s" for _ in keys)
    rows: Final = eventually(
        lambda: read_rows(
            f'SELECT request_id, model FROM "LiteLLM_SpendLogs" WHERE request_id IN ({placeholders})', keys
        ),
        lambda rows: len(rows) == 1,
        seconds=70,
    )
    return string_value(rows[0]["model"]).removeprefix("openai/")


def _spend_agrees(observed: Observed, token: str) -> None:
    candidates: Final = response_identities(observed.outcome.identity) | {
        observed.outcome.identity,
        f"chatcmpl-{token}",
        f"resp_{token}",
    }
    assert _spend_model(candidates) == observed.outcome.model, (observed, token)


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    directory: Final = tmp_path_factory.mktemp("session_pin_redis_first")
    with (
        gateway_from_environment() as lane,
        owned_redis(directory) as cache,
        redis_relay(cache.host, cache.port) as relay,
        wire_server(_judge) as judge,
        wire_server(_tier(SIMPLE_MODEL)) as simple,
        wire_server(_tier(COMPLEX_MODEL)) as complex_tier,
    ):
        router: Final = "pinned-" + uuid.uuid4().hex[:8]
        config: Final = _config(directory, (judge.url, simple.url, complex_tier.url), router, ("127.0.0.1", relay.port))
        with owned_proxy_process(
            lane, directory, {}, config=config, workers=2, healthcheck=WorkerHealthcheck.LENIENT
        ) as owned:
            probe: Final = _session()
            _chat_httpx(owned.gateway, router, probe, _turns("chat", (("simple", _token()),)))
            prefix: Final = _pin_key_for(cache, probe).removesuffix(probe)
            yield Rig(owned, cache, relay, judge, simple, complex_tier, router, prefix)


@pytest.mark.parametrize("client", tuple(CLIENTS), ids=tuple(CLIENTS))
def test_continuation_on_the_other_worker_follows_the_tier_the_session_moved_to(
    rig: Rig, client: str, record_property: Recorder
) -> None:
    endpoint, call = CLIENTS[client]
    session: Final = _session()
    first, second = _workers(rig.owned)
    script: Final = (("simple", _token()), ("continue", _token()), ("complex", _token()), ("continue", _token()))
    step1: Final = _step(rig, endpoint, call, session, script[:1], first)
    step2: Final = _step(rig, endpoint, call, session, script[:2], second)
    step3: Final = _step(rig, endpoint, call, session, script[:3], second)
    step4: Final = _step(rig, endpoint, call, session, script[:4], first)
    assert (step1.tier, step1.judge_posts) == ("simple", 1), step1
    assert (step2.tier, step2.judge_posts, step2.outcome.model) == ("simple", 0, step1.outcome.model), step2
    assert (step3.tier, step3.judge_posts) == ("complex", 1) and step3.outcome.model != step1.outcome.model, step3
    assert (step4.tier, step4.judge_posts, step4.outcome.model) == ("complex", 0, step3.outcome.model), step4
    assert _pin(rig.cache, rig.pin_prefix + session) == _pinned(step3.outcome.model, "COMPLEX")
    _record(record_property, step1, step2, step3, step4)
    for observed, (_, token) in zip((step1, step2, step3, step4), script, strict=True):
        _spend_agrees(observed, token)


def test_a_continuation_refreshes_the_pin_ttl_in_redis_instead_of_rewriting_it(
    rig: Rig, record_property: Recorder
) -> None:
    session: Final = _session()
    first, second = _workers(rig.owned)
    script: Final = (("simple", _token()), ("continue", _token()), ("complex", _token()), ("continue", _token()))
    _step(rig, "chat", _chat_httpx, session, script[:1], first)
    with monitored_commands(rig.cache.host, rig.cache.port) as commands:
        step2: Final = _step(rig, "chat", _chat_httpx, session, script[:2], second)
        on_miss: Final = commands_mentioning(commands, rig.pin_prefix + session)
    assert step2.tier == "simple", step2
    assert tuple(command[0] for command in on_miss) == ("GET", "EXPIRE"), on_miss
    assert on_miss[1][2] == str(PIN_TTL_SECONDS), on_miss
    _step(rig, "chat", _chat_httpx, session, script[:3], second)
    with monitored_commands(rig.cache.host, rig.cache.port) as commands:
        step4: Final = _step(rig, "chat", _chat_httpx, session, script[:4], first)
        on_stale_copy: Final = commands_mentioning(commands, rig.pin_prefix + session)
    assert step4.tier == "complex", step4
    assert tuple(command[0] for command in on_stale_copy) == ("GET", "EXPIRE"), on_stale_copy
    _record(record_property, step2, step4)


def test_a_pin_deleted_in_redis_is_not_resurrected_from_a_worker_copy(rig: Rig, record_property: Recorder) -> None:
    session: Final = _session()
    first, _ = _workers(rig.owned)
    script: Final = (("complex", _token()), ("continue", _token()))
    step1: Final = _step(rig, "chat", _chat_httpx, session, script[:1], first)
    assert (step1.tier, step1.judge_posts) == ("complex", 1), step1
    _delete_pin(rig.cache, rig.pin_prefix + session)
    step2: Final = _step(rig, "chat", _chat_httpx, session, script[:2], first)
    assert (step2.tier, step2.judge_posts) == ("complex", 1), step2
    _record(record_property, step1, step2)
    assert _pin(rig.cache, rig.pin_prefix + session) == _pinned(COMPLEX_MODEL, "COMPLEX")


def test_a_newer_pin_written_between_the_read_and_the_refresh_wins(rig: Rig, record_property: Recorder) -> None:
    session: Final = _session()
    key: Final = rig.pin_prefix + session
    first, second = _workers(rig.owned)
    script: Final = (("complex", _token()), ("continue", _token()), ("continue", _token()))
    _step(rig, "chat", _chat_httpx, session, script[:1], first)
    newer: Final = json.dumps(_pinned(SIMPLE_MODEL, "SIMPLE"))
    rig.relay.intercept(
        Interception(key, frozenset({"EXPIRE", "SET"}), lambda: _write_raw_pin(rig.cache, key, newer), "newer-pin")
    )
    try:
        step2: Final = _step(rig, "chat", _chat_httpx, session, script[:2], first)
        fired: Final = rig.relay.drain_fired()
    finally:
        rig.relay.clear()
    assert (step2.tier, step2.judge_posts) == ("complex", 0), step2
    assert fired == ("newer-pin:EXPIRE",), fired
    assert _pin(rig.cache, key) == _pinned(SIMPLE_MODEL, "SIMPLE")
    step3: Final = _step(rig, "chat", _chat_httpx, session, script[:3], second)
    assert (step3.tier, step3.judge_posts) == ("simple", 0), step3
    _record(record_property, step2, step3)


def test_a_pin_deleted_after_the_read_is_restored_only_while_no_newer_pin_exists(
    rig: Rig, record_property: Recorder
) -> None:
    session: Final = _session()
    key: Final = rig.pin_prefix + session
    first, _ = _workers(rig.owned)
    script: Final = (("complex", _token()), ("continue", _token()))
    _step(rig, "chat", _chat_httpx, session, script[:1], first)
    newer: Final = json.dumps(_pinned(SIMPLE_MODEL, "SIMPLE"))
    rig.relay.intercept(
        Interception(key, frozenset({"EXPIRE", "SET"}), lambda: _delete_pin(rig.cache, key), "deleted-after-read")
    )
    rig.relay.intercept(
        Interception(key, frozenset({"SET"}), lambda: _write_raw_pin(rig.cache, key, newer), "newer-before-restore")
    )
    try:
        step2: Final = _step(rig, "chat", _chat_httpx, session, script[:2], first)
        fired: Final = rig.relay.drain_fired()
        pending: Final = rig.relay.pending()
    finally:
        rig.relay.clear()
    assert (step2.tier, step2.judge_posts) == ("complex", 0), step2
    assert fired == ("deleted-after-read:EXPIRE", "newer-before-restore:SET") and pending == (), (fired, pending)
    assert _pin(rig.cache, key) == _pinned(SIMPLE_MODEL, "SIMPLE")
    _record(record_property, step2)


def test_a_model_only_pin_is_upgraded_in_place(rig: Rig, record_property: Recorder) -> None:
    session: Final = _session()
    key: Final = rig.pin_prefix + session
    first, _ = _workers(rig.owned)
    _write_raw_pin(rig.cache, key, json.dumps(COMPLEX_MODEL))
    script: Final = (("simple", _token()), ("continue", _token()))
    step: Final = _step(rig, "chat", _chat_httpx, session, script, first)
    assert (step.tier, step.judge_posts) == ("complex", 0), step
    upgraded: Final = _pin(rig.cache, key)
    assert isinstance(upgraded, dict), upgraded
    assert upgraded["model"] == COMPLEX_MODEL and upgraded["tier"] in ("COMPLEX", "REASONING"), upgraded
    _record(record_property, step)


@pytest.mark.parametrize(
    "raw",
    (
        json.dumps({"model": COMPLEX_MODEL, "tier": "BOGUS"}),
        "5",
        "",
        "x" * 5000,
        json.dumps([COMPLEX_MODEL]),
        json.dumps({"model": 7, "tier": "COMPLEX"}),
    ),
    ids=("unknown-tier", "integer", "empty", "5kb-garbage", "one-element-list", "non-string-model"),
)
def test_an_unusable_pin_in_redis_is_classified_past_and_replaced(
    rig: Rig, raw: str, record_property: Recorder
) -> None:
    session: Final = _session()
    key: Final = rig.pin_prefix + session
    first, _ = _workers(rig.owned)
    _write_raw_pin(rig.cache, key, raw)
    script: Final = (("complex", _token()), ("continue", _token()), ("continue", _token()))
    step: Final = _step(rig, "chat", _chat_httpx, session, script[:2], first)
    assert (step.tier, step.judge_posts) == ("complex", 1), step
    assert _pin(rig.cache, key) == _pinned(COMPLEX_MODEL, "COMPLEX")
    after: Final = _step(rig, "chat", _chat_httpx, session, script, first)
    assert (after.tier, after.judge_posts) == ("complex", 0), after
    _record(record_property, step, after)


def test_a_burst_of_continuations_across_both_workers_replays_the_pin_once_each(
    rig: Rig, record_property: Recorder
) -> None:
    session: Final = _session()
    key: Final = rig.pin_prefix + session
    first, second = _workers(rig.owned)
    opening: Final = (("complex", _token()), ("continue", _token()))
    _step(rig, "chat", _chat_httpx, session, opening[:1], first)
    _step(rig, "chat", _chat_httpx, session, opening, second)
    pinned: Final = _pin(rig.cache, key)
    _drain_all((rig.judge, rig.simple, rig.complex_tier))
    tokens: Final = tuple(_token() for _ in range(12))
    with ThreadPoolExecutor(max_workers=12) as pool:
        outcomes: Final = tuple(
            pool.map(
                lambda token: _chat_httpx(
                    rig.owned.gateway, rig.router, session, _turns("chat", (*opening, ("continue", token)))
                ),
                tokens,
            )
        )
    assert {outcome.model for outcome in outcomes} == {COMPLEX_MODEL}, outcomes
    assert len({outcome.identity for outcome in outcomes}) == 12, outcomes
    assert _posts(rig.judge) == () and _posts(rig.simple) == ()
    assert len(_posts(rig.complex_tier)) == 12
    assert _pin(rig.cache, key) == pinned
    record_property("response_ids", " ".join(outcome.identity for outcome in outcomes))
    for outcome, token in zip(outcomes, tokens, strict=True):
        _spend_agrees(Observed(outcome, 0, "complex"), token)


@contextmanager
def _own_rig(
    directory: Path,
    redis: tuple[str, int] | None,
    workers: int,
    healthcheck: WorkerHealthcheck,
    overrides: dict[str, str],
) -> Iterator[tuple[OwnedProxy, Wire, Wire, Wire, str]]:
    with (
        gateway_from_environment() as lane,
        wire_server(_judge) as judge,
        wire_server(_tier(SIMPLE_MODEL)) as simple,
        wire_server(_tier(COMPLEX_MODEL)) as complex_tier,
    ):
        router: Final = "own-" + uuid.uuid4().hex[:8]
        config: Final = _config(directory, (judge.url, simple.url, complex_tier.url), router, redis)
        with owned_proxy_process(
            lane, directory, overrides, config=config, workers=workers, healthcheck=healthcheck
        ) as owned:
            yield owned, judge, simple, complex_tier, router


def _own_step(
    owned: OwnedProxy,
    wires: tuple[Wire, Wire, Wire],
    router: str,
    session: str,
    script: Sequence[tuple[Turn, str]],
    worker: int | None,
) -> Observed:
    judge, simple, complex_tier = wires
    _drain_all(wires)
    with _only(owned, worker) if worker is not None else ExitStack():
        outcome: Final = _chat_httpx(owned.gateway, router, session, _turns("chat", script))
    return Observed(outcome, len(_posts(judge)), _tier_that_received(script[-1][1], simple, complex_tier))


def test_a_redis_outage_serves_continuations_from_worker_copies_and_recovery_reads_redis_again(
    tmp_path: Path, record_property: Recorder
) -> None:
    with owned_redis(tmp_path) as cache:
        with _own_rig(
            tmp_path,
            (cache.host, cache.port),
            2,
            WorkerHealthcheck.LENIENT,
            {"REDIS_CIRCUIT_BREAKER_RECOVERY_TIMEOUT": "1"},
        ) as (owned, judge, simple, complex_tier, router):
            wires: Final = (judge, simple, complex_tier)
            session: Final = _session()
            first, second = _workers(owned)
            script: Final = (("complex", _token()), ("continue", _token()), ("continue", _token()))
            step1: Final = _own_step(owned, wires, router, session, script[:1], first)
            assert (step1.tier, step1.judge_posts) == ("complex", 1), step1
            key: Final = _pin_key_for(cache, session)
            cache.stop()
            held: Final = _own_step(owned, wires, router, session, script[:2], first)
            assert (held.tier, held.judge_posts) == ("complex", 0), held
            cold: Final = _own_step(owned, wires, router, session, script[:3], second)
            assert (cold.tier, cold.judge_posts) == ("complex", 1), cold
            cache.start()
            assert _pin(cache, key) is None
            recovered: Final = eventually(
                lambda: _own_step(owned, wires, router, session, (*script, ("continue", _token())), first),
                lambda observed: observed.judge_posts == 1,
                seconds=30,
            )
            assert recovered.tier == "complex", recovered
            assert _pin(cache, key) == _pinned(COMPLEX_MODEL, "COMPLEX")
            _record(record_property, step1, held, cold, recovered)


def test_without_redis_the_pin_lives_in_worker_memory(tmp_path: Path, record_property: Recorder) -> None:
    with _own_rig(tmp_path, None, 1, WorkerHealthcheck.STRICT, {}) as (owned, judge, simple, complex_tier, router):
        wires: Final = (judge, simple, complex_tier)
        session: Final = _session()
        script: Final = (("simple", _token()), ("continue", _token()), ("complex", _token()), ("continue", _token()))
        steps: Final = tuple(_own_step(owned, wires, router, session, script[: index + 1], None) for index in range(4))
        assert tuple((step.tier, step.judge_posts) for step in steps) == (
            ("simple", 1),
            ("simple", 0),
            ("complex", 1),
            ("complex", 0),
        ), steps
        assert steps[3].outcome.model == steps[2].outcome.model != steps[0].outcome.model, steps
        _record(record_property, *steps)


def _burst_continuation(
    owned: OwnedProxy, router: str, session: str, opening: tuple[tuple[Turn, str], ...], token: str
) -> tuple[int | None, Outcome | None]:
    body: Final = {
        "model": router,
        "messages": list(_turns("chat", (*opening, ("continue", token)))),
        "tools": _tools("chat"),
    }
    try:
        with httpx.Client(timeout=60, trust_env=False, limits=NO_KEEPALIVE) as client:
            response: Final = client.post(
                f"{_origin(owned.gateway)}/v1/chat/completions", json=body, headers=_headers(owned.gateway, session)
            )
    except httpx.TransportError:
        return None, None
    if response.status_code != 200:
        return response.status_code, None
    payload: Final = JSON_OBJECT.validate_json(response.content)
    return 200, Outcome(string_value(payload["id"]), string_value(payload["model"]))


@pytest.mark.timeout(2 * graceful_stop_seconds() + 120)
def test_a_worker_killed_mid_burst_is_replaced_by_one_that_reads_the_pin_from_redis(
    tmp_path: Path, record_property: Recorder
) -> None:
    with owned_redis(tmp_path) as cache:
        with _own_rig(tmp_path, (cache.host, cache.port), 2, WorkerHealthcheck.STRICT, {}) as (
            owned,
            judge,
            simple,
            complex_tier,
            router,
        ):
            wires: Final = (judge, simple, complex_tier)
            session: Final = _session()
            opening: Final = (("complex", _token()), ("continue", _token()))
            step1: Final = _own_step(owned, wires, router, session, opening[:1], None)
            assert (step1.tier, step1.judge_posts) == ("complex", 1), step1
            workers: Final = _workers(owned)
            _drain_all(wires)
            tokens: Final = tuple(_token() for _ in range(20))
            with ThreadPoolExecutor(max_workers=20) as pool:
                futures: Final = tuple(
                    pool.submit(_burst_continuation, owned, router, session, opening, token) for token in tokens
                )
                with owned.replacing_worker(workers[0]) as kill:
                    kill()
                    records: Final = tuple(future.result() for future in futures)
            completed: Final = tuple(outcome for status, outcome in records if status == 200 and outcome is not None)
            assert completed, records
            assert {outcome.model for outcome in completed} == {COMPLEX_MODEL}, records
            assert len({outcome.identity for outcome in completed}) == len(completed), records
            assert all(status in (None, 200) or status >= 500 for status, _ in records), records
            assert _posts(judge) == () and _posts(simple) == ()
            probe: Final = _own_step(owned, wires, router, session, (*opening, ("continue", _token())), None)
            assert (probe.tier, probe.judge_posts, probe.outcome.model) == ("complex", 0, COMPLEX_MODEL), probe
            assert _pin(cache, _pin_key_for(cache, session)) == _pinned(COMPLEX_MODEL, "COMPLEX")
            record_property("burst_ids", " ".join(outcome.identity for outcome in completed))
            _record(record_property, probe)
