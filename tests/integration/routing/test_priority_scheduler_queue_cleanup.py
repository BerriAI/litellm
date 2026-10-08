from __future__ import annotations

import asyncio
import http.client
import json
import re
import signal
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from typing import Final, Literal, TypeVar
from urllib.parse import urlsplit

import httpx
import openai
import psutil
import pytest
import yaml
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.database import read_rows
from integration._support.openai_wire import answering_model_discovery, chat_reply, openai_error, responses_reply
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.redis_process import OwnedRedis, owned_redis
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter
from redis import Redis

T = TypeVar("T")
Endpoint = Literal["chat", "chat-stream", "completions", "completions-stream", "queue", "queue-stream"]
Instance = Literal["first", "second"]

MARKER: Final = re.compile(r"sched-[0-9a-f]{32}")
UNKNOWN_MODEL: Final = "Invalid model name passed in model="
NO_DEPLOYMENTS: Final = "No deployments available"
QUEUE_TIMEOUT: Final = "Request timed out while polling queue"
ROUTER_TIMEOUT_SECONDS: Final = 8
PROMPT_SECONDS: Final = 4
COOLDOWN_SECONDS: Final = 60
USAGE: Final[dict[str, JsonValue]] = {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}
STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
OWNED_CELL_TIMEOUT: Final = 2 * graceful_stop_seconds() + 120
PAIR_CELL_TIMEOUT: Final = 3 * graceful_stop_seconds() + 120
WORKER_HEALTHCHECK_ARGUMENTS: Final = ("--timeout_worker_healthcheck", str(int(graceful_stop_seconds())))
PINNED_CONNECTION_TIMEOUT_SECONDS: Final = 30
PINNED_LIMITS: Final = httpx.Limits(max_connections=1, max_keepalive_connections=1, keepalive_expiry=30)
ENDPOINTS: Final[tuple[Endpoint, ...]] = (
    "chat",
    "chat-stream",
    "completions",
    "completions-stream",
    "queue",
    "queue-stream",
)
PAIR_GROUPS: Final = tuple(f"sched-pair-{row}" for row in ("p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8"))
INMEM_GROUP: Final = "sched-inmem"
OUTAGE_GROUP: Final = "sched-outage"
KILL_GROUP: Final = "sched-kill"
GHOST_ENTRY: Final[list[JsonValue]] = [1, "ghost"]
JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
JSON_LIST: Final = TypeAdapter(list[JsonValue])


def new_marker() -> str:
    return f"sched-{uuid.uuid4().hex}"


def marker_in(text: str) -> str:
    found: Final = MARKER.search(text)
    assert found is not None, text[:300]
    return found.group(0)


def markers_in(text: str) -> frozenset[str]:
    return frozenset(MARKER.findall(text))


def upstream_name(group: str) -> str:
    return f"{group}-upstream"


def queue_key(group: str) -> str:
    return f"scheduler:queue:{group}"


def data_frame(frame: Mapping[str, JsonValue]) -> bytes:
    return b"data: " + json.dumps(frame).encode() + b"\n\n"


def text_completion_reply(marker: str, model: str, *, stream: bool) -> Reply:
    choice: Final[dict[str, JsonValue]] = {
        "text": f"served {marker}",
        "index": 0,
        "logprobs": None,
        "finish_reason": "stop",
    }
    body: Final[dict[str, JsonValue]] = {
        "id": f"cmpl-{marker}",
        "object": "text_completion",
        "created": 1,
        "model": model,
        "choices": [choice],
        "usage": USAGE,
    }
    if not stream:
        return Reply(body=json.dumps(body).encode())
    first: Final = data_frame({**body, "choices": [{**choice, "finish_reason": None}]})
    last: Final = data_frame({**body, "choices": [{**choice, "text": ""}]})
    return Reply(content_type="text/event-stream", chunks=(first, last + b"data: [DONE]\n\n"))


@dataclass(frozen=True, slots=True)
class Upstream:
    refusing: Mapping[str, threading.Event]
    held: SimpleQueue[str]
    release: threading.Event
    hold: frozenset[str]

    def respond(self, request: Request) -> Reply:
        body: Final = JSON_OBJECT.validate_json(request.body)
        model: Final = str(body["model"])
        refusal: Final = self.refusing.get(model)
        if refusal is not None and refusal.is_set():
            return openai_error(401)
        marker: Final = marker_in(request.body.decode())
        if model in self.hold:
            self.held.put(marker)
            assert self.release.wait(timeout=60), "The held burst was never released"
        stream: Final = body.get("stream") is True
        if request.target.endswith("/responses"):
            return responses_reply(f"resp_{marker}", model, f"served {marker}", stream=stream)
        if request.target.endswith("/completions") and not request.target.endswith("/chat/completions"):
            return text_completion_reply(marker, model, stream=stream)
        return chat_reply(f"chatcmpl-{marker}", model, f"served {marker}", stream=stream)


def upstream_for(groups: Sequence[str], *, hold: Sequence[str] = ()) -> Upstream:
    return Upstream(
        {upstream_name(group): threading.Event() for group in groups},
        SimpleQueue(),
        threading.Event(),
        frozenset(upstream_name(group) for group in hold),
    )


def received_markers(wire: Wire) -> tuple[str, ...]:
    return tuple(marker_in(request.body.decode()) for request in wire.drain() if request.method == "POST")


def assert_once(received: Sequence[str], markers: Sequence[str]) -> None:
    counts: Final = {marker: received.count(marker) for marker in markers}
    assert all(count == 1 for count in counts.values()), counts


def assert_never(received: Sequence[str], markers: Sequence[str]) -> None:
    assert not set(markers) & set(received), (markers, received)


def deployment_entry(group: str, wire: Wire) -> dict[str, JsonValue]:
    return {
        "model_name": group,
        "litellm_params": {
            "model": f"openai/{upstream_name(group)}",
            "api_base": wire.url + "/v1",
            "api_key": "synthetic-openai-key",
        },
        "model_info": {"id": f"{group}-deployment"},
    }


def owned_config(
    directory: Path,
    wire: Wire,
    groups: Sequence[str],
    settings: Mapping[str, JsonValue],
    *,
    cancel_on_disconnect: bool = False,
) -> Path:
    base: Final = JSON_OBJECT.validate_python(
        yaml.safe_load((Path(__file__).parents[1] / "proxy_config.yaml").read_text())
    )
    general_settings: Final = base.get("general_settings", {})
    assert isinstance(general_settings, dict)
    general: Final[dict[str, JsonValue]] = {
        **general_settings,
        **({"cancel_on_disconnect": True} if cancel_on_disconnect else {}),
    }
    config: Final[dict[str, JsonValue]] = {
        **base,
        "general_settings": general,
        "router_settings": dict(settings),
        "model_list": [deployment_entry(group, wire) for group in groups],
    }
    path: Final = directory / f"scheduler-{uuid.uuid4().hex[:8]}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def cooldown_settings(cache: OwnedRedis | None) -> dict[str, JsonValue]:
    redis: Final = {} if cache is None else {"redis_host": cache.host, "redis_port": cache.port}
    return {"num_retries": 0, "timeout": ROUTER_TIMEOUT_SECONDS, "cooldown_time": COOLDOWN_SECONDS, **redis}


def redis_settings(cache: OwnedRedis) -> dict[str, JsonValue]:
    return {"num_retries": 0, "redis_host": cache.host, "redis_port": cache.port}


def chat_body(group: str, marker: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {"model": group, "messages": [{"role": "user", "content": marker}], **extra}


def completion_body(group: str, marker: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {"model": group, "prompt": marker, **extra}


@dataclass(frozen=True, slots=True)
class Answer:
    status: int
    text: str
    headers: Mapping[str, str]
    seconds: float

    @property
    def identity(self) -> str:
        return str(JSON_OBJECT.validate_json(self.text)["id"])


def timed(send: Callable[[], httpx.Response]) -> Answer:
    started: Final = time.monotonic()
    response: Final = send()
    return Answer(response.status_code, response.text, dict(response.headers), time.monotonic() - started)


def settled(send: Callable[[], T], text: Callable[[T], str]) -> T:
    return eventually(send, lambda observed: UNKNOWN_MODEL not in text(observed), seconds=30)


def sdk_settled(call: Callable[[], T]) -> T:
    def attempt() -> T | openai.APIStatusError:
        try:
            return call()
        except openai.APIStatusError as error:
            if UNKNOWN_MODEL in error.message:
                return error
            raise

    outcome: Final = eventually(attempt, lambda observed: not isinstance(observed, openai.APIStatusError), seconds=30)
    assert not isinstance(outcome, openai.APIStatusError)
    return outcome


def post(gateway: Gateway, path: str, body: Mapping[str, JsonValue]) -> Answer:
    return settled(lambda: timed(lambda: gateway.request("POST", path, body)), lambda answer: answer.text)


def stream_text(gateway: Gateway, path: str, body: Mapping[str, JsonValue]) -> Answer:
    def send() -> Answer:
        started: Final = time.monotonic()
        with gateway.client.stream(
            "POST", path, json=body, headers={"Authorization": f"Bearer {gateway.key}"}
        ) as response:
            text: Final = "".join(response.iter_text())
            return Answer(response.status_code, text, dict(response.headers), time.monotonic() - started)

    return settled(send, lambda answer: answer.text)


def stream_identities(text: str) -> frozenset[str]:
    frames: Final = tuple(
        JSON_OBJECT.validate_json(line[len("data: ") :])
        for line in text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    )
    assert frames, text
    return frozenset(str(frame["id"]) for frame in frames)


def call(gateway: Gateway, endpoint: Endpoint, group: str, marker: str, priority: int = 1) -> Answer:
    match endpoint:
        case "chat":
            return post(gateway, "/v1/chat/completions", chat_body(group, marker, priority=priority))
        case "chat-stream":
            return stream_text(
                gateway, "/v1/chat/completions", chat_body(group, marker, priority=priority, stream=True)
            )
        case "completions":
            return post(gateway, "/v1/completions", completion_body(group, marker, priority=priority))
        case "completions-stream":
            return stream_text(
                gateway, "/v1/completions", completion_body(group, marker, priority=priority, stream=True)
            )
        case "queue":
            return post(gateway, "/queue/chat/completions", chat_body(group, marker, priority=priority))
        case "queue-stream":
            return stream_text(
                gateway, "/queue/chat/completions", chat_body(group, marker, priority=priority, stream=True)
            )


def served_identity(status: int, text: str, marker: str) -> str:
    assert status == 200, (status, text)
    identities: Final = (
        stream_identities(text) if text.startswith("data:") else frozenset({str(JSON_OBJECT.validate_json(text)["id"])})
    )
    assert identities in ({f"chatcmpl-{marker}"}, {f"cmpl-{marker}"}), (identities, marker)
    assert markers_in(text) == {marker}, text
    return next(iter(identities))


def assert_served(answer: Answer, marker: str) -> str:
    return served_identity(answer.status, answer.text, marker)


def spend_row_lands(identity: str) -> None:
    rows: Final = eventually(
        lambda: read_rows('SELECT spend FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (identity,)),
        lambda values: len(values) == 1,
        seconds=70,
    )
    assert rows[0]["spend"] is not None, rows


def sdk(gateway: Gateway) -> openai.OpenAI:
    return openai.OpenAI(base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0, timeout=30)


def queue_entries(cache: OwnedRedis, group: str) -> list[JsonValue] | None:
    with Redis(host=cache.host, port=cache.port) as client:
        raw: Final = client.get(queue_key(group))
    if raw is None:
        return None
    assert isinstance(raw, bytes), raw
    return JSON_LIST.validate_json(raw)


def write_queue(cache: OwnedRedis, group: str, entries: Sequence[JsonValue]) -> None:
    with Redis(host=cache.host, port=cache.port) as client:
        client.set(queue_key(group), json.dumps(list(entries)))


def persist_queue(cache: OwnedRedis, group: str) -> None:
    with Redis(host=cache.host, port=cache.port) as client:
        client.persist(queue_key(group))


def entry_priority(entry: JsonValue) -> int:
    assert isinstance(entry, list), entry
    priority: Final = entry[0]
    assert isinstance(priority, int), entry
    return priority


def waiting_entries(cache: OwnedRedis, group: str, priority: int) -> list[JsonValue]:
    def read() -> list[JsonValue]:
        return queue_entries(cache, group) or []

    return eventually(
        read, lambda found: any(entry_priority(entry) == priority for entry in found), seconds=PROMPT_SECONDS
    )


def cooled(gateway: Gateway, upstream: Upstream, group: str) -> None:
    upstream.refusing[upstream_name(group)].set()
    trip: Final = post(gateway, "/v1/chat/completions", chat_body(group, new_marker()))
    assert trip.status == 401, (trip.status, trip.text)
    eventually(
        lambda: post(gateway, "/v1/chat/completions", chat_body(group, new_marker())),
        lambda answer: answer.status == 429 and NO_DEPLOYMENTS in answer.text,
        seconds=10,
    )


def assert_refused_at_once(answer: Answer) -> None:
    assert answer.status == 429, (answer.status, answer.text)
    assert NO_DEPLOYMENTS in answer.text, answer.text
    assert answer.seconds < PROMPT_SECONDS, answer.seconds


def assert_timed_out_polling(answer: Answer) -> None:
    assert answer.status == 408, (answer.status, answer.text)
    assert QUEUE_TIMEOUT in answer.text, answer.text
    assert answer.seconds >= ROUTER_TIMEOUT_SECONDS, answer.seconds


@pytest.fixture(scope="module")
def rig_upstream() -> Iterator[tuple[Upstream, Wire]]:
    upstream: Final = upstream_for(())
    with wire_server(answering_model_discovery(upstream.respond)) as wire:
        yield upstream, wire


@pytest.fixture
def rig_model(gateway: Gateway, rig_upstream: tuple[Upstream, Wire]) -> Iterator[tuple[str, Wire]]:
    _, wire = rig_upstream
    with gateway.scenario() as scenario:
        yield scenario.model(api_base=wire.url + "/v1"), wire


def test_sdk_chat_with_priority_is_served_once_and_billed(gateway: Gateway, rig_model: tuple[str, Wire]) -> None:
    model, wire = rig_model
    marker: Final = new_marker()
    client: Final = sdk(gateway)
    raw: Final = sdk_settled(
        lambda: client.chat.completions.with_raw_response.create(
            model=model, messages=[{"role": "user", "content": marker}], extra_body={"priority": 1}
        )
    )
    assert served_identity(raw.status_code, raw.text, marker) == f"chatcmpl-{marker}"
    (upstream_request,) = tuple(request for request in wire.drain() if marker.encode() in request.body)
    assert "priority" not in JSON_OBJECT.validate_json(upstream_request.body), upstream_request.body
    spend_row_lands(f"chatcmpl-{marker}")


def test_sdk_chat_stream_with_priority_is_served_once_and_billed(gateway: Gateway, rig_model: tuple[str, Wire]) -> None:
    model, wire = rig_model
    marker: Final = new_marker()
    client: Final = sdk(gateway)
    chunks: Final = sdk_settled(
        lambda: tuple(
            client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": marker}], stream=True, extra_body={"priority": 1}
            )
        )
    )
    assert {chunk.id for chunk in chunks} == {f"chatcmpl-{marker}"}, chunks
    content: Final = "".join(chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices)
    assert content == f"served {marker}", chunks
    assert_once(received_markers(wire), (marker,))
    spend_row_lands(f"chatcmpl-{marker}")


def test_async_sdk_chat_with_priority_is_served_once(gateway: Gateway, rig_model: tuple[str, Wire]) -> None:
    model, wire = rig_model
    marker: Final = new_marker()

    async def send() -> tuple[int, str]:
        async with openai.AsyncOpenAI(
            base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0, timeout=30
        ) as client:
            raw: Final = await client.chat.completions.with_raw_response.create(
                model=model, messages=[{"role": "user", "content": marker}], extra_body={"priority": 1}
            )
            return raw.status_code, raw.text

    status, text = sdk_settled(lambda: asyncio.run(send()))
    assert served_identity(status, text, marker) == f"chatcmpl-{marker}"
    assert_once(received_markers(wire), (marker,))


def test_raw_chat_with_duplicate_priority_keys_takes_the_last_one(
    gateway: Gateway, rig_model: tuple[str, Wire]
) -> None:
    model, wire = rig_model
    marker: Final = new_marker()
    body: Final = (
        '{"model": "%s", "messages": [{"role": "user", "content": "%s"}], "priority": "not-an-int", "priority": 1}'
        % (model, marker)
    )

    def send() -> httpx.Response:
        return gateway.client.post(
            "/v1/chat/completions",
            content=body.encode(),
            headers={"Authorization": f"Bearer {gateway.key}", "Content-Type": "application/json"},
        )

    answer: Final = settled(lambda: timed(send), lambda observed: observed.text)
    assert_served(answer, marker)
    (upstream_request,) = tuple(request for request in wire.drain() if marker.encode() in request.body)
    assert "priority" not in JSON_OBJECT.validate_json(upstream_request.body), upstream_request.body


@pytest.mark.parametrize("endpoint", ("completions", "completions-stream", "queue", "queue-stream"))
def test_other_prioritized_endpoints_are_served_once_and_billed(
    gateway: Gateway, rig_model: tuple[str, Wire], endpoint: Endpoint
) -> None:
    model, wire = rig_model
    marker: Final = new_marker()
    answer: Final = call(gateway, endpoint, model, marker)
    identity: Final = assert_served(answer, marker)
    if endpoint == "queue":
        assert answer.headers.get("x-litellm-priority") == "1", answer.headers
    assert_once(received_markers(wire), (marker,))
    spend_row_lands(identity)


@pytest.mark.parametrize("priority", ("1", [1], "", "p" * 5120, 0), ids=("string", "list", "empty", "5kb", "zero"))
def test_non_integer_priority_bypasses_the_scheduler_and_is_forwarded(
    gateway: Gateway, rig_model: tuple[str, Wire], priority: JsonValue
) -> None:
    model, wire = rig_model
    marker: Final = new_marker()
    answer: Final = post(gateway, "/v1/chat/completions", chat_body(model, marker, priority=priority))
    assert_served(answer, marker)
    (upstream_request,) = tuple(request for request in wire.drain() if marker.encode() in request.body)
    assert JSON_OBJECT.validate_json(upstream_request.body).get("priority") == priority


def test_unauthenticated_prioritized_request_never_reaches_the_upstream(
    gateway: Gateway, rig_model: tuple[str, Wire]
) -> None:
    model, wire = rig_model
    marker: Final = new_marker()
    response: Final = gateway.client.post("/v1/chat/completions", json=chat_body(model, marker, priority=1))
    assert response.status_code == 401, response.text
    assert_never(received_markers(wire), (marker,))


@pytest.mark.parametrize("path", ("/v1/messages", "/v1/responses"), ids=("messages", "responses"))
def test_priority_on_unscheduled_routes_is_served_once(
    gateway: Gateway, rig_model: tuple[str, Wire], path: str
) -> None:
    model, wire = rig_model
    marker: Final = new_marker()
    body: Final[dict[str, JsonValue]] = (
        {"model": model, "max_tokens": 32, "messages": [{"role": "user", "content": marker}], "priority": 1}
        if path == "/v1/messages"
        else {"model": model, "input": marker, "priority": 1}
    )
    answer: Final = post(gateway, path, body)
    assert answer.status == 200, (answer.status, answer.text)
    assert markers_in(answer.text) == {marker}, answer.text
    assert_once(received_markers(wire), (marker,))


def pinned(gateway: Gateway, stack: ExitStack) -> Gateway:
    client: Final = stack.enter_context(
        httpx.Client(base_url=gateway.client.base_url, timeout=15, trust_env=False, limits=PINNED_LIMITS)
    )
    return Gateway(client, gateway.key, gateway.upstream_url)


@pytest.mark.timeout(OWNED_CELL_TIMEOUT)
def test_in_memory_queue_forgets_served_requests_before_a_cooldown(gateway: Gateway, tmp_path: Path) -> None:
    upstream: Final = upstream_for((INMEM_GROUP,))
    with ExitStack() as stack:
        wire: Final = stack.enter_context(wire_server(answering_model_discovery(upstream.respond)))
        config: Final = owned_config(tmp_path, wire, (INMEM_GROUP,), cooldown_settings(None))
        owned: Final = stack.enter_context(
            owned_proxy_process(
                gateway, tmp_path, {}, config=config, workers=2, extra_arguments=WORKER_HEALTHCHECK_ARGUMENTS
            )
        )
        worker: Final = pinned(owned.gateway, stack)
        served: Final = new_marker()
        assert_served(post(worker, "/v1/chat/completions", chat_body(INMEM_GROUP, served, priority=1)), served)
        cooled(worker, upstream, INMEM_GROUP)
        waiting: Final = (new_marker(), new_marker())
        for marker in waiting:
            assert_refused_at_once(post(worker, "/v1/chat/completions", chat_body(INMEM_GROUP, marker, priority=2)))
        received: Final = received_markers(wire)
        assert_once(received, (served,))
        assert_never(received, waiting)


@dataclass(frozen=True, slots=True)
class Pair:
    first: Gateway
    second: Gateway
    cache: OwnedRedis
    wire: Wire
    upstream: Upstream

    def at(self, instance: Instance) -> Gateway:
        return self.first if instance == "first" else self.second


@pytest.fixture(scope="module")
def pair(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Pair]:
    directory: Final = tmp_path_factory.mktemp("scheduler-pair")
    upstream: Final = upstream_for(PAIR_GROUPS)
    with ExitStack() as stack:
        gateway: Final = stack.enter_context(gateway_from_environment())
        cache: Final = stack.enter_context(owned_redis(directory))
        wire: Final = stack.enter_context(wire_server(answering_model_discovery(upstream.respond)))
        config: Final = owned_config(directory, wire, PAIR_GROUPS, cooldown_settings(cache), cancel_on_disconnect=True)
        overrides: Final = {"REDIS_HOST": cache.host, "REDIS_PORT": str(cache.port)}
        first: Final = stack.enter_context(
            owned_proxy_process(
                gateway, directory, overrides, config=config, workers=2, extra_arguments=WORKER_HEALTHCHECK_ARGUMENTS
            )
        )
        second: Final = stack.enter_context(
            owned_proxy_process(
                gateway, directory, overrides, config=config, workers=2, extra_arguments=WORKER_HEALTHCHECK_ARGUMENTS
            )
        )
        yield Pair(first.gateway, second.gateway, cache, wire, upstream)


@pytest.mark.timeout(PAIR_CELL_TIMEOUT)
def test_served_request_leaves_no_queue_entry_for_the_other_instance(pair: Pair) -> None:
    group: Final = PAIR_GROUPS[0]
    markers: Final = (new_marker(), new_marker(), new_marker())
    assert_served(post(pair.first, "/v1/chat/completions", chat_body(group, markers[0], priority=1)), markers[0])
    assert queue_entries(pair.cache, group) == []
    assert_served(post(pair.second, "/v1/chat/completions", chat_body(group, markers[1], priority=1)), markers[1])
    assert_served(post(pair.first, "/v1/chat/completions", chat_body(group, markers[2], priority=1)), markers[2])
    assert_once(received_markers(pair.wire), markers)


@pytest.mark.timeout(PAIR_CELL_TIMEOUT)
def test_concurrent_prioritized_burst_across_instances_and_endpoints_is_served(pair: Pair) -> None:
    group: Final = PAIR_GROUPS[1]
    primer: Final = new_marker()
    assert_served(post(pair.first, "/v1/chat/completions", chat_body(group, primer, priority=1)), primer)
    instances: Final[tuple[Instance, ...]] = ("first", "second")
    plan: Final = tuple((instance, endpoint, new_marker()) for instance in instances for endpoint in ENDPOINTS)

    def one(item: tuple[Instance, Endpoint, str]) -> Answer:
        instance, endpoint, marker = item
        return call(pair.at(instance), endpoint, group, marker)

    with ThreadPoolExecutor(max_workers=len(plan)) as pool:
        answers: Final = tuple(pool.map(one, plan))
    for (_, _, marker), answer in zip(plan, answers, strict=True):
        assert_served(answer, marker)
    closing: Final = new_marker()
    assert_served(post(pair.second, "/v1/chat/completions", chat_body(group, closing, priority=1)), closing)
    assert_once(received_markers(pair.wire), (primer, *(marker for _, _, marker in plan), closing))


@pytest.mark.timeout(PAIR_CELL_TIMEOUT)
def test_fresh_queue_during_a_cooldown_is_refused_at_once(pair: Pair) -> None:
    group: Final = PAIR_GROUPS[2]
    cooled(pair.second, pair.upstream, group)
    client: Final = sdk(pair.second)
    started: Final = time.monotonic()
    with pytest.raises(openai.APIStatusError) as refused:
        client.completions.create(model=group, prompt=new_marker(), extra_body={"priority": 1})
    assert refused.value.status_code == 429, refused.value.message
    assert NO_DEPLOYMENTS in refused.value.message
    assert time.monotonic() - started < PROMPT_SECONDS
    assert queue_entries(pair.cache, group) == []


@pytest.mark.timeout(PAIR_CELL_TIMEOUT)
def test_served_request_on_one_instance_does_not_block_a_waiter_on_the_other(pair: Pair) -> None:
    group: Final = PAIR_GROUPS[3]
    served: Final = new_marker()
    assert_served(post(pair.first, "/v1/chat/completions", chat_body(group, served, priority=1)), served)
    cooled(pair.second, pair.upstream, group)
    assert_refused_at_once(post(pair.second, "/v1/chat/completions", chat_body(group, new_marker(), priority=2)))


def assert_refused_before_the_router_timeout(answer: Answer) -> None:
    assert answer.status == 429, (answer.status, answer.text)
    assert NO_DEPLOYMENTS in answer.text, answer.text
    assert answer.seconds < ROUTER_TIMEOUT_SECONDS, answer.seconds


def assert_drained(cache: OwnedRedis, group: str) -> None:
    assert queue_entries(cache, group) in ([], None)


@pytest.mark.timeout(PAIR_CELL_TIMEOUT)
def test_waiter_behind_an_expired_dead_replica_entry_is_re_enqueued_and_proceeds(pair: Pair) -> None:
    group: Final = PAIR_GROUPS[4]
    cooled(pair.second, pair.upstream, group)
    write_queue(pair.cache, group, [GHOST_ENTRY])
    answer: Final = post(pair.second, "/queue/chat/completions", chat_body(group, new_marker(), priority=2))
    assert_refused_before_the_router_timeout(answer)
    assert_drained(pair.cache, group)


@pytest.mark.timeout(PAIR_CELL_TIMEOUT)
def test_waiter_behind_a_live_entry_times_out_and_removes_only_itself(pair: Pair) -> None:
    group: Final = PAIR_GROUPS[5]
    cooled(pair.second, pair.upstream, group)
    write_queue(pair.cache, group, [GHOST_ENTRY])
    with ThreadPoolExecutor(max_workers=1) as pool:
        waiter: Final = pool.submit(
            post, pair.second, "/v1/chat/completions", chat_body(group, new_marker(), priority=2)
        )
        waiting_entries(pair.cache, group, 2)
        persist_queue(pair.cache, group)
        assert_timed_out_polling(waiter.result(timeout=30))
    assert queue_entries(pair.cache, group) == [GHOST_ENTRY]


@pytest.mark.timeout(PAIR_CELL_TIMEOUT)
def test_waiter_proceeds_once_the_entry_ahead_is_cleared(pair: Pair) -> None:
    group: Final = PAIR_GROUPS[6]
    cooled(pair.second, pair.upstream, group)
    write_queue(pair.cache, group, [GHOST_ENTRY])
    with ThreadPoolExecutor(max_workers=1) as pool:
        waiter: Final = pool.submit(
            post, pair.second, "/v1/chat/completions", chat_body(group, new_marker(), priority=2)
        )
        entries: Final = waiting_entries(pair.cache, group, 2)
        write_queue(pair.cache, group, [entry for entry in entries if entry_priority(entry) == 2])
        assert_refused_before_the_router_timeout(waiter.result(timeout=30))
    assert_drained(pair.cache, group)


def pinned_connection(gateway: Gateway) -> http.client.HTTPConnection:
    address: Final = urlsplit(str(gateway.client.base_url))
    assert address.hostname is not None and address.port is not None, address
    return http.client.HTTPConnection(address.hostname, address.port, timeout=PINNED_CONNECTION_TIMEOUT_SECONDS)


def send_pinned(connection: http.client.HTTPConnection, key: str, body: Mapping[str, JsonValue]) -> None:
    connection.request(
        "POST",
        "/v1/chat/completions",
        body=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )


def post_pinned(connection: http.client.HTTPConnection, key: str, body: Mapping[str, JsonValue]) -> tuple[int, str]:
    send_pinned(connection, key, body)
    response: Final = connection.getresponse()
    text: Final = response.read().decode()
    assert connection.sock is not None, "the proxy closed the pinned connection"
    return response.status, text


def cooled_over(connection: http.client.HTTPConnection, key: str, upstream: Upstream, group: str) -> None:
    upstream.refusing[upstream_name(group)].set()
    trip: Final = post_pinned(connection, key, chat_body(group, new_marker()))
    assert trip[0] == 401, trip
    eventually(
        lambda: post_pinned(connection, key, chat_body(group, new_marker())),
        lambda answer: answer[0] == 429 and NO_DEPLOYMENTS in answer[1],
        seconds=10,
    )


@pytest.mark.timeout(PAIR_CELL_TIMEOUT)
def test_disconnected_waiter_is_removed_from_the_queue(pair: Pair) -> None:
    group: Final = PAIR_GROUPS[7]
    waiter: Final = pinned_connection(pair.second)
    cooled_over(waiter, pair.second.key, pair.upstream, group)
    write_queue(pair.cache, group, [GHOST_ENTRY])
    send_pinned(waiter, pair.second.key, chat_body(group, new_marker(), priority=2))
    waiting_entries(pair.cache, group, 2)
    persist_queue(pair.cache, group)
    waiter.close()
    eventually(
        lambda: queue_entries(pair.cache, group), lambda entries: entries == [GHOST_ENTRY], seconds=PROMPT_SECONDS
    )


Outcome = Answer | httpx.TransportError


def burst(gateway: Gateway, group: str, count: int) -> tuple[tuple[Endpoint, str, Outcome], ...]:
    plan: Final[tuple[tuple[Endpoint, str], ...]] = tuple(
        (ENDPOINTS[index % len(ENDPOINTS)], new_marker()) for index in range(count)
    )

    def one(item: tuple[Endpoint, str]) -> Outcome:
        endpoint, marker = item
        with httpx.Client(base_url=gateway.client.base_url, timeout=60, trust_env=False) as client:
            try:
                return call(Gateway(client, gateway.key, gateway.upstream_url), endpoint, group, marker)
            except httpx.TransportError as error:
                return error

    with ThreadPoolExecutor(max_workers=count) as pool:
        outcomes: Final = tuple(pool.map(one, plan))
    return tuple((endpoint, marker, outcome) for (endpoint, marker), outcome in zip(plan, outcomes, strict=True))


def assert_all_served(outcomes: Sequence[tuple[Endpoint, str, Outcome]]) -> tuple[str, ...]:
    for _, marker, outcome in outcomes:
        assert isinstance(outcome, Answer), repr(outcome)
        assert_served(outcome, marker)
    return tuple(marker for _, marker, _ in outcomes)


def served_eventually(gateway: Gateway, group: str, marker: str) -> None:
    def send() -> Outcome:
        try:
            return post(gateway, "/v1/chat/completions", chat_body(group, marker, priority=1))
        except httpx.TransportError as error:
            return error

    answer: Final = eventually(send, lambda outcome: isinstance(outcome, Answer), seconds=30)
    assert isinstance(answer, Answer)
    assert_served(answer, marker)


@pytest.mark.timeout(OWNED_CELL_TIMEOUT)
def test_prioritized_requests_survive_a_redis_outage(gateway: Gateway, tmp_path: Path) -> None:
    upstream: Final = upstream_for((OUTAGE_GROUP,))
    with owned_redis(tmp_path) as cache, wire_server(answering_model_discovery(upstream.respond)) as wire:
        config: Final = owned_config(tmp_path, wire, (OUTAGE_GROUP,), redis_settings(cache))
        overrides: Final = {
            "REDIS_HOST": cache.host,
            "REDIS_PORT": str(cache.port),
            "REDIS_CIRCUIT_BREAKER_RECOVERY_TIMEOUT": "1",
        }
        with owned_proxy_process(
            gateway, tmp_path, overrides, config=config, workers=2, extra_arguments=WORKER_HEALTHCHECK_ARGUMENTS
        ) as owned:
            before: Final = assert_all_served(burst(owned.gateway, OUTAGE_GROUP, 12))
            cache.stop()
            during: Final = assert_all_served(burst(owned.gateway, OUTAGE_GROUP, 12))
            cache.start()
            after: Final = assert_all_served(burst(owned.gateway, OUTAGE_GROUP, 12))
            assert_once(received_markers(wire), (*before, *during, *after))
            eventually(lambda: queue_entries(cache, OUTAGE_GROUP), lambda entries: entries in (None, []), seconds=10)


def established_upstream_connections(pid: int, wire: Wire) -> int:
    port: Final = urlsplit(wire.url).port
    return sum(
        1
        for connection in psutil.Process(pid).net_connections(kind="tcp")
        if connection.status == psutil.CONN_ESTABLISHED and connection.raddr and connection.raddr.port == port
    )


@pytest.mark.timeout(OWNED_CELL_TIMEOUT)
def test_sibling_worker_keeps_serving_prioritized_requests_after_a_worker_is_killed(
    gateway: Gateway, tmp_path: Path
) -> None:
    upstream: Final = upstream_for((KILL_GROUP,), hold=(KILL_GROUP,))
    with owned_redis(tmp_path) as cache, wire_server(answering_model_discovery(upstream.respond)) as wire:
        config: Final = owned_config(tmp_path, wire, (KILL_GROUP,), redis_settings(cache))
        overrides: Final = {"REDIS_HOST": cache.host, "REDIS_PORT": str(cache.port)}
        with owned_proxy_process(
            gateway, tmp_path, overrides, config=config, workers=2, extra_arguments=WORKER_HEALTHCHECK_ARGUMENTS
        ) as owned:
            workers: Final = eventually(
                lambda: tuple(int(found.group(1)) for found in STARTED_WORKER.finditer(owned.log.read_text())),
                lambda pids: len(pids) == 2,
                seconds=graceful_stop_seconds(),
            )
            with ThreadPoolExecutor(max_workers=1) as pool:
                pending: Final = pool.submit(burst, owned.gateway, KILL_GROUP, 20)
                eventually(upstream.held.qsize, lambda size: size == 20, seconds=60)
                held_by: Final = {pid: established_upstream_connections(pid, wire) for pid in workers}
                assert sum(held_by.values()) == 20, held_by
                victim_pid, survivor_pid = sorted(workers, key=held_by.__getitem__)
                victim: Final = psutil.Process(victim_pid)
                victim.suspend()
                victim.send_signal(signal.SIGKILL)
                upstream.release.set()
                outcomes: Final = pending.result(timeout=90)
            answered: Final = tuple(outcome for outcome in outcomes if isinstance(outcome[2], Answer))
            failed: Final = tuple(outcome for outcome in outcomes if not isinstance(outcome[2], Answer))
            assert len(answered) == held_by[survivor_pid], (held_by, len(answered))
            assert len(failed) == held_by[victim_pid], (held_by, len(failed))
            assert_all_served(answered)
            follow_up: Final = new_marker()
            served_eventually(owned.gateway, KILL_GROUP, follow_up)
            eventually(
                lambda: len(STARTED_WORKER.findall(owned.log.read_text())),
                lambda count: count == 3,
                seconds=graceful_stop_seconds(),
            )
            assert_once(received_markers(wire), (*(marker for _, marker, _ in outcomes), follow_up))
