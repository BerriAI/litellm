from __future__ import annotations

import dataclasses
import json
import re
import uuid
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway, eventually, held, object_value, string_value
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.redis_process import OwnedRedis, owned_redis, redis_commands
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter
from redis import Redis

pytestmark: Final = [pytest.mark.timeout(600)]

JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
MODEL_INFO_ENTRIES: Final = TypeAdapter(tuple[dict[str, JsonValue], ...])
OPENAI_MODEL: Final = "gpt-4o-mini"
MASTER_KEY: Final = "sk-integration-cooldown-read-failure"
GOOD_KEY: Final = "synthetic-cooldown-good-key"
BAD_KEY: Final = "synthetic-cooldown-revoked-key"
COOLDOWN_KEY: Final = re.compile(r"^deployment:(.+):cooldown$")
COOLDOWN_LIST: Final = re.compile(r"cooldown_list=\[([^\]]*)\]")
REDIS_READS: Final = frozenset({"GET", "MGET"})
ENDPOINTS: Final = ("/v1/chat/completions", "/v1/messages", "/v1/responses")
ENDPOINT_IDS: Final = ("chat-completions", "messages", "responses")
WORKERS: Final = 2
BURST_ROUNDS: Final = 4
REDIS_SOCKET_TIMEOUT_SECONDS: Final = 1.0
UNAUTHORIZED: Final = json.dumps(
    {"error": {"message": "Incorrect API key provided", "type": "invalid_request_error", "code": "invalid_api_key"}}
).encode()


@dataclass(frozen=True, slots=True)
class Group:
    name: str
    ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class Rig:
    owned: OwnedProxy
    gateway: Gateway
    redis: OwnedRedis
    wire: Wire
    flaky: Group
    flaky_bad_id: str
    neighbour: Group
    dead: Group
    streamed: Group
    burst: Group
    tier: Group
    smart: Group
    bystanders: Group

    @property
    def configured_ids(self) -> frozenset[str]:
        groups: Final = (
            self.flaky,
            self.neighbour,
            self.dead,
            self.streamed,
            self.burst,
            self.tier,
            self.smart,
            self.bystanders,
        )
        return frozenset().union(*(group.ids for group in groups))


def _streams(request: Request) -> bool:
    body: Final = JSON_OBJECT.validate_python(json.loads(request.body or b"{}"))
    return body.get("stream") is True


def _sse(*events: Mapping[str, JsonValue], done: bool) -> tuple[bytes, ...]:
    frames: Final = tuple(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events)
    return (*frames[:-1], frames[-1] + (b"data: [DONE]\n\n" if done else b""))


def _chat_reply(stream: bool) -> Reply:
    identity: Final = f"chatcmpl-{uuid.uuid4().hex}"
    usage: Final = {"prompt_tokens": 7, "completion_tokens": 4, "total_tokens": 11}
    if not stream:
        return Reply(
            body=json.dumps(
                {
                    "id": identity,
                    "object": "chat.completion",
                    "created": 1700000000,
                    "model": OPENAI_MODEL,
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "scoped"}, "finish_reason": "stop"}
                    ],
                    "usage": usage,
                }
            ).encode()
        )
    chunk: Final = {"id": identity, "object": "chat.completion.chunk", "created": 1700000000, "model": OPENAI_MODEL}
    return Reply(
        content_type="text/event-stream",
        chunks=_sse(
            {**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "scoped"}}]},
            {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            {**chunk, "choices": [], "usage": usage},
            done=True,
        ),
    )


def _responses_reply(stream: bool) -> Reply:
    identity: Final = f"resp_{uuid.uuid4().hex}"
    completed: Final = {
        "id": identity,
        "object": "response",
        "created_at": 1700000000,
        "status": "completed",
        "model": OPENAI_MODEL,
        "output": [
            {
                "id": f"msg_{identity}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "scoped", "annotations": []}],
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 7,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 4,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 11,
        },
    }
    if not stream:
        return Reply(body=json.dumps(completed).encode())
    return Reply(
        content_type="text/event-stream",
        chunks=_sse(
            {
                "type": "response.created",
                "sequence_number": 0,
                "response": {**completed, "status": "in_progress", "output": [], "usage": None},
            },
            {
                "type": "response.output_text.delta",
                "sequence_number": 1,
                "item_id": f"msg_{identity}",
                "output_index": 0,
                "content_index": 0,
                "delta": "scoped",
            },
            {"type": "response.completed", "sequence_number": 2, "response": completed},
            done=False,
        ),
    )


def _reply(request: Request) -> Reply:
    if request.target == "/v1/models":
        return Reply(body=json.dumps({"object": "list", "data": [{"id": OPENAI_MODEL, "object": "model"}]}).encode())
    if BAD_KEY in request.headers.get("authorization", ""):
        return Reply(status=401, body=UNAUTHORIZED)
    if request.target == "/v1/responses":
        return _responses_reply(_streams(request))
    return _chat_reply(_streams(request))


def _group(name: str, prefix: str, count: int) -> Group:
    return Group(name, frozenset(f"{prefix}-{uuid.uuid4().hex[:8]}" for _ in range(count)))


def _deployments(
    group: Group, api_base: str, bad_ids: frozenset[str] = frozenset()
) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        {
            "model_name": group.name,
            "litellm_params": {
                "model": f"openai/{OPENAI_MODEL}",
                "api_base": api_base,
                "api_key": BAD_KEY if deployment_id in bad_ids else GOOD_KEY,
            },
            "model_info": {"id": deployment_id},
        }
        for deployment_id in sorted(group.ids)
    )


def _registered_ids(gateway: Gateway) -> frozenset[str]:
    entries: Final = MODEL_INFO_ENTRIES.validate_python(gateway.get("/model/info")["data"])
    return frozenset(string_value(object_value(entry["model_info"])["id"]) for entry in entries)


@pytest.fixture(scope="module", params=("simple-shuffle", "usage-based-routing-v2"))
def rig(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    strategy: Final = str(request.param)
    tmp_path: Final = tmp_path_factory.mktemp(f"cooldown-failure-{strategy}")
    run_id: Final = uuid.uuid4().hex[:8]
    flaky: Final = _group(f"flaky-{run_id}", "flaky", 2)
    flaky_bad_id: Final = min(flaky.ids)
    neighbour: Final = _group(f"neighbour-{run_id}", "neighbour", 1)
    dead: Final = _group(f"dead-{run_id}", "dead", 1)
    streamed: Final = _group(f"streamed-{run_id}", "streamed", 2)
    burst: Final = _group(f"burst-{run_id}", "burst", 2)
    tier: Final = _group(f"tier-{run_id}", "tier", 2)
    smart: Final = _group(f"smart-{run_id}", "smart", 1)
    bystanders: Final = _group(f"bystanders-{run_id}", "bystander", 4)
    with owned_redis(tmp_path) as cache, wire_server(_reply) as wire:
        api_base: Final = f"{wire.url}/v1"
        model_list: Final = (
            *_deployments(flaky, api_base, frozenset({flaky_bad_id})),
            *_deployments(neighbour, api_base, neighbour.ids),
            *_deployments(dead, api_base, dead.ids),
            *_deployments(streamed, api_base),
            *_deployments(burst, api_base),
            *_deployments(tier, api_base),
            {
                "model_name": smart.name,
                "litellm_params": {
                    "model": "auto_router/complexity_router",
                    "complexity_router_default_model": tier.name,
                    "complexity_router_config": {
                        "tiers": {level: tier.name for level in ("SIMPLE", "MEDIUM", "COMPLEX", "REASONING")}
                    },
                },
                "model_info": {"id": min(smart.ids)},
            },
            *_deployments(bystanders, api_base),
        )
        configuration: Final = JSON_OBJECT.validate_python(
            yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        )
        config_path: Final = tmp_path / "cooldown-failure.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    **configuration,
                    "model_list": list(model_list),
                    "router_settings": {
                        "routing_strategy": strategy,
                        "redis_host": cache.host,
                        "redis_port": cache.port,
                        "num_retries": 2,
                        "cooldown_time": 60,
                    },
                }
            )
        )
        with httpx.Client(base_url=wire.url, timeout=15, trust_env=False) as bootstrap_client:
            bootstrap: Final = Gateway(bootstrap_client, MASTER_KEY, wire.url)
            with owned_proxy_process(
                bootstrap,
                tmp_path,
                {
                    "REDIS_HOST": cache.host,
                    "REDIS_PORT": str(cache.port),
                    "REDIS_SOCKET_TIMEOUT": str(REDIS_SOCKET_TIMEOUT_SECONDS),
                    "DEFAULT_COOLDOWN_REDIS_READ_INTERVAL_SECONDS": "0.001",
                },
                config=config_path,
                workers=WORKERS,
                client_timeout=120,
            ) as owned:
                # A fresh connection per request lets every worker answer, and survives a killed worker
                with httpx.Client(
                    base_url=str(owned.gateway.client.base_url),
                    timeout=120,
                    trust_env=False,
                    limits=httpx.Limits(max_keepalive_connections=0),
                ) as unpooled:
                    gateway: Final = dataclasses.replace(owned.gateway, client=unpooled)
                    built: Final = Rig(
                        owned,
                        gateway,
                        cache,
                        wire,
                        flaky,
                        flaky_bad_id,
                        neighbour,
                        dead,
                        streamed,
                        burst,
                        tier,
                        smart,
                        bystanders,
                    )
                    eventually(
                        lambda: _registered_ids(gateway),
                        lambda registered: built.configured_ids <= registered,
                        seconds=30,
                    )
                    yield built


def _payload(endpoint: str, model: str, stream: bool = False) -> dict[str, JsonValue]:
    marker: Final = f"cooldown-failure-{uuid.uuid4().hex}"
    if endpoint == "/v1/responses":
        return {"model": model, "input": marker, "max_output_tokens": 16, "store": False, "stream": stream}
    return {"model": model, "messages": [{"role": "user", "content": marker}], "max_tokens": 16, "stream": stream}


def _headers(endpoint: str) -> Mapping[str, str]:
    return {"anthropic-version": "2023-06-01"} if endpoint == "/v1/messages" else {}


def _post(rig: Rig, endpoint: str, model: str, stream: bool = False) -> httpx.Response:
    return rig.gateway.request("POST", endpoint, _payload(endpoint, model, stream), headers=_headers(endpoint))


def _cooldown_reads(commands: tuple[tuple[str, ...], ...]) -> tuple[frozenset[str], ...]:
    """The deployment ids whose cooldown key each Redis read carried, one entry per read that carried any"""
    reads: Final = tuple(command for command in commands if command and command[0].upper() in REDIS_READS)
    matched: Final = tuple(
        frozenset(match.group(1) for argument in command[1:] if (match := COOLDOWN_KEY.match(argument)))
        for command in reads
    )
    return tuple(ids for ids in matched if ids)


def _sent(
    rig: Rig, endpoint: str, model: str, stream: bool = False
) -> tuple[httpx.Response, tuple[frozenset[str], ...]]:
    """Sends one request and returns it with the deployment ids of every cooldown read Redis served for it"""
    with redis_commands(rig.redis.host, rig.redis.port) as commands:
        response: Final = _post(rig, endpoint, model, stream)
    return response, _cooldown_reads(commands.drain())


def _read_ids(reads: tuple[frozenset[str], ...]) -> frozenset[str]:
    return frozenset().union(*reads)


def _bad_key_requests(requests: tuple[Request, ...]) -> int:
    return sum(BAD_KEY in request.headers.get("authorization", "") for request in requests)


def _is_cooling(rig: Rig, deployment_id: str) -> bool:
    with Redis(host=rig.redis.host, port=rig.redis.port, socket_timeout=1) as client:
        return bool(client.exists(f"deployment:{deployment_id}:cooldown"))


def _cool_flaky_deployment(rig: Rig) -> None:
    """Sends to the flaky group until its revoked deployment has failed once and is cooling in Redis"""

    def attempt() -> bool:
        response: Final = _post(rig, "/v1/chat/completions", rig.flaky.name)
        assert response.status_code == 200, response.text
        return _is_cooling(rig, rig.flaky_bad_id)

    eventually(attempt, lambda cooling: cooling, seconds=60)


def test_a_cooled_deployment_stays_skipped_by_workers_that_never_saw_it_fail(rig: Rig) -> None:
    _cool_flaky_deployment(rig)
    for pid in sorted(rig.owned.worker_pids()):
        rig.owned.kill_worker(pid)
    rig.wire.drain()

    sent: Final = tuple(_sent(rig, "/v1/chat/completions", rig.flaky.name) for _ in range(8))

    assert tuple(response.status_code for response, _ in sent) == (200,) * len(sent), [r.text for r, _ in sent]
    assert {response.headers["x-litellm-model-id"] for response, _ in sent} <= rig.flaky.ids - {rig.flaky_bad_id}
    assert _bad_key_requests(rig.wire.drain()) == 0
    # A worker that already holds the cooling key in memory reads only the keys it lacks
    for _, reads in sent:
        assert reads, "every request should read cooldown state from Redis"
        assert _read_ids(reads) <= rig.flaky.ids, f"read {sorted(_read_ids(reads) - rig.flaky.ids)} outside the group"


def test_the_no_deployments_error_still_lists_cooled_deployments_outside_the_group(rig: Rig) -> None:
    neighbour_id: Final = min(rig.neighbour.ids)
    dead_id: Final = min(rig.dead.ids)
    for group, deployment_id in ((rig.neighbour, neighbour_id), (rig.dead, dead_id)):
        failed = _post(rig, "/v1/chat/completions", group.name)
        assert failed.status_code == 401, failed.text
        eventually(partial(_is_cooling, rig, deployment_id), lambda cooling: cooling, seconds=30)

    refused: Final = _post(rig, "/v1/chat/completions", rig.dead.name)

    assert refused.status_code == 429, refused.text
    assert "No deployments available for selected model" in refused.text
    listed: Final = COOLDOWN_LIST.search(refused.text)
    assert listed is not None, refused.text
    listed_ids: Final = frozenset(name.strip().strip("'\"") for name in listed.group(1).split(","))
    assert {neighbour_id, dead_id} <= listed_ids, listed_ids


@pytest.mark.parametrize("endpoint", ENDPOINTS, ids=ENDPOINT_IDS)
def test_a_streamed_request_reads_cooldown_keys_of_its_model_group_only(rig: Rig, endpoint: str) -> None:
    response, reads = _sent(rig, endpoint, rig.streamed.name, stream=True)

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "scoped" in response.text
    assert response.headers["x-litellm-model-id"] in rig.streamed.ids
    assert reads, "the request should read cooldown state from Redis"
    assert _read_ids(reads) <= rig.streamed.ids, f"read {sorted(_read_ids(reads) - rig.streamed.ids)} outside the group"


def test_a_complexity_router_request_reads_cooldown_keys_of_its_entry_and_tier_only(rig: Rig) -> None:
    response, reads = _sent(rig, "/v1/chat/completions", rig.smart.name)

    assert response.status_code == 200, response.text
    assert response.headers["x-litellm-model-id"] in rig.tier.ids
    allowed: Final = rig.smart.ids | rig.tier.ids
    assert 1 <= len(reads) <= 2, reads
    assert _read_ids(reads) <= allowed, f"read {sorted(_read_ids(reads) - allowed)} outside the route"


def test_requests_keep_answering_while_redis_hangs_and_each_lands_once(rig: Rig) -> None:
    calls: Final = tuple((endpoint, stream) for endpoint in ENDPOINTS for stream in (False, True)) * BURST_ROUNDS

    with rig.redis.paused(), ThreadPoolExecutor(max_workers=8) as pool:
        futures: Final = tuple(pool.submit(_post, rig, endpoint, rig.burst.name, stream) for endpoint, stream in calls)
        done, pending = wait(futures, timeout=120)
        assert not pending, f"{len(pending)} of {len(calls)} requests still waiting on a hung Redis after 120s"
    statuses: Final = tuple(future.result().status_code for future in futures)
    assert statuses == (200,) * len(calls), [
        future.result().text for future in done if future.result().status_code != 200
    ]

    # The breaker that opened while Redis hung skips Redis until it closes, so the first reads may come from memory
    recoveries: Final[list[tuple[httpx.Response, tuple[frozenset[str], ...]]]] = []

    def recover() -> tuple[httpx.Response, tuple[frozenset[str], ...]]:
        recoveries.append(_sent(rig, "/v1/chat/completions", rig.burst.name))
        return recoveries[-1]

    recovered, reads = eventually(recover, lambda sent: bool(sent[1]), seconds=60)

    assert all(response.status_code == 200 for response, _ in recoveries), [r.text for r, _ in recoveries]
    landed: Final = len(calls) + len(recoveries)
    rows: Final = held(
        lambda: rig.gateway.rows(
            'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE model_group = %s', (rig.burst.name,)
        ),
        lambda found: len(found) == landed,
        holding=3,
        seconds=90,
    )
    assert len({string_value(row["request_id"]) for row in rows}) == landed
    assert recovered.headers["x-litellm-model-id"] in rig.burst.ids
    assert reads == (rig.burst.ids,), f"read {reads}, expected one read of {sorted(rig.burst.ids)}"
