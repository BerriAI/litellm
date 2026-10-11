"""Chaos rows for the response cache on an owned two-worker proxy.

Each owned proxy carries its own deployments pointed at scripted upstream scenarios, so both workers know
them from boot. A 24-request burst (chat, text completions, Responses and Anthropic Messages, two of each
streamed at 500 ms per frame so they are still in flight when the outage lands) runs while the cache store
is stopped (C1), paused (C2) or while one worker has just been SIGKILLed (C3): every request answers 200
with content and lands exactly once in the spend log. After recovery an upstream answer with no choices
still reaches the upstream again on the next identical request. C4 runs the in-memory cache mode, where
each worker keeps its own store and the cached object is read back without a JSON round trip.
"""

from __future__ import annotations

import os
import threading
import uuid
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Final

import httpx
import psutil
import pytest
import yaml
from pydantic import JsonValue
from redis import Redis

from tests.integration._support.client import JSON_OBJECT, Gateway, Scenario, eventually, string_value
from tests.integration._support.database import read_rows
from tests.integration._support.process import owned_proxy_process
from tests.integration._support.redis_process import owned_redis
from tests.integration._support.upstream import ScenarioHandle
from tests.integration.cost_calculation.cost_tracking_case import JsonResponse, SseResponse
from tests.integration.caching.response_cache_case import (
    CHAT,
    CHAT_MODEL,
    CHAT_STREAM,
    MESSAGE,
    MESSAGE_STREAM,
    MESSAGES_MODEL,
    PROVIDER_KEY,
    RESPONSE,
    RESPONSE_STREAM,
    TEXT,
    TEXT_MODEL,
    TEXT_STREAM,
    Upstream,
    chat_body,
    choices,
    content_of_first_choice,
    emptied,
    json_response,
    message_body,
    prompt,
    rescript,
    response_id,
    scripted,
    slowed,
)

RecordProperty = Callable[[str, object], None]

WORKERS: Final = 2
REMOVE_FROM_ENVIRONMENT: Final = ("DATABASE_URL_READ_REPLICA",)
BURST_TIMEOUT_SECONDS: Final = 60
CHAOS_AFTER_ANSWERS: Final = 4
STREAM_FRAME_DELAY_MS: Final = 500
PER_ENDPOINT: Final = 6
STREAMS_PER_ENDPOINT: Final = 2
LOCAL_BATCH: Final = 20
SPEND_SQL: Final = 'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE api_key = %s'


@dataclass(frozen=True, slots=True)
class Deployment:
    name: str
    litellm_model: str
    handle: ScenarioHandle

    def entry(self) -> dict[str, JsonValue]:
        return {
            "model_name": self.name,
            "litellm_params": {
                "model": self.litellm_model,
                "api_base": self.handle.api_base(),
                "api_key": PROVIDER_KEY,
            },
        }


@dataclass(frozen=True, slots=True)
class Fleet:
    chat: Deployment
    chat_stream: Deployment
    text: Deployment
    text_stream: Deployment
    responses: Deployment
    responses_stream: Deployment
    messages: Deployment
    messages_stream: Deployment
    flip: Deployment

    def burst_members(self) -> tuple[Deployment, ...]:
        return (
            self.chat,
            self.chat_stream,
            self.text,
            self.text_stream,
            self.responses,
            self.responses_stream,
            self.messages,
            self.messages_stream,
        )

    def entries(self) -> list[dict[str, JsonValue]]:
        return [deployment.entry() for deployment in (*self.burst_members(), self.flip)]


@dataclass(frozen=True, slots=True)
class Call:
    label: str
    path: str
    body: dict[str, JsonValue]

    @property
    def streamed(self) -> bool:
        return self.label.startswith("stream-")


@dataclass(frozen=True, slots=True)
class Answer:
    call: Call
    status: int | None
    text: str

    @property
    def has_content(self) -> bool:
        marker: Final = "streamed" if self.call.streamed else "scripted"
        return self.status == 200 and marker in self.text

    def describe(self) -> str:
        return f"{self.call.label}: {self.status} {self.text[:160]!r}"


def _deployment(scenario: Scenario, kind: str, litellm_model: str, response: JsonResponse | SseResponse) -> Deployment:
    return Deployment(f"chaos-{kind}-{uuid.uuid4().hex[:8]}", litellm_model, scripted(scenario, response))


def _fleet(scenario: Scenario) -> Fleet:
    return Fleet(
        chat=_deployment(scenario, "chat", CHAT_MODEL, json_response(CHAT)),
        chat_stream=_deployment(scenario, "chat-stream", CHAT_MODEL, slowed(CHAT_STREAM, STREAM_FRAME_DELAY_MS)),
        text=_deployment(scenario, "text", TEXT_MODEL, json_response(TEXT)),
        text_stream=_deployment(scenario, "text-stream", TEXT_MODEL, slowed(TEXT_STREAM, STREAM_FRAME_DELAY_MS)),
        responses=_deployment(scenario, "responses", CHAT_MODEL, json_response(RESPONSE)),
        responses_stream=_deployment(
            scenario, "responses-stream", CHAT_MODEL, slowed(RESPONSE_STREAM, STREAM_FRAME_DELAY_MS)
        ),
        messages=_deployment(scenario, "messages", MESSAGES_MODEL, json_response(MESSAGE)),
        messages_stream=_deployment(
            scenario, "messages-stream", MESSAGES_MODEL, slowed(MESSAGE_STREAM, STREAM_FRAME_DELAY_MS)
        ),
        flip=_deployment(scenario, "flip", CHAT_MODEL, json_response(emptied(CHAT, "choices"))),
    )


def _config(directory: Path, fleet: Fleet, cache_params: dict[str, JsonValue] | None = None) -> Path:
    """The rig's proxy config with the fleet as its model list and, when given, another response cache."""
    rig: Final = yaml.safe_load((Path(__file__).resolve().parents[1] / "proxy_config.yaml").read_text())
    assert isinstance(rig, dict)
    litellm_settings: Final = rig["litellm_settings"]
    assert isinstance(litellm_settings, dict)
    config: Final = {
        **rig,
        "model_list": fleet.entries(),
        "litellm_settings": {**litellm_settings, **({"cache_params": cache_params} if cache_params else {})},
    }
    path: Final = directory / f"proxy_config-{uuid.uuid4().hex[:8]}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def _overrides() -> dict[str, str]:
    return {"DATABASE_URL": os.environ["DATABASE_URL"]}


def _burst_calls(fleet: Fleet, per_endpoint: int, streams_per_endpoint: int) -> tuple[Call, ...]:
    def calls() -> Iterator[Call]:
        for index in range(per_endpoint):
            streamed: Final = index < streams_per_endpoint
            prefix: Final = "stream" if streamed else "plain"
            extra: Final[dict[str, JsonValue]] = {"stream": True} if streamed else {}
            chat: Final = fleet.chat_stream if streamed else fleet.chat
            text: Final = fleet.text_stream if streamed else fleet.text
            responses: Final = fleet.responses_stream if streamed else fleet.responses
            messages: Final = fleet.messages_stream if streamed else fleet.messages
            yield Call(f"{prefix}-chat-{index}", "/v1/chat/completions", chat_body(chat.name, prompt(), **extra))
            yield Call(f"{prefix}-text-{index}", "/v1/completions", {"model": text.name, "prompt": prompt(), **extra})
            yield Call(
                f"{prefix}-responses-{index}", "/v1/responses", {"model": responses.name, "input": prompt(), **extra}
            )
            yield Call(f"{prefix}-messages-{index}", "/v1/messages", message_body(messages.name, prompt(), **extra))

    return tuple(calls())


class Burst:
    """Every call submitted at once against ``target``; ``chaos_point`` is set once ``CHAOS_AFTER_ANSWERS``
    have answered (or failed), while the slowed streams are still in flight."""

    def __init__(self, target: str, key: str) -> None:
        self._target: Final = target
        self._key: Final = key
        self._lock: Final = threading.Lock()
        self._answers = 0  # rebind-ok: counter behind _lock
        self._futures: dict[str, Future[Answer]] = {}
        self.chaos_point: Final = threading.Event()

    def start(self, pool: ThreadPoolExecutor, calls: Sequence[Call]) -> None:
        assert not self._futures, "burst already started"
        self._futures.update((call.label, pool.submit(self._send, call)) for call in calls)
        assert self.chaos_point.wait(BURST_TIMEOUT_SECONDS), (
            f"fewer than {CHAOS_AFTER_ANSWERS} requests answered within {BURST_TIMEOUT_SECONDS}s"
        )

    def _send(self, call: Call) -> Answer:
        try:
            with httpx.Client(base_url=self._target, timeout=BURST_TIMEOUT_SECONDS, trust_env=False) as client:
                response: Final = client.post(
                    call.path, json=call.body, headers={"Authorization": f"Bearer {self._key}"}
                )
            answer = Answer(call, response.status_code, response.text)
        except httpx.HTTPError as error:
            answer = Answer(call, None, f"{type(error).__name__}: {error}"[:200])
        with self._lock:
            self._answers += 1
            if self._answers >= CHAOS_AFTER_ANSWERS:
                self.chaos_point.set()
        return answer

    def answered(self) -> int:
        with self._lock:
            return self._answers

    def pending_streams(self) -> tuple[str, ...]:
        return tuple(
            label for label, future in self._futures.items() if label.startswith("stream-") and not future.done()
        )

    def outcomes(self) -> tuple[Answer, ...]:
        return tuple(future.result(timeout=BURST_TIMEOUT_SECONDS + 30) for future in self._futures.values())


def _assert_all_answered_with_content(outcomes: Sequence[Answer]) -> None:
    failures: Final = tuple(answer.describe() for answer in outcomes if not answer.has_content)
    assert not failures, (
        f"{len(failures)} of {len(outcomes)} burst requests lack a 200 with content:\n  " + "\n  ".join(failures)
    )


def _plain_ids(outcomes: Sequence[Answer]) -> frozenset[str]:
    """The ids of the non-streamed chat, text and Messages answers, which the spend log records verbatim."""
    return frozenset(
        string_value(JSON_OBJECT.validate_json(answer.text)["id"])
        for answer in outcomes
        if not answer.call.streamed and answer.call.path != "/v1/responses"
    )  # comprehension-ok: one filter over the burst


def _spend_request_ids(key: str) -> frozenset[str]:
    return frozenset(
        string_value(row["request_id"]) for row in read_rows(SPEND_SQL, (sha256(key.encode()).hexdigest(),))
    )


def _assert_every_burst_request_landed_once(key: str, outcomes: Sequence[Answer]) -> None:
    rows: Final = eventually(lambda: _spend_request_ids(key), lambda ids: len(ids) >= len(outcomes), seconds=70)
    assert len(rows) == len(outcomes), f"{len(rows)} spend rows for {len(outcomes)} burst requests: {sorted(rows)}"
    missing: Final = _plain_ids(outcomes) - rows
    assert not missing, f"burst answers without a spend row: {sorted(missing)}"


def _chat(target: str, key: str, body: dict[str, JsonValue]) -> httpx.Response:
    with httpx.Client(base_url=target, timeout=30, trust_env=False) as client:
        return client.post("/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {key}"})


def _assert_empty_answer_reaches_upstream_again(target: str, key: str, upstream: Upstream, flip: Deployment) -> None:
    """Through ``target`` after the outage: the scripted empty answer is never served again once the
    upstream answers for real, and the real answer is the one served from the cache afterwards."""
    request: Final = chat_body(flip.name, prompt())
    empty: Final = _chat(target, key, request)
    assert choices(empty) == [], empty.text
    assert upstream.calls(flip.handle.scenario_id) == 1
    rescript(flip.handle, json_response(CHAT))
    refilled: Final = _chat(target, key, request)
    assert content_of_first_choice(refilled) == "scripted", f"the empty answer was served: {refilled.text}"
    assert upstream.calls(flip.handle.scenario_id) == 2
    hit: Final = eventually(lambda: _chat(target, key, request), lambda r: response_id(r) == response_id(refilled))
    assert content_of_first_choice(hit) == "scripted", hit.text
    assert upstream.calls(flip.handle.scenario_id) >= 2


def _workers(root: psutil.Process) -> tuple[psutil.Process, ...]:
    """uvicorn's worker children of the owned proxy root, spawned through ``multiprocessing.spawn``."""

    def spawned() -> Iterator[psutil.Process]:
        for child in root.children():
            try:
                cmdline = child.cmdline()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            if any("multiprocessing.spawn" in part for part in cmdline):
                yield child

    return tuple(sorted(spawned(), key=lambda process: process.pid))


def _cache_ping(target: str, key: str) -> httpx.Response:
    with httpx.Client(base_url=target, timeout=30, trust_env=False) as client:
        return client.get("/cache/ping", headers={"Authorization": f"Bearer {key}"})


def _cache_status(response: httpx.Response) -> str:
    assert response.status_code == 200, f"/cache/ping: {response.status_code} {response.text}"
    return string_value(JSON_OBJECT.validate_json(response.content)["status"])


def _proxy_url(gateway: Gateway) -> str:
    return str(gateway.client.base_url).rstrip("/")


@pytest.mark.timeout(300)
def test_redis_stopped_mid_burst_answers_every_request_and_never_pins_an_empty_answer_after(
    gateway: Gateway, tmp_path: Path, record_property: RecordProperty
) -> None:
    upstream: Final = Upstream(gateway.upstream_url)
    with gateway.scenario() as scenario, owned_redis(tmp_path) as store:
        fleet: Final = _fleet(scenario)
        key: Final = scenario.key(models=[deployment.name for deployment in (*fleet.burst_members(), fleet.flip)])
        with (
            owned_proxy_process(
                gateway,
                tmp_path,
                {
                    **_overrides(),
                    "REDIS_HOST": store.host,
                    "REDIS_PORT": str(store.port),
                    "REDIS_CIRCUIT_BREAKER_RECOVERY_TIMEOUT": "0",
                },
                config=_config(tmp_path, fleet),
                remove_environment=REMOVE_FROM_ENVIRONMENT,
                workers=WORKERS,
            ) as owned,
            ThreadPoolExecutor(PER_ENDPOINT * 4) as pool,
        ):
            target: Final = _proxy_url(owned.gateway)
            assert _cache_status(_cache_ping(target, gateway.key)) == "healthy"
            burst: Final = Burst(target, key)
            burst.start(pool, _burst_calls(fleet, PER_ENDPOINT, STREAMS_PER_ENDPOINT))
            pending: Final = burst.pending_streams()
            record_property("answered_before_outage", burst.answered())
            assert pending, "no stream was in flight when Redis stopped"
            store.stop()
            down: Final = _cache_ping(target, gateway.key)
            assert down.status_code == 503, f"/cache/ping with Redis stopped: {down.status_code} {down.text}"
            assert "Service Unhealthy" in down.text, down.text
            outcomes: Final = burst.outcomes()
            store.start()
            recovered: Final = eventually(
                lambda: _cache_ping(target, gateway.key), lambda r: r.status_code == 200, seconds=60
            )
            assert _cache_status(recovered) == "healthy"
            _assert_all_answered_with_content(outcomes)
            _assert_every_burst_request_landed_once(key, outcomes)
            _assert_empty_answer_reaches_upstream_again(target, key, upstream, fleet.flip)


@pytest.mark.timeout(300)
def test_redis_paused_mid_burst_answers_every_request_and_never_pins_an_empty_answer_after(
    gateway: Gateway, tmp_path: Path, record_property: RecordProperty
) -> None:
    upstream: Final = Upstream(gateway.upstream_url)
    with gateway.scenario() as scenario, owned_redis(tmp_path) as store:
        fleet: Final = _fleet(scenario)
        key: Final = scenario.key(models=[deployment.name for deployment in (*fleet.burst_members(), fleet.flip)])
        with (
            owned_proxy_process(
                gateway,
                tmp_path,
                {**_overrides(), "REDIS_HOST": store.host, "REDIS_PORT": str(store.port)},
                config=_config(tmp_path, fleet),
                remove_environment=REMOVE_FROM_ENVIRONMENT,
                workers=WORKERS,
            ) as owned,
            ThreadPoolExecutor(PER_ENDPOINT * 4) as pool,
            Redis(host=store.host, port=store.port) as control,
        ):
            target: Final = _proxy_url(owned.gateway)
            assert _cache_status(_cache_ping(target, gateway.key)) == "healthy"
            burst: Final = Burst(target, key)
            burst.start(pool, _burst_calls(fleet, PER_ENDPOINT, STREAMS_PER_ENDPOINT))
            pending: Final = burst.pending_streams()
            record_property("answered_before_pause", burst.answered())
            assert pending, "no stream was in flight when Redis was paused"
            assert control.client_pause(2000) is True
            outcomes: Final = burst.outcomes()
            assert _cache_status(_cache_ping(target, gateway.key)) == "healthy"
            _assert_all_answered_with_content(outcomes)
            _assert_every_burst_request_landed_once(key, outcomes)
            _assert_empty_answer_reaches_upstream_again(target, key, upstream, fleet.flip)


@pytest.mark.timeout(300)
def test_worker_killed_before_burst_leaves_the_survivor_serving_and_never_pins_an_empty_answer(
    gateway: Gateway, tmp_path: Path, record_property: RecordProperty
) -> None:
    upstream: Final = Upstream(gateway.upstream_url)
    with gateway.scenario() as scenario:
        fleet: Final = _fleet(scenario)
        key: Final = scenario.key(models=[deployment.name for deployment in (*fleet.burst_members(), fleet.flip)])
        with (
            owned_proxy_process(
                gateway,
                tmp_path,
                _overrides(),
                config=_config(tmp_path, fleet),
                remove_environment=REMOVE_FROM_ENVIRONMENT,
                workers=WORKERS,
            ) as owned,
            ThreadPoolExecutor(PER_ENDPOINT * 4) as pool,
        ):
            target: Final = _proxy_url(owned.gateway)
            root: Final = psutil.Process(owned.process.pid)
            before: Final = _workers(root)
            assert len(before) == WORKERS, [process.pid for process in before]
            victim: Final = before[0]
            victim.kill()
            victim.wait(timeout=10)
            with httpx.Client(base_url=target, timeout=15, trust_env=False) as fresh:
                readiness: Final = fresh.get("/health/readiness")
            assert readiness.status_code == 200, f"/health/readiness with worker {victim.pid} dead: {readiness.text}"
            burst: Final = Burst(target, key)
            burst.start(pool, _burst_calls(fleet, PER_ENDPOINT // 2, STREAMS_PER_ENDPOINT // 2))
            outcomes: Final = burst.outcomes()
            respawned: Final = eventually(
                lambda: tuple(process.pid for process in _workers(root)),
                lambda pids: len(pids) == WORKERS and victim.pid not in pids,
                seconds=60,
            )
            record_property(
                "worker_pids", {"before": [process.pid for process in before], "killed": victim.pid, "after": respawned}
            )
            _assert_all_answered_with_content(outcomes)
            _assert_every_burst_request_landed_once(key, outcomes)
            _assert_empty_answer_reaches_upstream_again(target, key, upstream, fleet.flip)


def _batch(target: str, key: str, body: dict[str, JsonValue], size: int) -> tuple[httpx.Response, ...]:
    """``size`` identical requests, each on a fresh connection so both workers take traffic."""
    return tuple(_chat(target, key, body) for _ in range(size))


@pytest.mark.timeout(300)
def test_in_memory_cache_mode_never_serves_an_empty_answer_from_a_worker_store(
    gateway: Gateway, tmp_path: Path
) -> None:
    upstream: Final = Upstream(gateway.upstream_url)
    with gateway.scenario() as scenario:
        fleet: Final = _fleet(scenario)
        key: Final = scenario.key(models=[fleet.flip.name])
        with owned_proxy_process(
            gateway,
            tmp_path,
            _overrides(),
            config=_config(tmp_path, fleet, cache_params={"type": "local"}),
            remove_environment=REMOVE_FROM_ENVIRONMENT,
            workers=WORKERS,
        ) as owned:
            target: Final = _proxy_url(owned.gateway)
            request: Final = chat_body(fleet.flip.name, prompt())
            first_batch: Final = _batch(target, key, request, LOCAL_BATCH)
            assert all(choices(response) == [] for response in first_batch), [r.text for r in first_batch]
            assert upstream.calls(fleet.flip.handle.scenario_id) == LOCAL_BATCH, (
                "an empty answer was served from a worker store"
            )
            rescript(fleet.flip.handle, json_response(CHAT))
            second_batch: Final = _batch(target, key, request, LOCAL_BATCH)
            served_empty: Final = tuple(response.text for response in second_batch if choices(response) == [])
            assert not served_empty, (
                f"{len(served_empty)} of {LOCAL_BATCH} answers still empty after the upstream recovered"
            )
            assert all(content_of_first_choice(response) == "scripted" for response in second_batch)
            after_second: Final = upstream.calls(fleet.flip.handle.scenario_id)
            assert LOCAL_BATCH < after_second <= 2 * LOCAL_BATCH, after_second
            third_batch: Final = _batch(target, key, request, LOCAL_BATCH)
            assert all(content_of_first_choice(response) == "scripted" for response in third_batch)
            assert upstream.calls(fleet.flip.handle.scenario_id) - after_second <= WORKERS, (
                "the real answer is not cached"
            )
