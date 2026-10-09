from __future__ import annotations

import json
import signal
import subprocess
import uuid
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from itertools import product
from pathlib import Path
from typing import Final, Literal, TypeAlias

import httpx
import psutil
import pytest
from anthropic import Anthropic
from anthropic import APIError as AnthropicApiError
from integration._support.client import JSON_OBJECT, Gateway, Scenario, eventually, object_value
from integration._support.database import read_rows
from integration._support.process import graceful_stop_seconds, owned_proxy_process, owned_upstream
from integration._support.upstream import ScenarioHandle, delete_scenario, register_scenario
from openai import APIError as OpenAiApiError
from openai import OpenAI
from pydantic import JsonValue

from tests.integration.cost_calculation.cost_tracking_case import JsonResponse, RoutedResponse, SseResponse

_Kind: TypeAlias = Literal["chat", "chat_stream", "messages", "responses", "completions", "moderations"]

_ALL_KINDS: Final[tuple[_Kind, ...]] = ("chat", "chat_stream", "messages", "responses", "completions", "moderations")
_CHAT_KINDS: Final[tuple[_Kind, ...]] = ("chat", "chat_stream")
_NO_CACHE: Final = {"cache": {"no-cache": True}}
_SPEND_QUERY: Final = 'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id = %s'
_CHAT: Final = JsonResponse(
    content_type="application/json",
    body={
        "id": "chatcmpl-$UNIQUE_ID",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "scripted"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    },
)
_MODERATION: Final = JsonResponse(
    content_type="application/json",
    body={
        "id": "modr-$UNIQUE_ID",
        "model": "omni-moderation-latest",
        "results": [{"flagged": False, "categories": {}, "category_scores": {}}],
    },
)
_RESPONSES: Final = JsonResponse(
    content_type="application/json",
    body={
        "id": "resp_$UNIQUE_ID",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-4o-mini",
        "output": [
            {
                "type": "message",
                "id": "msg_$UNIQUE_ID",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "scripted", "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    },
)
_PLAIN: Final = RoutedResponse(
    content_type="application/x-routed",
    routes={"POST /chat/completions": _CHAT, "POST /moderations": _MODERATION, "POST /responses": _RESPONSES},
)
_STREAM: Final = SseResponse(
    content_type="text/event-stream",
    frames=(
        'data: {"id":"chatcmpl-$UNIQUE_ID","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini",'
        '"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}',
        'data: {"id":"chatcmpl-$UNIQUE_ID","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini",'
        '"choices":[{"index":0,"delta":{"content":"streamed"},"finish_reason":null}]}',
        'data: {"id":"chatcmpl-$UNIQUE_ID","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini",'
        '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
        "data: [DONE]",
    ),
    frame_delay_ms=150,
)


@dataclass(frozen=True, slots=True)
class _Call:
    kind: _Kind
    marker: str


@dataclass(frozen=True, slots=True)
class _Completed:
    call: _Call
    response_id: str


@dataclass(frozen=True, slots=True)
class _Failed:
    call: _Call
    error: str


@dataclass(frozen=True, slots=True)
class _Clients:
    openai: OpenAI
    anthropic: Anthropic
    plain_model: str
    stream_model: str


class _Observations:
    def __init__(self, url: str) -> None:
        self.url = url.rstrip("/")
        self.items: tuple[Mapping[str, JsonValue], ...] = ()

    def read(self) -> tuple[Mapping[str, JsonValue], ...]:
        with httpx.Client(timeout=10, trust_env=False) as client:
            payload: Final = JSON_OBJECT.validate_python(client.get(f"{self.url}/__observations").json())
        requests: Final = payload.get("requests")
        assert isinstance(requests, list)
        self.items = (*self.items, *(object_value(item) for item in requests if isinstance(item, dict)))
        return self.items


def _register(
    scenario: Scenario, upstream_url: str, label: str, response: RoutedResponse | SseResponse
) -> ScenarioHandle:
    handle: Final = register_scenario(f"chaos-{label}-{uuid.uuid4().hex}", response, control_url=upstream_url)
    scenario.cleanups.callback(delete_scenario, handle)
    return handle


def _deployments(scenario: Scenario, upstream_url: str) -> tuple[str, str]:
    plain_handle: Final = _register(scenario, upstream_url, "plain", _PLAIN)
    stream_handle: Final = _register(scenario, upstream_url, "stream", _STREAM)
    return (
        scenario.model(api_base=plain_handle.api_base(), api_key=plain_handle.scenario_id),
        scenario.model(api_base=stream_handle.api_base(), api_key=stream_handle.scenario_id),
    )


def _clients(gateway: Gateway, deployments: tuple[str, str]) -> _Clients:
    base: Final = str(gateway.client.base_url).rstrip("/")
    clients: Final = _Clients(
        openai=OpenAI(base_url=f"{base}/v1", api_key=gateway.key, max_retries=0),
        anthropic=Anthropic(base_url=base, api_key=gateway.key, max_retries=0),
        plain_model=deployments[0],
        stream_model=deployments[1],
    )
    _import_lazy_sdk_resources_on_this_thread(clients)
    return clients


def _import_lazy_sdk_resources_on_this_thread(clients: _Clients) -> None:
    resources: Final = (
        clients.openai.chat.completions,
        clients.openai.completions,
        clients.openai.responses,
        clients.openai.moderations,
        clients.anthropic.messages,
    )
    assert all(resource is not None for resource in resources)


def _invoke(clients: _Clients, call: _Call) -> str:
    match call.kind:
        case "chat":
            return clients.openai.chat.completions.create(
                model=clients.plain_model, messages=[{"role": "user", "content": call.marker}], extra_body=_NO_CACHE
            ).id
        case "chat_stream":
            chunks: Final = tuple(
                clients.openai.chat.completions.create(
                    model=clients.stream_model,
                    messages=[{"role": "user", "content": call.marker}],
                    stream=True,
                    extra_body=_NO_CACHE,
                )
            )
            assert chunks[-1].choices[0].finish_reason == "stop", chunks
            assert len({chunk.id for chunk in chunks}) == 1, chunks
            return chunks[0].id
        case "messages":
            return clients.anthropic.messages.create(
                model=clients.plain_model,
                max_tokens=64,
                messages=[{"role": "user", "content": call.marker}],
                extra_body=_NO_CACHE,
            ).id
        case "responses":
            return clients.openai.responses.create(
                model=clients.plain_model, input=call.marker, extra_body=_NO_CACHE
            ).id
        case "completions":
            return clients.openai.completions.create(
                model=clients.plain_model, prompt=call.marker, extra_body=_NO_CACHE
            ).id
        case "moderations":
            return clients.openai.moderations.create(
                model=clients.plain_model, input=call.marker, extra_body=_NO_CACHE
            ).id


def _attempt(clients: _Clients, call: _Call) -> _Completed | _Failed:
    try:
        return _Completed(call, _invoke(clients, call))
    except (OpenAiApiError, AnthropicApiError, httpx.HTTPError) as error:
        return _Failed(call, f"{type(error).__name__}: {str(error)[:200]}")


def _every_worker_serves_every_kind(gateway: Gateway, deployments: tuple[str, str]) -> bool:
    calls: Final = _burst_calls(_ALL_KINDS, 2)
    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        futures: Final = _submit(pool, _clients(gateway, deployments), calls)
        outcomes: Final = tuple(future.result(timeout=60) for future in futures)
    return all(isinstance(outcome, _Completed) for outcome in outcomes)


def _burst_calls(kinds: tuple[_Kind, ...], per_kind: int) -> tuple[_Call, ...]:
    return tuple(_Call(kind, f"burst-{kind}-{uuid.uuid4().hex}") for kind, _ in product(kinds, range(per_kind)))


def _submit(
    pool: ThreadPoolExecutor, clients: _Clients, calls: tuple[_Call, ...]
) -> tuple[Future[_Completed | _Failed], ...]:
    return tuple(pool.submit(_attempt, clients, call) for call in calls)


def _done_count(futures: tuple[Future[_Completed | _Failed], ...]) -> int:
    return sum(future.done() for future in futures)


def _liveliness(gateway: Gateway) -> int:
    return gateway.client.get("/health/liveliness").status_code


def _spend_rows(request_id: str) -> tuple[Mapping[str, JsonValue], ...]:
    return tuple(read_rows(_SPEND_QUERY, (request_id,)))


def _marker_counts(items: tuple[Mapping[str, JsonValue], ...], markers: tuple[str, ...]) -> tuple[int, ...]:
    bodies: Final = tuple(json.dumps(item.get("body")) for item in items)
    return tuple(sum(marker in body for body in bodies) for marker in markers)


def _landed_once(response_ids: tuple[str, ...]) -> tuple[tuple[Mapping[str, JsonValue], ...], ...]:
    rows: Final = tuple(
        eventually(partial(_spend_rows, response_id), lambda values: len(values) == 1, seconds=90)
        for response_id in response_ids
    )
    assert all(found[0]["request_id"] == response_id for found, response_id in zip(rows, response_ids)), rows
    return rows


def _is_worker(child: psutil.Process) -> bool:
    try:
        return "spawn_main" in " ".join(child.cmdline())
    except psutil.Error:
        return False


def _worker_alive(worker: psutil.Process) -> bool:
    try:
        return worker.is_running() and worker.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _alive_workers(process: subprocess.Popen[bytes]) -> tuple[psutil.Process, ...]:
    children: Final = tuple(child for child in psutil.Process(process.pid).children() if _is_worker(child))
    return tuple(child for child in children if _worker_alive(child))


def _process_tree(process: subprocess.Popen[bytes]) -> str:
    root: Final = psutil.Process(process.pid)
    return "\n".join(_process_line(member) for member in (root, *root.children(recursive=True)))


def _process_line(member: psutil.Process) -> str:
    try:
        return f"{member.pid} {' '.join(member.cmdline())}"
    except psutil.Error:
        return f"{member.pid} <exited>"


def _kind_counts(outcomes: tuple[_Completed | _Failed, ...]) -> str:
    kinds: Final = tuple(outcome.call.kind for outcome in outcomes)
    return str({kind: kinds.count(kind) for kind in _ALL_KINDS if kind in kinds})


def test_c01_upstream_pause_mid_burst_keeps_liveliness_and_lands_every_id_once(
    gateway: Gateway,
    tmp_path: Path,
    record_property: pytest.RecordProperty,
) -> None:
    with owned_upstream(tmp_path) as slot, gateway.scenario() as scenario:
        upstream: Final = slot.process
        assert upstream is not None
        deployments: Final = _deployments(scenario, slot.url)
        eventually(partial(_every_worker_serves_every_kind, gateway, deployments), lambda served: served, seconds=90)
        clients: Final = _clients(gateway, deployments)
        calls: Final = _burst_calls(_ALL_KINDS, 5)
        observations: Final = _Observations(slot.url)
        with ThreadPoolExecutor(max_workers=len(calls)) as pool:
            futures: Final = _submit(pool, clients, calls)
            eventually(partial(_done_count, futures), lambda done: done >= 1, seconds=60)
            upstream.send_signal(signal.SIGSTOP)
            try:
                done_at_pause: Final = _done_count(futures)
                paused_liveliness: Final = tuple(_liveliness(gateway) for _ in range(3))
            finally:
                upstream.send_signal(signal.SIGCONT)
            done_at_resume: Final = _done_count(futures)
            outcomes: Final = tuple(future.result(timeout=180) for future in futures)
        record_property("c01_burst_size", len(calls))
        record_property("c01_done_at_pause", done_at_pause)
        record_property("c01_done_at_resume", done_at_resume)
        record_property("c01_paused_liveliness_statuses", str(paused_liveliness))
        assert paused_liveliness == (200, 200, 200), paused_liveliness
        assert done_at_resume < len(calls), done_at_resume
        failures: Final = tuple(outcome for outcome in outcomes if isinstance(outcome, _Failed))
        assert not failures, failures
        completed: Final = tuple(outcome for outcome in outcomes if isinstance(outcome, _Completed))
        record_property("c01_completed_by_kind", _kind_counts(completed))
        response_ids: Final = tuple(outcome.response_id for outcome in completed)
        assert len(set(response_ids)) == len(calls), response_ids
        spend_rows: Final = _landed_once(response_ids)
        record_property("c01_spend_query", _SPEND_QUERY)
        record_property("c01_spend_row_counts", str(tuple(len(rows) for rows in spend_rows)))
        markers: Final = tuple(call.marker for call in calls)
        eventually(
            observations.read,
            lambda items: _marker_counts(items, markers) == (1,) * len(markers),
            seconds=60,
        )
        record_property("c01_upstream_marker_counts", str(_marker_counts(observations.items, markers)))


@pytest.mark.timeout(2 * graceful_stop_seconds() + 120)
def test_c02_worker_sigkill_mid_burst_keeps_serving_and_respawns(
    gateway: Gateway,
    tmp_path: Path,
    record_property: pytest.RecordProperty,
) -> None:
    with owned_upstream(tmp_path) as slot, gateway.scenario() as scenario:
        deployments: Final = _deployments(scenario, slot.url)
        with owned_proxy_process(gateway, tmp_path, {"INTEGRATION_UPSTREAM_URL": slot.url}, workers=2) as owned:
            eventually(
                partial(_every_worker_serves_every_kind, owned.gateway, deployments), lambda served: served, seconds=90
            )
            clients: Final = _clients(owned.gateway, deployments)
            workers: Final = eventually(
                partial(_alive_workers, owned.process), lambda found: len(found) == 2, seconds=60
            )
            record_property("c02_process_tree_before_kill", _process_tree(owned.process))
            victim: Final = workers[0]
            calls: Final = _burst_calls(_CHAT_KINDS, 15)
            observations: Final = _Observations(slot.url)
            with ThreadPoolExecutor(max_workers=len(calls)) as pool:
                futures: Final = _submit(pool, clients, calls)
                eventually(partial(_done_count, futures), lambda done: done >= 1, seconds=60)
                victim.kill()
                eventually(partial(_worker_alive, victim), lambda alive: not alive, seconds=10)
                done_at_kill: Final = _done_count(futures)
                survivor: Final = eventually(
                    lambda: _attempt(clients, _Call("chat", f"after-kill-{uuid.uuid4().hex}")),
                    lambda outcome: isinstance(outcome, _Completed),
                    seconds=30,
                )
                outcomes: Final = tuple(future.result(timeout=180) for future in futures)
            record_property("c02_burst_size", len(calls))
            record_property("c02_killed_worker_pid", victim.pid)
            record_property("c02_done_at_kill", done_at_kill)
            assert isinstance(survivor, _Completed), survivor
            completed: Final = tuple(outcome for outcome in outcomes if isinstance(outcome, _Completed))
            failed: Final = tuple(outcome for outcome in outcomes if isinstance(outcome, _Failed))
            record_property("c02_completed_by_kind", _kind_counts(completed))
            record_property("c02_inflight_failure_count", len(failed))
            record_property("c02_inflight_failure_errors", str(tuple(outcome.error for outcome in failed)))
            assert completed, outcomes
            respawned: Final = eventually(
                partial(_alive_workers, owned.process),
                lambda found: len(found) == 2 and victim.pid not in {worker.pid for worker in found},
                seconds=120,
            )
            record_property("c02_process_tree_after_respawn", _process_tree(owned.process))
            record_property("c02_respawned_worker_pids", str(tuple(worker.pid for worker in respawned)))
            after_respawn: Final = eventually(
                lambda: _attempt(clients, _Call("chat", f"after-respawn-{uuid.uuid4().hex}")),
                lambda outcome: isinstance(outcome, _Completed),
                seconds=60,
            )
            assert isinstance(after_respawn, _Completed), after_respawn
            _landed_once((survivor.response_id, after_respawn.response_id))
            response_ids: Final = tuple(outcome.response_id for outcome in completed)
            assert len(set(response_ids)) == len(completed), response_ids
            landed: Final = tuple(outcome for outcome in completed if len(_spend_rows(outcome.response_id)) == 1)
            lost: Final = tuple(outcome for outcome in completed if len(_spend_rows(outcome.response_id)) == 0)
            assert len(landed) + len(lost) == len(completed), (landed, lost)
            record_property("c02_spend_query", _SPEND_QUERY)
            record_property("c02_burst_ids_landed_once", len(landed))
            record_property("c02_burst_ids_lost", len(lost))
            record_property("c02_burst_ids_lost_by_kind", _kind_counts(lost))
            completed_markers: Final = tuple(outcome.call.marker for outcome in completed)
            eventually(
                observations.read,
                lambda items: _marker_counts(items, completed_markers) == (1,) * len(completed_markers),
                seconds=60,
            )
            failed_markers: Final = tuple(outcome.call.marker for outcome in failed)
            record_property(
                "c02_failed_calls_seen_upstream",
                sum(count == 1 for count in _marker_counts(observations.items, failed_markers)),
            )
