"""A team's Arize sampling rate decides once per trace, so every pod and worker serving a call of it agrees.

Two proxies share one database, as two pods behind a load balancer do: the first runs one worker, the second
two. Calls of one trace carry the caller's ``traceparent`` and land on different pods and workers; the team's
Arize space must receive every call of a trace or none of them.
"""

import asyncio
import json
import re
import uuid
from collections.abc import Callable, Generator, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, replace
from itertools import chain, product
from pathlib import Path
from typing import Final, Literal, TypeAlias
from urllib.parse import urlparse

import httpx
import pytest
import yaml
from integration._support.client import (
    Scenario,
    eventually,
    gateway_from_environment,
    list_value,
    object_value,
    until_asserted,
)
from integration._support.database import scratch_database
from integration._support.otlp_sink import Span, SpanSinks, configure_sink, recorded_requests, recorded_spans
from integration._support.process import OwnedProxy, WorkerHealthcheck, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

MODEL: Final = "team-sampling"
SERVER: Final = 2
CALL_ID: Final = "litellm.call_id"
OPERATION: Final = "gen_ai.operation.name"
SPACE: Final = "space-sampled"
CREDENTIALS: Final = {"arize_space_key": SPACE, "arize_api_key": "arize-sampled"}
FAILING: Final = re.compile(rb"sampling-fail")
JSON: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
TRACES: Final = 16
Endpoint: TypeAlias = Literal["chat", "responses", "messages"]
Verdicts: TypeAlias = Mapping[str, tuple[bool, ...]]


def _chat_reply(stream: bool) -> Reply:
    usage: Final[JsonValue] = {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}
    if not stream:
        return Reply(
            body=json.dumps(
                {
                    "id": "chatcmpl-sampling",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
                    ],
                    "usage": usage,
                }
            ).encode()
        )
    chunk: Final = {"id": "chatcmpl-sampling", "object": "chat.completion.chunk", "created": 1, "model": "gpt-4o-mini"}
    deltas: Final[tuple[JsonValue, ...]] = (
        {**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}}]},
        {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {**chunk, "choices": [], "usage": usage},
    )
    return Reply(
        content_type="text/event-stream",
        chunks=(*(b"data: " + json.dumps(delta).encode() + b"\n\n" for delta in deltas), b"data: [DONE]\n\n"),
    )


def _responses_reply(stream: bool) -> Reply:
    response: Final[Mapping[str, JsonValue]] = {
        "id": "resp_sampling",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-4o-mini",
        "output": [
            {
                "id": "msg_sampling",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "ok", "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 7, "output_tokens": 2, "total_tokens": 9},
    }
    if not stream:
        return Reply(body=json.dumps(response).encode())
    events: Final[tuple[Mapping[str, JsonValue], ...]] = (
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": {**response, "status": "in_progress", "output": []},
        },
        {
            "type": "response.output_text.delta",
            "sequence_number": 1,
            "item_id": "msg_sampling",
            "output_index": 0,
            "content_index": 0,
            "delta": "ok",
        },
        {"type": "response.completed", "sequence_number": 2, "response": dict(response)},
    )
    return Reply(
        content_type="text/event-stream",
        chunks=tuple(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events),
    )


def _upstream(request: Request) -> Reply:
    if FAILING.search(request.body) is not None:
        return Reply(status=500, body=b'{"error":{"message":"scripted upstream failure","type":"server_error"}}')
    stream: Final = object_value(JSON.validate_json(request.body)).get("stream") is True
    if request.target.endswith("/responses"):
        return _responses_reply(stream)
    return _chat_reply(stream)


@dataclass(frozen=True, slots=True)
class Sent:
    trace_id: str
    parent_span_id: str
    call_id: str


@dataclass(frozen=True, slots=True)
class Call:
    path: str
    body: Mapping[str, JsonValue]
    headers: Mapping[str, str]
    sent: Sent
    expected: int


def _call(
    key: str, trace_id: str | None, endpoint: Endpoint = "chat", stream: bool = False, content: str = "hello"
) -> Call:
    parent: Final = uuid.uuid4().hex[:16]
    call_id: Final = str(uuid.uuid4())
    messages: Final[JsonValue] = [{"role": "user", "content": content}]
    path, body = {
        "chat": ("/v1/chat/completions", {"model": MODEL, "messages": messages, "stream": stream}),
        "responses": ("/v1/responses", {"model": MODEL, "input": content, "stream": stream}),
        "messages": ("/v1/messages", {"model": MODEL, "max_tokens": 16, "messages": messages, "stream": stream}),
    }[endpoint]
    headers: Final = {
        "Authorization": f"Bearer {key}",
        "x-litellm-call-id": call_id,
        "connection": "close",
        **({} if trace_id is None else {"traceparent": f"00-{trace_id}-{parent}-01"}),
    }
    expected: Final = 500 if FAILING.search(content.encode()) else 200
    return Call(path, body, headers, Sent(trace_id or "", parent if trace_id else "", call_id), expected)


@dataclass(frozen=True, slots=True)
class Pods:
    """``first`` serves with one worker, ``second`` with two."""

    first: OwnedProxy
    second: OwnedProxy
    scenario: Scenario
    sinks: SpanSinks

    def team_key(self, variables: Mapping[str, str]) -> str:
        team: Final = self.scenario.team()
        attached: Final = self.first.gateway.request(
            "POST", f"/team/{team}/callback", {"callback_name": "arize", "callback_vars": dict(variables)}
        )
        assert attached.status_code == 200, attached.text
        return self.scenario.key(team_id=team)

    def send(self, pod: OwnedProxy, call: Call) -> Sent:
        with pod.gateway.client.stream("POST", call.path, json=call.body, headers=call.headers) as response:
            text: Final = response.read().decode()
        assert response.status_code == call.expected, text
        return call.sent

    async def send_async(self, client: httpx.AsyncClient, call: Call) -> Sent:
        async with client.stream("POST", call.path, json=call.body, headers=call.headers) as response:
            text: Final = (await response.aread()).decode()
        assert response.status_code == call.expected, text
        return call.sent

    def on_each_worker(self, pod: OwnedProxy, call: Callable[[], Call]) -> tuple[Sent, ...]:
        """One call served by each worker of ``pod``, the others suspended while it runs."""
        workers: Final = sorted(pod.worker_pids())
        assert len(workers) == pod.gateway.workers, workers
        sent: Final[list[Sent]] = []  # mutable-ok: one call per worker, in worker order
        for serving in workers:
            with _suspended_all_but(pod, workers, serving):
                sent.append(self.send(pod, call()))
        return tuple(sent)

    def fence(self, key: str) -> tuple[Sent, ...]:
        """Calls of an unsampled team on the sampled team's exporter, one per worker of both pods.

        Each worker sends spans to that exporter in the order they were handed to it, so all of these
        arriving means every call any worker decided before them has been sent too.
        """
        return (self.send(self.first, _call(key, uuid.uuid4().hex)),) + self.on_each_worker(
            self.second, lambda: _call(key, uuid.uuid4().hex)
        )


@contextmanager
def _suspended_all_but(pod: OwnedProxy, workers: Sequence[int], serving: int) -> Generator[None]:
    if len(workers) == 1:
        yield
        return
    others: Final = [pid for pid in workers if pid != serving]
    with pod.suspended(others[0]):
        yield


def _space_id(request: Mapping[str, JsonValue]) -> JsonValue:
    return {name.lower(): value for name, value in object_value(request.get("headers") or {}).items()}.get(
        "arize-space-id"
    )


def _space_span_ids(url: str) -> frozenset[str]:
    """Span ids the Arize sink received in export requests sent with the team's space id."""
    ours: Final = (
        list_value(request.get("span_ids") or []) for request in recorded_requests(url) if _space_id(request) == SPACE
    )
    return frozenset(span_id for span_id in chain.from_iterable(ours) if isinstance(span_id, str))


def _team_spans(pods: Pods, since: int) -> tuple[Span, ...]:
    owned: Final = _space_span_ids(pods.sinks.arize)
    return tuple(span for span in recorded_spans(pods.sinks.arize, since)[1] if span["span_id"] in owned)


def _delivered(spans: tuple[Span, ...], sent: Sequence[Sent]) -> frozenset[str]:
    """Call ids of ``sent`` whose server span is among ``spans``."""
    by_parent: Final = {(call.trace_id, call.parent_span_id): call.call_id for call in sent}
    return frozenset(
        by_parent[(span["trace_id"], span["parent_span_id"])]
        for span in spans
        if span["kind"] == SERVER and (span["trace_id"], span["parent_span_id"]) in by_parent
    )


def _await_logged(pods: Pods, since: int, sent: Sequence[Sent]) -> None:
    """Wait until every call of ``sent`` has its model span in the operator's unsampled copy."""
    expected: Final = {call.call_id for call in sent}
    eventually(
        lambda: (
            {
                str(span["attributes"][CALL_ID])
                for span in recorded_spans(pods.sinks.operator, since)[1]
                if OPERATION in span["attributes"] and CALL_ID in span["attributes"]
            }
            & expected
        ),
        lambda logged: logged == expected,
        seconds=60,
    )


@dataclass(frozen=True, slots=True)
class Cursors:
    operator: int
    arize: int


def _cursors(pods: Pods) -> Cursors:
    return Cursors(recorded_spans(pods.sinks.operator)[0], recorded_spans(pods.sinks.arize)[0])


def _await_flushed(pods: Pods, cursors: Cursors, fence_key: str) -> None:
    """Wait until every worker of both pods has sent the team's exporter everything it decided so far."""
    fences: Final = pods.fence(fence_key)

    def arrived() -> None:
        delivered: Final = _delivered(_team_spans(pods, cursors.arize), fences)
        assert {fence.call_id for fence in fences} <= delivered, "the unsampled team's calls have not arrived yet"

    until_asserted(arrived, seconds=30)


def _settled(
    pods: Pods,
    cursors: Cursors,
    by_trace: Mapping[str, Sequence[Sent]],
    fence_key: str,
    holds: Callable[[tuple[bool, ...]], bool] = lambda kept: len(set(kept)) == 1,
) -> Verdicts:
    """Per trace, which of its calls reached the team's space, once every worker has sent what it decided
    and ``holds`` is true of every trace."""
    calls: Final = tuple(chain.from_iterable(by_trace.values()))
    _await_logged(pods, cursors.operator, calls)
    _await_flushed(pods, cursors, fence_key)

    def check() -> Verdicts:
        delivered: Final = _delivered(_team_spans(pods, cursors.arize), calls)
        verdicts: Final = {
            trace_id: tuple(call.call_id in delivered for call in sent) for trace_id, sent in by_trace.items()
        }
        broken: Final = {trace_id: kept for trace_id, kept in verdicts.items() if not holds(kept)}
        assert broken == {}, f"{len(broken)} of {len(verdicts)} traces were split (per call, kept): {broken}"
        return verdicts

    return until_asserted(check, seconds=30)


def _trace_ids(cell: str, count: int = TRACES) -> tuple[str, ...]:
    """Fixed RFC 4122 shaped trace ids, like the uuid4 hex a caller mints for ``traceparent``."""
    return tuple(uuid.uuid5(uuid.NAMESPACE_URL, f"team-sampling/{cell}/{index}").hex for index in range(count))


def _kept(verdicts: Verdicts) -> int:
    return sum(1 for kept in verdicts.values() if all(kept))


@pytest.mark.parametrize(
    ("variables", "content"),
    [({"arize_success_sampling_rate": "0.5"}, "hello"), ({"arize_error_sampling_rate": "0.5"}, "sampling-fail")],
    ids=["success-rate", "error-rate"],
)
def test_each_trace_reaches_the_team_space_whole_or_not_at_all(
    pods: Pods, variables: Mapping[str, str], content: str
) -> None:
    """The ticket's case: three calls per trace load balanced over the pods."""
    sampled: Final = pods.team_key({**CREDENTIALS, **variables})
    cursors: Final = _cursors(pods)
    by_trace: Final = {
        trace_id: tuple(
            pods.send(pod, _call(sampled, trace_id, content=content)) for pod in (pods.first, pods.second, pods.first)
        )
        for trace_id in _trace_ids("-".join(sorted(variables)))
    }

    verdicts: Final = _settled(pods, cursors, by_trace, pods.team_key(CREDENTIALS))

    assert 0 < _kept(verdicts) < TRACES, f"a rate of 0.5 kept {_kept(verdicts)} of {TRACES} traces"


@pytest.mark.parametrize("client", ["sync", "async"])
def test_every_endpoint_follows_the_verdict_of_its_trace(pods: Pods, client: str) -> None:
    sampled: Final = pods.team_key({**CREDENTIALS, "arize_success_sampling_rate": "0.5"})
    cursors: Final = _cursors(pods)
    plan: Final[tuple[tuple[OwnedProxy, Endpoint, bool], ...]] = (
        (pods.first, "chat", False),
        (pods.second, "chat", True),
        (pods.first, "responses", False),
        (pods.second, "messages", True),
    )

    async def send_trace(trace_id: str) -> tuple[Sent, ...]:
        clients: Final = {
            id(pod): httpx.AsyncClient(base_url=str(pod.gateway.client.base_url), timeout=15, trust_env=False)
            for pod in (pods.first, pods.second)
        }
        try:
            return tuple(
                await asyncio.gather(
                    *(
                        pods.send_async(clients[id(pod)], _call(sampled, trace_id, endpoint, stream))
                        for pod, endpoint, stream in plan
                    )
                )
            )
        finally:
            for opened in clients.values():
                await opened.aclose()

    by_trace: Final = {
        trace_id: (
            tuple(pods.send(pod, _call(sampled, trace_id, endpoint, stream)) for pod, endpoint, stream in plan)
            if client == "sync"
            else asyncio.run(send_trace(trace_id))
        )
        for trace_id in _trace_ids(f"endpoints-{client}")
    }

    verdicts: Final = _settled(pods, cursors, by_trace, pods.team_key(CREDENTIALS))

    assert 0 < _kept(verdicts) < TRACES


def test_the_workers_of_one_pod_agree(pods: Pods) -> None:
    sampled: Final = pods.team_key({**CREDENTIALS, "arize_success_sampling_rate": "0.5"})
    cursors: Final = _cursors(pods)
    by_trace: Final = {
        trace_id: pods.on_each_worker(pods.second, lambda trace_id=trace_id: _call(sampled, trace_id))
        for trace_id in _trace_ids("workers")
    }

    verdicts: Final = _settled(pods, cursors, by_trace, pods.team_key(CREDENTIALS))

    assert 0 < _kept(verdicts) < TRACES


def test_a_replaced_worker_reaches_the_verdict_its_predecessor_reached(pods: Pods) -> None:
    """A worker that restarts forgets the verdicts it remembered; the calls it serves next for the same
    traces still answer the way the first ones did."""
    sampled: Final = pods.team_key({**CREDENTIALS, "arize_success_sampling_rate": "0.5"})
    cursors: Final = _cursors(pods)
    fence_key: Final = pods.team_key(CREDENTIALS)
    traces: Final = _trace_ids("replaced")
    before: Final = {
        trace_id: pods.on_each_worker(pods.second, lambda t=trace_id: _call(sampled, t)) for trace_id in traces
    }
    _await_logged(pods, cursors.operator, tuple(chain.from_iterable(before.values())))
    _await_flushed(pods, cursors, fence_key)
    for victim in sorted(pods.second.worker_pids()):
        pods.second.kill_worker(victim)
    by_trace: Final = {
        trace_id: before[trace_id] + pods.on_each_worker(pods.second, lambda t=trace_id: _call(sampled, t))
        for trace_id in traces
    }

    verdicts: Final = _settled(pods, cursors, by_trace, fence_key)

    assert 0 < _kept(verdicts) < TRACES


def test_a_failed_call_is_kept_wherever_a_successful_call_of_its_trace_is(pods: Pods) -> None:
    """One call of the trace fails on one pod and answers to the error rate, another succeeds on the other
    pod and answers to the success rate: with the error rate above the success rate, no trace keeps its
    successful call and drops its failed one."""
    sampled: Final = pods.team_key(
        {**CREDENTIALS, "arize_success_sampling_rate": "0.3", "arize_error_sampling_rate": "0.7"}
    )
    cursors: Final = _cursors(pods)
    by_trace: Final = {
        trace_id: (
            pods.send(pods.first, _call(sampled, trace_id, content="sampling-fail")),
            pods.send(pods.second, _call(sampled, trace_id)),
        )
        for trace_id in _trace_ids("mixed", 32)
    }

    verdicts: Final = _settled(
        pods, cursors, by_trace, pods.team_key(CREDENTIALS), holds=lambda kept: kept[0] or not kept[1]
    )

    assert any(kept == (True, False) for kept in verdicts.values()), "no trace sat between the two rates"
    assert any(kept == (True, True) for kept in verdicts.values()), "no trace sat under both rates"


def test_a_failed_call_answers_to_the_error_rate_on_a_pod_that_dropped_its_trace(pods: Pods) -> None:
    """A pod that already dropped a successful call of the trace still keeps a later failed call of it at an
    unset error rate, as the pod that only served the failed call does."""
    sampled: Final = pods.team_key({**CREDENTIALS, "arize_success_sampling_rate": "0.3"})
    cursors: Final = _cursors(pods)
    traces: Final = _trace_ids("failed-after-dropped")
    succeeded: Final = {trace_id: pods.send(pods.first, _call(sampled, trace_id)) for trace_id in traces}
    _await_logged(pods, cursors.operator, tuple(succeeded.values()))
    by_trace: Final = {
        trace_id: (
            succeeded[trace_id],
            pods.send(pods.first, _call(sampled, trace_id, content="sampling-fail")),
            pods.send(pods.second, _call(sampled, trace_id, content="sampling-fail")),
        )
        for trace_id in traces
    }

    verdicts: Final = _settled(
        pods, cursors, by_trace, pods.team_key(CREDENTIALS), holds=lambda kept: kept[1] and kept[2]
    )

    assert any(not kept[0] for kept in verdicts.values()), "no successful call was dropped before its trace failed"


def test_a_successful_call_answers_to_the_success_rate_on_a_pod_that_kept_its_failed_call(pods: Pods) -> None:
    """A pod that already kept a failed call of the trace drops a later successful call of it at a zero
    success rate, as the pod that only served the successful call does."""
    sampled: Final = pods.team_key({**CREDENTIALS, "arize_success_sampling_rate": "0.0"})
    cursors: Final = _cursors(pods)
    traces: Final = _trace_ids("succeeded-after-kept")
    failed: Final = {
        trace_id: pods.send(pods.first, _call(sampled, trace_id, content="sampling-fail")) for trace_id in traces
    }
    _await_logged(pods, cursors.operator, tuple(failed.values()))
    by_trace: Final = {
        trace_id: (
            failed[trace_id],
            pods.send(pods.first, _call(sampled, trace_id)),
            pods.send(pods.second, _call(sampled, trace_id)),
        )
        for trace_id in traces
    }

    _settled(pods, cursors, by_trace, pods.team_key(CREDENTIALS), holds=lambda kept: kept == (True, False, False))


def test_concurrent_calls_of_one_trace_on_one_worker_each_answer_to_their_own_outcome(pods: Pods) -> None:
    """A failed and a successful call of the trace in flight at once on the single worker of the first pod
    are decided apart: the failed one is kept at an unset error rate, the successful one dropped at a zero
    success rate."""
    sampled: Final = pods.team_key({**CREDENTIALS, "arize_success_sampling_rate": "0.0"})
    cursors: Final = _cursors(pods)
    traces: Final = _trace_ids("concurrent-mixed")
    plan: Final = tuple(product(traces, ("sampling-fail", "hello")))

    def send(job: tuple[str, str]) -> Sent:
        return pods.send(pods.first, _call(sampled, job[0], content=job[1]))

    with ThreadPoolExecutor(max_workers=8) as pool:
        sent: Final = tuple(pool.map(send, plan))
    by_trace: Final = {trace_id: tuple(call for call in sent if call.trace_id == trace_id) for trace_id in traces}

    _settled(pods, cursors, by_trace, pods.team_key(CREDENTIALS), holds=lambda kept: kept == (True, False))


@pytest.mark.parametrize(
    ("variables", "expected"),
    [({"arize_success_sampling_rate": "1.0"}, TRACES), ({"arize_success_sampling_rate": "0.0"}, 0), ({}, TRACES)],
    ids=["one", "zero", "unset"],
)
def test_a_rate_at_either_end_keeps_every_trace_or_none(
    pods: Pods, variables: Mapping[str, str], expected: int
) -> None:
    sampled: Final = pods.team_key({**CREDENTIALS, **variables})
    cursors: Final = _cursors(pods)
    by_trace: Final = {
        trace_id: tuple(pods.send(pod, _call(sampled, trace_id)) for pod in (pods.first, pods.second))
        for trace_id in _trace_ids(f"edge-{'-'.join(f'{k}={v}' for k, v in sorted(variables.items()))}")
    }

    verdicts: Final = _settled(pods, cursors, by_trace, pods.team_key(CREDENTIALS))

    assert _kept(verdicts) == expected


def test_a_request_without_a_traceparent_is_sampled_whole_at_the_rate(pods: Pods) -> None:
    """With no caller trace the proxy starts one per request: each request's server and model spans travel
    together and the rate still keeps some requests and drops others."""
    sampled: Final = pods.team_key({**CREDENTIALS, "arize_success_sampling_rate": "0.5"})
    cursors: Final = _cursors(pods)
    sent: Final = tuple(
        pods.send(pod, _call(sampled, None)) for _, pod in product(range(12), (pods.first, pods.second))
    )
    _await_logged(pods, cursors.operator, sent)
    trace_of: Final = {
        str(span["attributes"][CALL_ID]): span["trace_id"]
        for span in recorded_spans(pods.sinks.operator, cursors.operator)[1]
        if CALL_ID in span["attributes"]
    }
    fences: Final = pods.fence(pods.team_key(CREDENTIALS))

    def check() -> Mapping[str, tuple[bool, bool]]:
        spans: Final = _team_spans(pods, cursors.arize)
        assert {fence.call_id for fence in fences} <= _delivered(spans, fences), (
            "the unsampled team's calls have not arrived yet"
        )
        kinds: Final = {
            call.call_id: (
                any(span["kind"] == SERVER for span in spans if span["trace_id"] == trace_of[call.call_id]),
                any(OPERATION in span["attributes"] for span in spans if span["trace_id"] == trace_of[call.call_id]),
            )
            for call in sent
        }
        partial: Final = {call_id: kept for call_id, kept in kinds.items() if kept[0] != kept[1]}
        assert partial == {}, f"requests reached the space without their whole tree (server, model): {partial}"
        return kinds

    kept: Final = sum(1 for server, _ in until_asserted(check, seconds=30).values() if server)
    assert 0 < kept < len(sent), f"a rate of 0.5 kept {kept} of {len(sent)} requests"


def test_a_key_level_destination_follows_the_verdict_of_its_trace(pods: Pods) -> None:
    team: Final = pods.scenario.team()
    sampled: Final = pods.scenario.key(
        team_id=team,
        metadata={
            "logging": [
                {
                    "callback_name": "arize",
                    "callback_type": "success",
                    "callback_vars": {**CREDENTIALS, "arize_success_sampling_rate": "0.5"},
                }
            ]
        },
    )
    cursors: Final = _cursors(pods)
    by_trace: Final = {
        trace_id: tuple(pods.send(pod, _call(sampled, trace_id)) for pod in (pods.first, pods.second, pods.first))
        for trace_id in _trace_ids("key-level")
    }

    verdicts: Final = _settled(pods, cursors, by_trace, pods.team_key(CREDENTIALS))

    assert 0 < _kept(verdicts) < TRACES


def test_a_concurrent_burst_keeps_every_trace_whole(pods: Pods) -> None:
    """Four calls per trace, all in flight at once across both pods."""
    sampled: Final = pods.team_key({**CREDENTIALS, "arize_success_sampling_rate": "0.5"})
    cursors: Final = _cursors(pods)
    traces: Final = _trace_ids("burst", 24)
    plan: Final = tuple(product(traces, (pods.first, pods.second, pods.first, pods.second)))

    def send(job: tuple[str, OwnedProxy]) -> Sent:
        return pods.send(job[1], _call(sampled, job[0]))

    with ThreadPoolExecutor(max_workers=16) as pool:
        sent: Final = list(pool.map(send, plan))
    by_trace: Final = {trace_id: tuple(call for call in sent if call.trace_id == trace_id) for trace_id in traces}

    verdicts: Final = _settled(pods, cursors, by_trace, pods.team_key(CREDENTIALS))

    assert 0 < _kept(verdicts) < len(traces)


def test_a_stalled_team_space_during_a_burst_keeps_serving_and_each_trace_whole(pods: Pods) -> None:
    """The team's Arize endpoint stops answering while a burst of calls runs over both pods: every call is
    still served, and once the endpoint answers again each trace arrives whole or not at all."""
    sampled: Final = pods.team_key({**CREDENTIALS, "arize_success_sampling_rate": "0.5"})
    cursors: Final = _cursors(pods)
    traces: Final = _trace_ids("stalled")
    plan: Final = tuple(product(traces, (pods.first, pods.second)))

    def send(job: tuple[str, OwnedProxy]) -> Sent:
        return pods.send(job[1], _call(sampled, job[0]))

    configure_sink(pods.sinks.arize, paused=True)
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            sent: Final = tuple(pool.map(send, plan))
    finally:
        configure_sink(pods.sinks.arize, paused=False)
    by_trace: Final = {trace_id: tuple(call for call in sent if call.trace_id == trace_id) for trace_id in traces}

    verdicts: Final = _settled(pods, cursors, by_trace, pods.team_key(CREDENTIALS))

    assert 0 < _kept(verdicts) < len(traces)


def _write_config(directory: Path, sinks: SpanSinks, provider: Wire) -> Path:
    shared: Final = object_value(
        JSON.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    )
    config: Final = {
        **shared,
        "model_list": [
            *list_value(shared.get("model_list") or []),
            {
                "model_name": MODEL,
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "integration-provider-key",
                    "api_base": provider.url + "/v1",
                },
            },
        ],
        "litellm_settings": {
            **object_value(shared.get("litellm_settings") or {}),
            "callbacks": ["langfuse_otel"],
            "otel_tenant_destination_mode": "additive",
            "provider_url_destination_allowed_hosts": [urlparse(sinks.operator).netloc, urlparse(sinks.arize).netloc],
        },
        "router_settings": {**object_value(shared.get("router_settings") or {}), "num_retries": 0},
        "general_settings": {**object_value(shared.get("general_settings") or {}), "user_api_key_cache_ttl": 2},
    }
    path: Final = directory / "team-sampling.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@contextmanager
def _two_pods(provider: Wire, sinks: SpanSinks, directory: Path) -> Generator[Pods]:
    environment: Final = {
        "LITELLM_OTEL_V2": "1",
        "OTEL_BSP_SCHEDULE_DELAY": "300",
        "MAPPER_NAMES": "genai,openinference",
        "LANGFUSE_HOST": sinks.operator,
        "LANGFUSE_PUBLIC_KEY": "pk-lf-operator",
        "LANGFUSE_SECRET_KEY": "sk-lf-operator",
        "ARIZE_SPACE_KEY": "space-operator",
        "ARIZE_API_KEY": "arize-operator",
        "ARIZE_HTTP_ENDPOINT": sinks.arize + "/v1/traces",
    }
    removed: Final = ("LITELLM_OTEL_TENANT_DESTINATION_MODE", "OTEL_TRACES_SAMPLER", "OTEL_TRACES_SAMPLER_ARG")
    config: Final = _write_config(directory, sinks, provider)
    for pod in ("first", "second"):
        (directory / pod).mkdir()
    with gateway_from_environment() as shared, scratch_database("team_sampling") as database_url:
        overrides: Final = {**environment, "DATABASE_URL": database_url}
        with (
            owned_proxy_process(
                shared, directory / "first", overrides, config=config, remove_environment=removed
            ) as first,
            owned_proxy_process(
                shared,
                directory / "second",
                overrides,
                config=config,
                remove_environment=removed,
                workers=2,
                healthcheck=WorkerHealthcheck.LENIENT,
            ) as second,
            replace(first.gateway, database_url=database_url).scenario() as scenario,
        ):
            eventually(second.worker_pids, lambda pids: len(pids) == 2, seconds=120)
            yield Pods(first, second, scenario, sinks)


@pytest.fixture(scope="module")
def provider() -> Iterator[Wire]:
    with wire_server(_upstream) as wire:
        yield wire


@pytest.fixture(scope="module")
def pods(provider: Wire, audit_sinks: SpanSinks, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Pods]:
    with _two_pods(provider, audit_sinks, tmp_path_factory.mktemp("team-sampling")) as started:
        yield started
