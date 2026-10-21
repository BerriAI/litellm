"""Auto mode under disruption: a classifier outage in the middle of a mixed burst across the chat bridge, the
Responses bridge, and a native backend, streaming and not, concurrent classifier calls, the proxy stopped mid burst,
and a worker killed while the other keeps serving."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Final

import httpx
import pytest
from integration._support.client import eventually
from integration._support.process import graceful_stop_seconds, stop_root_process
from integration._support.safeguards_rig import (
    CHAT_MODEL,
    NATIVE_MODEL,
    RESPONSES_MODEL,
    Responder,
    Rig,
    auto_mode_body,
    chat_backend,
    classifier_dropping_the_connection,
    classifier_flagging_everything,
    classifier_slow,
    flagged,
    json_body,
    native_backend,
    native_message,
    nonce,
    responses_backend,
    safeguards_rig,
    stream_results,
    stream_tool_use_ids,
    text_block,
    tool_id,
    tool_use_block,
    tool_use_ids,
    unavailable,
)
from integration._support.wire import Reply, Request
from pydantic import JsonValue

pytestmark = pytest.mark.timeout(2 * graceful_stop_seconds() + 300)

BURST: Final = 10
MIX: Final = tuple(product((CHAT_MODEL, RESPONSES_MODEL, NATIVE_MODEL), (False, True))) * 4


@dataclass(frozen=True, slots=True)
class Outcome:
    request_nonce: str
    ids: tuple[str, ...]
    results: JsonValue


def scripted_backend(model: str, identity: str, stream: bool) -> Responder:
    if model == CHAT_MODEL:
        return chat_backend(identity, stream=stream)
    if model == RESPONSES_MODEL:
        return responses_backend(identity, stream=stream)
    message: Final = native_message(identity, content=(text_block("Deleting."), tool_use_block(identity)))
    return native_backend(message, stream=stream)


def send(rig: Rig, model: str, stream: bool, request_nonce: str) -> Outcome:
    body: Final = auto_mode_body(model, request_nonce)
    if stream:
        status, raw = rig.stream(body)
        assert status == 200, raw
        return Outcome(request_nonce, stream_tool_use_ids(raw), stream_results(raw))
    reply: Final = json_body(rig.post(body))
    return Outcome(request_nonce, tool_use_ids(reply), reply.get("safeguard_results"))


def mixed_burst(rig: Rig, classifier: Responder) -> tuple[Outcome, ...]:
    plan: Final = tuple((model, stream, nonce()) for model, stream in MIX)
    for model, stream, request_nonce in plan:
        rig.register(request_nonce, scripted_backend(model, tool_id(request_nonce), stream), classifier)
    with ThreadPoolExecutor(max_workers=len(plan)) as pool:
        return tuple(pool.map(lambda entry: send(rig, entry[0], entry[1], entry[2]), plan))


def chat_burst(rig: Rig, classifier: Responder) -> tuple[Outcome, ...]:
    nonces: Final = tuple(nonce() for _ in range(BURST))
    for request_nonce in nonces:
        rig.register(request_nonce, chat_backend(tool_id(request_nonce), stream=False), classifier)
    with ThreadPoolExecutor(max_workers=BURST) as pool:
        return tuple(pool.map(lambda request_nonce: send(rig, CHAT_MODEL, False, request_nonce), nonces))


def assert_landed_once(rig: Rig, received: Sequence[Request], request_nonce: str) -> None:
    own: Final = tuple(request for request in received if request_nonce.encode() in request.body)
    assert len(rig.backend_calls(own)) == 1, [request.target for request in own]
    assert len(rig.classifier_calls(own)) == 1, [request.target for request in own]


def assert_every_id_landed_once(rig: Rig, outcomes: Sequence[Outcome]) -> None:
    received: Final = rig.wire.drain()
    for outcome in outcomes:
        assert outcome.ids == (tool_id(outcome.request_nonce),), outcome
        assert_landed_once(rig, received, outcome.request_nonce)


def classifier_released_together(parties: int) -> Responder:
    barrier: Final = threading.Barrier(parties, timeout=20)

    def respond(request: Request) -> Reply:
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            return Reply(status=503, body=b'{"error": {"message": "classifier calls did not overlap"}}')
        return classifier_flagging_everything(request)

    return respond


def test_classifier_outage_mid_burst_degrades_then_recovers(rig: Rig) -> None:
    outage: Final = mixed_burst(rig, classifier_dropping_the_connection)
    for outcome in outage:
        assert outcome.results == unavailable(outcome.ids, "error"), outcome
    assert_every_id_landed_once(rig, outage)
    recovered: Final = mixed_burst(rig, classifier_flagging_everything)
    for outcome in recovered:
        assert outcome.results == flagged(outcome.ids), outcome
    assert_every_id_landed_once(rig, recovered)


def test_classifier_calls_of_concurrent_requests_overlap(rig: Rig) -> None:
    outcomes: Final = chat_burst(rig, classifier_released_together(BURST))
    for outcome in outcomes:
        assert outcome.results == flagged(outcome.ids), outcome
    assert_every_id_landed_once(rig, outcomes)


def test_proxy_stopped_mid_burst_fails_in_flight_requests_cleanly(tmp_path: Path) -> None:
    with safeguards_rig(tmp_path) as rig:
        nonces: Final = tuple(nonce() for _ in range(BURST))
        for request_nonce in nonces:
            rig.register(request_nonce, chat_backend(tool_id(request_nonce), stream=False), classifier_slow(4))

        def attempt(request_nonce: str) -> str:
            try:
                response: Final = rig.post(auto_mode_body(CHAT_MODEL, request_nonce))
            except httpx.HTTPError:
                return "error"
            return str(response.status_code)

        with ThreadPoolExecutor(max_workers=BURST) as pool:
            futures: Final = tuple(pool.submit(attempt, request_nonce) for request_nonce in nonces)
            eventually(lambda: rig.wire.received.qsize(), lambda count: count >= 2 * BURST, seconds=60)
            assert stop_root_process(rig.owned.process)
            outcomes: Final = tuple(future.result(timeout=graceful_stop_seconds() + 60) for future in futures)
        assert set(outcomes) <= {"200", "500", "502", "503", "error"}, outcomes
        received: Final = rig.wire.drain()
        for request_nonce in nonces:
            assert_landed_once(rig, received, request_nonce)
    with safeguards_rig(tmp_path) as fresh:
        request_nonce: Final = nonce()
        fresh.register(request_nonce, chat_backend(tool_id(request_nonce), stream=False))
        body: Final = json_body(fresh.post(auto_mode_body(CHAT_MODEL, request_nonce)))
        assert body["safeguard_results"] == flagged(tool_use_ids(body)), body


def classify_on(rig: Rig) -> Outcome:
    request_nonce: Final = nonce()
    rig.register(request_nonce, chat_backend(tool_id(request_nonce), stream=False))
    return send(rig, CHAT_MODEL, False, request_nonce)


def test_one_worker_killed_while_the_other_keeps_serving(tmp_path: Path) -> None:
    with safeguards_rig(tmp_path) as rig:
        victim: Final = min(rig.owned.worker_pids())
        with rig.owned.replacing_worker(victim) as kill:
            kill()
            during: Final = tuple(classify_on(rig) for _ in range(5))
        after: Final = classify_on(rig)
        for outcome in (*during, after):
            assert outcome.results == flagged(outcome.ids), outcome
        assert_every_id_landed_once(rig, (*during, after))
        assert victim not in rig.owned.worker_pids()
