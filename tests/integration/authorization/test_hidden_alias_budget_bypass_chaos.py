import os
import re
import signal
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import SimpleQueue
from typing import Final

import httpx
import pytest
from pydantic import JsonValue
from integration._support.client import Gateway, eventually
from integration._support.process import owned_proxy_process, owned_upstream
from integration.authorization._hidden_alias_budget import (
    BUDGET_EXCEEDED,
    AliasRig,
    alias_rig,
    assert_free_row,
    chat_statuses,
    exhausted_key,
    fresh_chat,
    fresh_message,
    fresh_response,
    hidden,
    install_aliases,
    landed_all_once,
    remove_aliases,
    settle_candidate,
    settle_chat,
    upstream_hits,
    upstream_requests,
)

pytestmark: Final = pytest.mark.timeout(240)

_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
_WAVE: Final = 8
_STREAM: Final[Mapping[str, JsonValue]] = {"stream": True}


@pytest.fixture(scope="module")
def rig() -> Iterator[AliasRig]:
    with alias_rig() as built:
        yield built


def _burst_markers(prefix: str, count: int) -> tuple[str, ...]:
    return tuple(f"{prefix}-{index}-{uuid.uuid4().hex}" for index in range(count))


def _send_all(send: Callable[[str], httpx.Response], markers: tuple[str, ...]) -> tuple[httpx.Response, ...]:
    with ThreadPoolExecutor(max_workers=len(markers)) as pool:
        return tuple(pool.map(send, markers))


def _mixed_call(rig: AliasRig, key: str, marker: str) -> httpx.Response:
    senders: Final[Mapping[str, Callable[[], httpx.Response]]] = {
        "chat": lambda: fresh_chat(rig.gateway, rig.hidden_free, key, marker),
        "chatstream": lambda: fresh_chat(rig.gateway, rig.hidden_free, key, marker, _STREAM),
        "messages": lambda: fresh_message(rig.gateway, rig.hidden_responses, key, marker),
        "messagesstream": lambda: fresh_message(rig.gateway, rig.hidden_responses_stream, key, marker, _STREAM),
        "responses": lambda: fresh_response(rig.gateway, rig.hidden_responses, key, marker),
        "responsesstream": lambda: fresh_response(rig.gateway, rig.hidden_responses_stream, key, marker, _STREAM),
    }
    return senders[marker.split("-")[1]]()


_KINDS: Final = ("chat", "chatstream", "messages", "messagesstream", "responses", "responsesstream")


def test_mixed_concurrent_burst_through_hidden_free_aliases_lands_each_call_once(rig: AliasRig) -> None:
    markers: Final = tuple(f"burst-{_KINDS[index % len(_KINDS)]}-{index}-{uuid.uuid4().hex}" for index in range(30))
    with rig.gateway.scenario() as scenario:
        key: Final = exhausted_key(rig, scenario)
        upstream_requests(rig.gateway.upstream_url)
        responses: Final = _send_all(lambda marker: _mixed_call(rig, key, marker), markers)
        assert [response.status_code for response in responses] == [200] * len(markers), [
            response.text for response in responses if response.status_code != 200
        ]
        observed: Final = upstream_requests(rig.gateway.upstream_url)
        assert {marker: upstream_hits(observed, marker) for marker in markers} == dict.fromkeys(markers, 1)
        rows: Final = landed_all_once(key, frozenset(markers))
        assert len(rows) == len(markers), rows
        for row in rows:
            assert float(str(row["spend"])) == 0.0, row
            assert row["status"] == "success", row


def test_alias_flipped_to_a_free_group_during_a_burst_only_ever_serves_or_refuses(rig: AliasRig) -> None:
    alias: Final = "flip-" + uuid.uuid4().hex
    seen: Final[SimpleQueue[tuple[str, int]]] = SimpleQueue()
    with rig.gateway.scenario() as scenario:
        install_aliases(rig.gateway, {alias: hidden(rig.paid)})
        scenario.cleanups.callback(remove_aliases, rig.gateway, frozenset({alias}))
        settle_chat(rig, alias, rig.gateway.key, 200)
        key: Final = exhausted_key(rig, scenario)
        settle_chat(rig, alias, key, BUDGET_EXCEEDED)
        upstream_requests(rig.gateway.upstream_url)

        def wave(candidate: Gateway) -> frozenset[int]:
            markers: Final = _burst_markers("flip", _WAVE)
            responses: Final = _send_all(lambda marker: fresh_chat(candidate, alias, key, marker), markers)
            for marker, response in zip(markers, responses, strict=True):
                seen.put((marker, response.status_code))
            return frozenset(response.status_code for response in responses)

        assert wave(rig.gateway) == frozenset({BUDGET_EXCEEDED})
        flip: Final = threading.Thread(target=install_aliases, args=(rig.gateway, {alias: hidden(rig.free)}))
        flip.start()
        eventually(lambda: wave(rig.gateway) | wave(rig.gateway), lambda found: found == frozenset({200}), seconds=60)
        flip.join(timeout=30)
        assert not flip.is_alive()
        settle_chat(rig, alias, key, 200)
        collected: Final = tuple(seen.get() for _ in range(seen.qsize()))
        assert {status for _, status in collected} <= {200, BUDGET_EXCEEDED}, collected
        served: Final = frozenset(marker for marker, status in collected if status == 200)
        refused: Final = frozenset(marker for marker, status in collected if status == BUDGET_EXCEEDED)
        assert served and refused, collected
        observed: Final = upstream_requests(rig.gateway.upstream_url)
        assert all(upstream_hits(observed, marker) == 1 for marker in served), collected
        assert all(upstream_hits(observed, marker) == 0 for marker in refused), collected
        for row in landed_all_once(key, served):
            assert_free_row(row, alias)


def _tolerant_status(candidate: Gateway, model: str, key: str, marker: str) -> int | None:
    try:
        return fresh_chat(candidate, model, key, marker).status_code
    except httpx.TransportError:
        return None


@pytest.mark.timeout(480)
def test_upstream_outage_behind_hidden_free_alias_is_a_provider_error_and_recovers(
    rig: AliasRig, tmp_path: Path
) -> None:
    alias: Final = "hidden-outage-" + uuid.uuid4().hex
    with owned_upstream(tmp_path) as slot, rig.gateway.scenario() as scenario:
        group: Final = scenario.model(api_base=f"{slot.url}/v1", input_cost_per_token=0, output_cost_per_token=0)
        install_aliases(rig.gateway, {alias: hidden(group)})
        scenario.cleanups.callback(remove_aliases, rig.gateway, frozenset({alias}))
        settle_chat(rig, alias, rig.gateway.key, 200)
        key: Final = exhausted_key(rig, scenario)
        before: Final = _burst_markers("outage-before", 10)
        served_before: Final = _send_all(lambda marker: fresh_chat(rig.gateway, alias, key, marker), before)
        assert [response.status_code for response in served_before] == [200] * 10
        slot.stop()
        during: Final = _burst_markers("outage-during", 10)
        failed: Final = _send_all(lambda marker: fresh_chat(rig.gateway, alias, key, marker), during)
        for response in failed:
            assert response.status_code >= 500, response.text
            assert "budget" not in response.text.lower(), response.text
        assert rig.gateway.request("GET", "/health/liveliness").status_code == 200
        unrelated: Final = fresh_chat(rig.gateway, rig.hidden_free, key, "outage-unrelated-" + uuid.uuid4().hex)
        assert unrelated.status_code == 200, unrelated.text
        slot.start()
        settle_candidate(rig.gateway, alias, key, 200, seconds=90)
        after: Final = _burst_markers("outage-after", 10)
        served_after: Final = _send_all(lambda marker: fresh_chat(rig.gateway, alias, key, marker), after)
        assert [response.status_code for response in served_after] == [200] * 10
        observed: Final = upstream_requests(slot.url)
        assert {marker: upstream_hits(observed, marker) for marker in after} == dict.fromkeys(after, 1)
        for row in landed_all_once(key, frozenset(before + after)):
            assert_free_row(row, alias)


def _worker_startups(log: Path) -> tuple[tuple[int, ...], int]:
    text: Final = log.read_text()
    started: Final = tuple(int(found[1]) for found in _STARTED_WORKER.finditer(text))
    return started, text.count("Application startup complete.")


@pytest.mark.timeout(480)
def test_killed_worker_leaves_the_sibling_serving_hidden_free_aliases(rig: AliasRig, tmp_path: Path) -> None:
    with rig.gateway.scenario() as scenario:
        key: Final = exhausted_key(rig, scenario)
        with owned_proxy_process(rig.gateway, tmp_path, {}, workers=2) as owned:
            candidate: Final = owned.gateway
            workers, _ = eventually(
                lambda: _worker_startups(owned.log),
                lambda found: len(found[0]) == 2 and found[1] == 2,
                seconds=120,
            )
            settle_candidate(candidate, rig.hidden_free, key, 200, seconds=120)
            settle_candidate(candidate, rig.hidden_paid, key, BUDGET_EXCEEDED, seconds=120)
            os.kill(workers[0], signal.SIGKILL)
            eventually(
                lambda: _tolerant_status(candidate, rig.hidden_free, key, "kill-probe-" + uuid.uuid4().hex),
                lambda found: found == 200,
                seconds=60,
            )
            assert chat_statuses(candidate, rig.hidden_free, key, 8) == {200}
            assert chat_statuses(candidate, rig.hidden_paid, key, 8) == {BUDGET_EXCEEDED}
            eventually(
                lambda: _worker_startups(owned.log),
                lambda found: len(found[0]) == 3 and found[1] == 3,
                seconds=180,
            )
            settle_candidate(candidate, rig.hidden_free, key, 200, seconds=120)
            settle_candidate(candidate, rig.hidden_paid, key, BUDGET_EXCEEDED, seconds=120)
            markers: Final = _burst_markers("kill-after", 10)
            served: Final = _send_all(lambda marker: fresh_chat(candidate, rig.hidden_free, key, marker), markers)
            assert [response.status_code for response in served] == [200] * 10
            for row in landed_all_once(key, frozenset(markers)):
                assert_free_row(row, rig.hidden_free)
