import re
import threading
import uuid
from collections.abc import Iterator
from pathlib import Path
from queue import SimpleQueue
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, eventually
from integration._support.process import owned_proxy_process, owned_upstream
from integration.authorization._hidden_alias_budget import (
    fresh_chat,
    landed_all_once,
    reset_upstream_requests,
    upstream_hits,
    upstream_requests,
)
from integration.authorization._key_disallowed_models import (
    BURST_ROUTES,
    DENIED,
    KEY_DENIAL,
    DenyRig,
    denial,
    deny_rig,
    send,
    send_all,
)

pytestmark: Final = [pytest.mark.timeout(480), pytest.mark.shares_worker_state]

_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")


@pytest.fixture(scope="module")
def rig() -> Iterator[DenyRig]:
    with deny_rig() as built:
        yield built


def _route(marker: str) -> str:
    return marker.split("-")[1]


def _markers(prefix: str, count: int) -> tuple[str, ...]:
    return tuple(
        f"{prefix}-{BURST_ROUTES[index % len(BURST_ROUTES)]}-{index}-{uuid.uuid4().hex}" for index in range(count)
    )


def _status(candidate: Gateway, model: str, key: str) -> int | None:
    try:
        return fresh_chat(candidate, model, key, f"probe-{uuid.uuid4().hex}").status_code
    except httpx.TransportError:
        return None


def _settled(candidate: Gateway, model: str, key: str, status: int) -> None:
    eventually(
        lambda: frozenset(_status(candidate, model, key) for _ in range(8)),
        lambda seen: seen == frozenset({status}),
        seconds=120,
    )


def test_a_mixed_concurrent_burst_refuses_every_disallowed_call_and_serves_every_allowed_one_once(rig: DenyRig) -> None:
    refused_markers: Final = _markers("refused", 30)
    served_markers: Final = _markers("served", 30)
    reset_upstream_requests(rig.gateway.upstream_url)
    calls: Final = tuple((marker, rig.denied) for marker in refused_markers) + tuple(
        (marker, rig.allowed) for marker in served_markers
    )
    by_marker: Final = dict(calls)
    responses: Final = dict(
        zip(
            by_marker,
            send_all(
                lambda marker: send(
                    rig.gateway, _route(marker), by_marker[marker].serving(_route(marker)), rig.key, marker
                ),
                tuple(by_marker),
            ),
            strict=True,
        )
    )
    assert [denial(responses[marker]) for marker in refused_markers] == [KEY_DENIAL] * len(refused_markers)
    assert [responses[marker].status_code for marker in served_markers] == [200] * len(served_markers), [
        responses[marker].text for marker in served_markers if responses[marker].status_code != 200
    ]
    observed: Final = upstream_requests(rig.gateway.upstream_url)
    assert {marker: upstream_hits(observed, marker) for marker in refused_markers} == dict.fromkeys(refused_markers, 0)
    assert {marker: upstream_hits(observed, marker) for marker in served_markers} == dict.fromkeys(served_markers, 1)
    rows: Final = landed_all_once(rig.key, frozenset(served_markers))
    assert {row["status"] for row in rows} == {"success"}, rows


def test_flipping_disallowed_models_during_a_burst_only_ever_serves_or_refuses(rig: DenyRig) -> None:
    seen: Final[SimpleQueue[tuple[str, int]]] = SimpleQueue()
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key()
        reset_upstream_requests(rig.gateway.upstream_url)

        def wave() -> frozenset[int]:
            markers: Final = tuple(f"flip-{index}-{uuid.uuid4().hex}" for index in range(8))
            responses: Final = send_all(lambda marker: fresh_chat(rig.gateway, rig.denied.chat, key, marker), markers)
            for marker, response in zip(markers, responses, strict=True):
                seen.put((marker, response.status_code))
            return frozenset(response.status_code for response in responses)

        assert wave() == frozenset({200})
        for disallowed, status in (([rig.denied.chat], DENIED), ([], 200), ([rig.denied.chat], DENIED)):
            flip = threading.Thread(
                target=rig.gateway.post, args=("/key/update", {"key": key, "disallowed_models": disallowed})
            )
            flip.start()
            eventually(lambda: wave() | wave(), lambda found, status=status: found == frozenset({status}), seconds=60)
            flip.join(timeout=30)
            assert not flip.is_alive()
        collected: Final = tuple(seen.get() for _ in range(seen.qsize()))
        assert {status for _, status in collected} == {200, DENIED}, collected
        observed: Final = upstream_requests(rig.gateway.upstream_url)
        assert {marker: upstream_hits(observed, marker) for marker, _ in collected} == {
            marker: 1 if status == 200 else 0 for marker, status in collected
        }
        landed_all_once(key, frozenset(marker for marker, status in collected if status == 200))


def test_a_provider_outage_never_turns_a_refusal_into_a_provider_error(rig: DenyRig, tmp_path: Path) -> None:
    with owned_upstream(tmp_path) as slot, rig.gateway.scenario() as scenario:
        allowed: Final = scenario.model(api_base=f"{slot.url}/v1")
        refused_model: Final = scenario.model(api_base=f"{slot.url}/v1")
        key: Final = scenario.key(disallowed_models=[refused_model])
        _settled(rig.gateway, allowed, key, 200)
        _settled(rig.gateway, refused_model, key, DENIED)
        slot.stop()
        during: Final = tuple(f"outage-{index}-{uuid.uuid4().hex}" for index in range(10))
        refused: Final = send_all(lambda marker: fresh_chat(rig.gateway, refused_model, key, marker), during)
        assert [denial(response) for response in refused] == [KEY_DENIAL] * len(during)
        failed: Final = send_all(lambda marker: fresh_chat(rig.gateway, allowed, key, marker), during)
        assert all(response.status_code >= 500 for response in failed), [response.text for response in failed]
        assert rig.gateway.request("GET", "/health/liveliness").status_code == 200
        slot.start()
        _settled(rig.gateway, allowed, key, 200)
        after: Final = tuple(f"recovered-{index}-{uuid.uuid4().hex}" for index in range(10))
        refused_after: Final = send_all(lambda marker: fresh_chat(rig.gateway, refused_model, key, marker), after)
        assert [denial(response) for response in refused_after] == [KEY_DENIAL] * len(after)
        assert sum(upstream_hits(upstream_requests(slot.url), marker) for marker in after) == 0


def _worker_startups(log: Path) -> tuple[tuple[int, ...], int]:
    text: Final = log.read_text()
    return tuple(int(found[1]) for found in _STARTED_WORKER.finditer(text)), text.count("Application startup complete.")


def test_a_killed_worker_leaves_its_sibling_and_its_replacement_refusing_disallowed_models(
    rig: DenyRig, tmp_path: Path
) -> None:
    with owned_proxy_process(rig.gateway, tmp_path, {}, workers=2) as owned:
        candidate: Final = owned.gateway
        workers, _ = eventually(
            lambda: _worker_startups(owned.log), lambda found: len(found[0]) == 2 and found[1] == 2, seconds=120
        )
        _settled(candidate, rig.allowed.chat, rig.key, 200)
        _settled(candidate, rig.denied.chat, rig.key, DENIED)
        with owned.replacing_worker(workers[0]) as kill:
            kill()
            eventually(lambda: _status(candidate, rig.allowed.chat, rig.key), lambda found: found == 200, seconds=60)
            during: Final = tuple(f"kill-{index}-{uuid.uuid4().hex}" for index in range(8))
            refused: Final = send_all(lambda marker: fresh_chat(candidate, rig.denied.chat, rig.key, marker), during)
            assert [denial(response) for response in refused] == [KEY_DENIAL] * len(during)
        eventually(lambda: _worker_startups(owned.log), lambda found: len(found[0]) == 3 and found[1] == 3, seconds=180)
        _settled(candidate, rig.allowed.chat, rig.key, 200)
        after: Final = tuple(f"replaced-{index}-{uuid.uuid4().hex}" for index in range(16))
        refused_after: Final = send_all(lambda marker: fresh_chat(candidate, rig.denied.chat, rig.key, marker), after)
        assert [denial(response) for response in refused_after] == [KEY_DENIAL] * len(after)


def test_a_restarted_proxy_still_refuses_a_keys_disallowed_models(rig: DenyRig, tmp_path: Path) -> None:
    seen: Final[SimpleQueue[int | None]] = SimpleQueue()
    stop: Final = threading.Event()

    def hammer(candidate: Gateway) -> None:
        while not stop.is_set():
            seen.put(_status(candidate, rig.denied.chat, rig.key))

    with owned_proxy_process(rig.gateway, tmp_path / "first", {}, workers=2) as first:
        _settled(first.gateway, rig.denied.chat, rig.key, DENIED)
        burst: Final = tuple(threading.Thread(target=hammer, args=(first.gateway,)) for _ in range(8))
        for thread in burst:
            thread.start()
        eventually(seen.qsize, lambda count: count >= 24, seconds=60)
    stop.set()
    for thread in burst:
        thread.join(timeout=60)
    statuses: Final = tuple(seen.get() for _ in range(seen.qsize()))
    assert DENIED in statuses and 200 not in statuses, statuses
    with owned_proxy_process(rig.gateway, tmp_path / "second", {}, workers=2) as second:
        _settled(second.gateway, rig.denied.chat, rig.key, DENIED)
        _settled(second.gateway, rig.allowed.chat, rig.key, 200)
        markers: Final = tuple(f"restarted-{index}-{uuid.uuid4().hex}" for index in range(10))
        served: Final = send_all(lambda marker: fresh_chat(second.gateway, rig.allowed.chat, rig.key, marker), markers)
        assert [response.status_code for response in served] == [200] * len(markers)
        landed_all_once(rig.key, frozenset(markers))
