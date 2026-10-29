import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from itertools import chain
from typing import Final

import httpx
from integration._support.client import Gateway, eventually
from integration._support.wire import Request, wire_server
from integration._support.ports import allocate_port

from integration.providers._vertex_batch_rows import (
    ROUTES,
    Route,
    chat_row,
    encoded,
    error_of,
    gcs_peer,
    held_gcs_peer,
    jsonl,
    roleless_row,
    stored_custom_ids,
    unconvertible_line_message,
    upload,
    vertex_model,
)


@dataclass(frozen=True, slots=True)
class _Upload:
    route: Route
    bad_line: int | None
    identity: str


@dataclass(frozen=True, slots=True)
class _Answer:
    sent: _Upload
    response: httpx.Response


def _uploads(count: int, *, bad: bool) -> tuple[_Upload, ...]:
    return tuple(
        _Upload(ROUTES[index % 2], 2 + index if bad else None, f"{'bad' if bad else 'good'}-{uuid.uuid4().hex}")
        for index in range(count)
    )


def _content(item: _Upload, model: str) -> bytes:
    if item.bad_line is None:
        return jsonl(encoded(chat_row(item.identity, model, "kept")))
    leading: Final = tuple(
        encoded(chat_row(f"{item.identity}-{index}", model, "fine")) for index in range(1, item.bad_line)
    )
    return jsonl(*leading, encoded(roleless_row(item.identity, model)))


def _burst_client(gateway: Gateway) -> httpx.Client:
    return httpx.Client(
        base_url=str(gateway.client.base_url),
        timeout=180,
        trust_env=False,
        limits=httpx.Limits(max_connections=64, max_keepalive_connections=0),
    )


def _send(gateway: Gateway, client: httpx.Client, model: str, item: _Upload) -> _Answer:
    return _Answer(item, upload(gateway, item.route, model, _content(item, model), client=client))


def _submit(
    pool: ThreadPoolExecutor, gateway: Gateway, client: httpx.Client, model: str, items: tuple[_Upload, ...]
) -> tuple[Future[_Answer], ...]:
    return tuple(pool.submit(_send, gateway, client, model, item) for item in items)


def _burst(gateway: Gateway, client: httpx.Client, model: str, items: tuple[_Upload, ...]) -> tuple[_Answer, ...]:
    with ThreadPoolExecutor(max_workers=len(items)) as pool:
        return tuple(future.result(timeout=180) for future in _submit(pool, gateway, client, model, items))


def _landed(requests: tuple[Request, ...]) -> tuple[str, ...]:
    return tuple(sorted(chain.from_iterable(stored_custom_ids(request) for request in requests)))


def _assert_names_its_own_line(answer: _Answer) -> None:
    assert answer.sent.bad_line is not None
    assert answer.response.status_code == 400, (answer.sent, answer.response.text)
    message: Final = str(error_of(answer.response)["message"])
    assert message.startswith(unconvertible_line_message(answer.sent.bad_line) + "missing field 'role'"), (
        answer.sent,
        message,
    )


def _assert_each_bad_file_names_its_own_line(answers: tuple[_Answer, ...]) -> None:
    for answer in answers:
        _assert_names_its_own_line(answer)


def _assert_each_good_file_landed_once(answers: tuple[_Answer, ...], requests: tuple[Request, ...]) -> None:
    assert all(answer.response.status_code == 200 for answer in answers), [a.response.text for a in answers]
    assert _landed(requests) == tuple(sorted(answer.sent.identity for answer in answers))


def _mixed(gateway: Gateway, client: httpx.Client, model: str, bad: int, good: int) -> tuple[_Answer, ...]:
    return _burst(gateway, client, model, _uploads(bad, bad=True) + _uploads(good, bad=False))


def _split(answers: tuple[_Answer, ...]) -> tuple[tuple[_Answer, ...], tuple[_Answer, ...]]:
    return (
        tuple(answer for answer in answers if answer.sent.bad_line is not None),
        tuple(answer for answer in answers if answer.sent.bad_line is None),
    )


def test_concurrent_mixed_burst_names_each_bad_files_own_line_and_lands_each_good_file_once(gateway: Gateway) -> None:
    with wire_server(gcs_peer) as wire, gateway.scenario() as scenario, _burst_client(gateway) as client:
        model: Final = vertex_model(gateway, scenario, wire.url)
        bad, good = _split(_mixed(gateway, client, model, 12, 12))
        _assert_each_bad_file_names_its_own_line(bad)
        _assert_each_good_file_landed_once(good, wire.drain())


def test_bucket_outage_answers_500_without_blaming_a_row_and_recovers_on_restart(gateway: Gateway) -> None:
    port: Final = allocate_port()
    with gateway.scenario() as scenario, _burst_client(gateway) as client:
        model: Final = vertex_model(gateway, scenario, f"http://127.0.0.1:{port}")
        with wire_server(gcs_peer, port=port) as before:
            bad_before, good_before = _split(_mixed(gateway, client, model, 6, 6))
            _assert_each_bad_file_names_its_own_line(bad_before)
            _assert_each_good_file_landed_once(good_before, before.drain())
        during: Final = _mixed(gateway, client, model, 6, 6)
        assert all(answer.response.status_code == 500 for answer in during), [a.response.text for a in during]
        assert not any("Batch input file line" in answer.response.text for answer in during)
        assert gateway.client.get("/health/liveliness").status_code == 200
        with wire_server(gcs_peer, port=port) as after:
            bad_after, good_after = _split(_mixed(gateway, client, model, 6, 6))
            _assert_each_bad_file_names_its_own_line(bad_after)
            _assert_each_good_file_landed_once(good_after, after.drain())


def test_slow_bucket_holds_good_uploads_without_delaying_bad_rows_400s(gateway: Gateway) -> None:
    release: Final = threading.Event()
    good_items: Final = _uploads(6, bad=False)
    with (
        wire_server(held_gcs_peer(release)) as wire,
        gateway.scenario() as scenario,
        _burst_client(gateway) as client,
        ThreadPoolExecutor(max_workers=12) as pool,
    ):
        try:
            model: Final = vertex_model(gateway, scenario, wire.url)
            held: Final = _submit(pool, gateway, client, model, good_items)
            eventually(wire.received.qsize, lambda size: size == len(good_items), seconds=60)
            bad: Final = tuple(
                future.result(timeout=60)
                for future in _submit(pool, gateway, client, model, _uploads(6, bad=True))
            )
            _assert_each_bad_file_names_its_own_line(bad)
            assert not any(future.done() for future in held)
            release.set()
            _assert_each_good_file_landed_once(tuple(future.result(timeout=120) for future in held), wire.drain())
        finally:
            release.set()
