import asyncio
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Final

import httpx
import pytest

from litellm.repositories.lens_export_repository import ClaimedEvent, ClaimedPayload
from litellm.tracing.ledger_export import (
    CostRunner,
    ExportSettings,
    Supported,
    Unreachable,
    Unsupported,
    UploadRunner,
    probe_lens,
    run_loop,
)

SETTINGS: Final = ExportSettings(batch_size=10, cost_batch_bytes=1024, upload_batch_bytes=4096)
BASE_URL: Final = "http://lens.test/prefix"


@dataclass
class FakeStore:
    events: tuple[ClaimedEvent, ...] = ()
    payloads: tuple[ClaimedPayload, ...] = ()
    acknowledged: list[str] = field(default_factory=list)  # mutable-ok: records calls for assertions
    retried: list[tuple[str, str]] = field(default_factory=list)  # mutable-ok: records calls for assertions
    parked: list[tuple[str, str]] = field(default_factory=list)  # mutable-ok: records calls for assertions
    payload_claims: list[int] = field(default_factory=list)  # mutable-ok: records byte budgets
    releases: int = 0
    acknowledged_event: asyncio.Event = field(default_factory=asyncio.Event)

    async def claim_events(self, limit: int, lease_seconds: int) -> tuple[ClaimedEvent, ...]:
        claimed, self.events = self.events[:limit], self.events[limit:]
        return claimed

    async def acknowledge_events(self, event_ids: Sequence[str]) -> int:
        self.acknowledged.extend(event_ids)
        if event_ids:
            self.acknowledged_event.set()
        return len(event_ids)

    async def retry_events(self, event_ids: Sequence[str], error: str) -> int:
        self.retried.extend((event_id, error) for event_id in event_ids)
        return len(event_ids)

    async def park_events(self, event_ids: Sequence[str], error: str) -> int:
        self.parked.extend((event_id, error) for event_id in event_ids)
        return len(event_ids)

    async def claim_payloads(self, limit: int, lease_seconds: int, max_bytes: int) -> tuple[ClaimedPayload, ...]:
        self.payload_claims.append(max_bytes)
        claimed, self.payloads = self.payloads[:limit], self.payloads[limit:]
        return claimed

    async def acknowledge_payloads(self, payloads: Sequence[ClaimedPayload]) -> int:
        self.acknowledged.extend(f"{payload.event_id}/{payload.field}" for payload in payloads)
        return len(payloads)

    async def retry_payloads(self, payloads: Sequence[ClaimedPayload], error: str) -> int:
        self.retried.extend((f"{payload.event_id}/{payload.field}", error) for payload in payloads)
        return len(payloads)

    async def park_payloads(self, payloads: Sequence[ClaimedPayload], error: str) -> int:
        self.parked.extend((f"{payload.event_id}/{payload.field}", error) for payload in payloads)
        return len(payloads)

    async def release(self) -> int:
        self.releases += 1
        return 0


def _event(event_id: str, padding: int = 0) -> ClaimedEvent:
    return ClaimedEvent(event_id=event_id, attempts=0, event=json.dumps({"request_id": event_id, "pad": "x" * padding}))


def _payload(event_id: str, content: str = "[]") -> ClaimedPayload:
    return ClaimedPayload(
        event_id=event_id,
        field="messages",
        team_id="team",
        api_key="hash",
        event_time_ms=1,
        content=content,
        bytes=len(content.encode()),
        sha256="0" * 64,
        attempts=0,
    )


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=BASE_URL, transport=httpx.MockTransport(handler))


def _results(*results: dict[str, object]) -> httpx.Response:
    return httpx.Response(200, json={"version": 1, "results": list(results)})


@pytest.mark.asyncio
async def test_each_event_is_acknowledged_parked_or_retried_by_its_own_result() -> None:
    store: Final = FakeStore(events=(_event("stored"), _event("rejected"), _event("missing")))
    paths: Final[list[str]] = []  # mutable-ok: captures request paths

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return _results(
            {"event_id": "stored", "status": "stored"},
            {"event_id": "rejected", "status": "rejected", "reason": "missing_event_time"},
        )

    assert await CostRunner(store, _client(handler), SETTINGS).step() == 1
    assert paths == ["/prefix/internal/ledger"]
    assert store.acknowledged == ["stored"]
    assert store.parked == [("rejected", "Rejected")]
    assert store.retried == [("missing", "MissingResult")]


@pytest.mark.parametrize(
    "respond",
    [
        lambda request: httpx.Response(503),
        lambda request: httpx.Response(404),
        lambda request: httpx.Response(422, json={"code": "unsupported_version"}),
        lambda request: httpx.Response(200, content=b"not json"),
    ],
    ids=["unavailable", "route_missing", "older_version", "invalid_response"],
)
@pytest.mark.asyncio
async def test_transient_failures_reschedule_every_event_and_never_park_or_acknowledge(
    respond: Callable[[httpx.Request], httpx.Response],
) -> None:
    store: Final = FakeStore(events=(_event("a"), _event("b")))

    assert await CostRunner(store, _client(respond), SETTINGS).step() == 0
    assert store.acknowledged == []
    assert store.parked == []
    assert [event_id for event_id, _ in store.retried] == ["a", "b"]


@pytest.mark.asyncio
async def test_network_errors_reschedule_the_claim() -> None:
    store: Final = FakeStore(events=(_event("a"),))

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    assert await CostRunner(store, _client(handler), SETTINGS).step() == 0
    assert store.retried == [("a", "ConnectError")]
    assert store.parked == []


@pytest.mark.asyncio
async def test_a_refused_batch_is_split_so_one_malformed_event_cannot_block_the_rest() -> None:
    store: Final = FakeStore(events=(_event("good"), _event("bad")))

    def handler(request: httpx.Request) -> httpx.Response:
        events = json.loads(request.content)["events"]
        if len(events) > 1 or events[0]["event_id"] == "bad":
            return httpx.Response(400)
        return _results({"event_id": "good", "status": "stored"})

    assert await CostRunner(store, _client(handler), SETTINGS).step() == 1
    assert store.acknowledged == ["good"]
    assert store.parked == [("bad", "HTTP 400")]


@pytest.mark.asyncio
async def test_cost_requests_are_bounded_by_encoded_bytes() -> None:
    store: Final = FakeStore(events=(_event("a", 400), _event("b", 400), _event("c", 400)))
    sizes: Final[list[tuple[int, int]]] = []  # mutable-ok: captures request sizes

    def handler(request: httpx.Request) -> httpx.Response:
        events = json.loads(request.content)["events"]
        sizes.append((len(events), len(request.content)))
        return _results(*({"event_id": event["event_id"], "status": "stored"} for event in events))

    assert await CostRunner(store, _client(handler), SETTINGS).step() == 3
    assert [count for count, _ in sizes] == [2, 1]
    assert store.acknowledged == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_undecodable_events_are_parked_and_never_posted() -> None:
    store: Final = FakeStore(events=(ClaimedEvent(event_id="broken", attempts=0, event="{"),))

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("nothing should be posted")

    assert await CostRunner(store, _client(handler), SETTINGS).step() == 0
    assert store.parked == [("broken", "UndecodableEvent")]


@pytest.mark.parametrize(
    ("status", "acknowledged", "parked", "retried"),
    [
        (204, ["e/messages"], [], []),
        (409, [], [("e/messages", "PayloadStoreDisabled")], []),
        (422, [], [("e/messages", "HTTP 422")], []),
        (503, [], [], [("e/messages", "HTTP 503")]),
    ],
    ids=["stored", "store_disabled", "integrity_failure", "unavailable"],
)
@pytest.mark.asyncio
async def test_bodies_are_acknowledged_only_once_stored(
    status: int, acknowledged: list[str], parked: list[tuple[str, str]], retried: list[tuple[str, str]]
) -> None:
    store: Final = FakeStore(payloads=(_payload("e", '["雪"]'),))
    bodies: Final[list[bytes]] = []  # mutable-ok: captures uploads

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/prefix/internal/ledger/payloads"
        bodies.append(request.content)
        return httpx.Response(status)

    await UploadRunner(store, _client(handler), SETTINGS).step()

    assert store.acknowledged == acknowledged
    assert store.parked == parked
    assert store.retried == retried
    assert store.payload_claims == [SETTINGS.upload_batch_bytes]
    assert "雪".encode() in bodies[0]
    assert json.loads(bodies[0])["content"] == '["雪"]'


@pytest.mark.asyncio
async def test_costs_keep_flowing_while_the_payload_store_hangs_and_stop_releases_both_leases() -> None:
    store: Final = FakeStore(events=(_event("cost"),), payloads=(_payload("body"),))
    never: Final = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/payloads"):
            await never.wait()
        return _results({"event_id": "cost", "status": "stored"})

    client: Final = httpx.AsyncClient(base_url=BASE_URL, transport=httpx.MockTransport(handler))
    stop: Final = asyncio.Event()
    uploads: Final = asyncio.create_task(
        run_loop(
            UploadRunner(store, client, SETTINGS).step, store.release, stop=stop, active=lambda: True, idle_seconds=0.01
        )
    )
    costs: Final = asyncio.create_task(
        run_loop(
            CostRunner(store, client, SETTINGS).step, store.release, stop=stop, active=lambda: True, idle_seconds=0.01
        )
    )

    await asyncio.wait_for(store.acknowledged_event.wait(), timeout=5)
    stop.set()
    uploads.cancel()
    await asyncio.wait_for(asyncio.gather(costs, uploads, return_exceptions=True), timeout=5)

    assert store.acknowledged == ["cost"]
    assert store.releases == 2


@pytest.mark.asyncio
async def test_inactive_runner_claims_nothing() -> None:
    store: Final = FakeStore(events=(_event("a"),))
    stop: Final = asyncio.Event()
    passes: Final[list[bool]] = []  # mutable-ok: records gate checks

    def gate() -> bool:
        passes.append(False)
        if len(passes) == 2:
            stop.set()
        return False

    await asyncio.wait_for(
        run_loop(
            CostRunner(store, _client(lambda request: _results()), SETTINGS).step,
            store.release,
            stop=stop,
            active=gate,
            idle_seconds=0.01,
        ),
        timeout=5,
    )

    assert store.events == (_event("a"),)
    assert store.releases == 1


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (
            httpx.Response(
                200,
                json={"ledger": {"versions": [1], "payload_store": True, "inline_bytes": 4096, "preview_bytes": 512}},
            ),
            Supported,
        ),
        (httpx.Response(200, json={"storage_ready": True}), Unsupported),
        (httpx.Response(200, json={"ledger": {"versions": [2]}}), Unsupported),
        (httpx.Response(503), Unreachable),
        (httpx.Response(200, content=b"<html>"), Unreachable),
    ],
    ids=["ledger_v1", "older_lens", "newer_only", "outage", "not_lens"],
)
@pytest.mark.asyncio
async def test_only_a_lens_that_answers_without_ledger_v1_counts_as_unsupported(
    response: httpx.Response, expected: type
) -> None:
    paths: Final[list[str]] = []  # mutable-ok: captures request paths

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return response

    assert isinstance(await probe_lens(_client(handler)), expected)
    assert paths == ["/prefix/internal/status"]
