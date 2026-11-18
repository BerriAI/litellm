import asyncio
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Final

import httpx
import pytest

from litellm.repositories.lens_export_repository import ClaimedEvent, ClaimedPayload
from litellm.tracing.ledger_export import ExportSettings, LensLedgerExporter, ledger_capabilities

SETTINGS: Final = ExportSettings(batch_size=10, max_attempts=3, retention_days=2, cleanup_seconds=60)


@dataclass
class FakeStore:
    events: tuple[ClaimedEvent, ...] = ()
    payloads: tuple[ClaimedPayload, ...] = ()
    acknowledged: list[str] = field(default_factory=list)  # mutable-ok: records calls for assertions
    retried: list[tuple[str, str]] = field(default_factory=list)  # mutable-ok: records calls for assertions
    parked: list[tuple[str, str]] = field(default_factory=list)  # mutable-ok: records calls for assertions
    cleanups: list[int] = field(default_factory=list)  # mutable-ok: records calls for assertions

    async def claim_events(self, limit: int, lease_seconds: int) -> tuple[ClaimedEvent, ...]:
        claimed, self.events = self.events[:limit], self.events[limit:]
        return claimed

    async def acknowledge_events(self, event_ids: Sequence[str]) -> int:
        self.acknowledged.extend(event_ids)
        return len(event_ids)

    async def retry_events(self, event_ids: Sequence[str], error: str, max_attempts: int) -> int:
        self.retried.extend((event_id, error) for event_id in event_ids)
        return len(event_ids)

    async def park_events(self, event_ids: Sequence[str], error: str) -> int:
        self.parked.extend((event_id, error) for event_id in event_ids)
        return len(event_ids)

    async def claim_payloads(self, limit: int, lease_seconds: int) -> tuple[ClaimedPayload, ...]:
        claimed, self.payloads = self.payloads[:limit], self.payloads[limit:]
        return claimed

    async def acknowledge_payloads(self, payloads: Sequence[ClaimedPayload]) -> int:
        self.acknowledged.extend(f"{payload.event_id}/{payload.field}" for payload in payloads)
        return len(payloads)

    async def retry_payloads(self, payloads: Sequence[ClaimedPayload], error: str, max_attempts: int) -> int:
        self.retried.extend((f"{payload.event_id}/{payload.field}", error) for payload in payloads)
        return len(payloads)

    async def park_payloads(self, payloads: Sequence[ClaimedPayload], error: str) -> int:
        self.parked.extend((f"{payload.event_id}/{payload.field}", error) for payload in payloads)
        return len(payloads)

    async def cleanup(self, retention_days: int) -> int:
        self.cleanups.append(retention_days)
        return 0


def _event(event_id: str, record: object = None) -> ClaimedEvent:
    return ClaimedEvent(
        event_id=event_id, attempts=0, event=json.dumps(record if record is not None else {"request_id": event_id})
    )


def _payload(event_id: str) -> ClaimedPayload:
    return ClaimedPayload(
        event_id=event_id,
        field="messages",
        team_id="team",
        api_key="hash",
        event_time_ms=1,
        content="[]",
        bytes=2,
        sha256="0" * 64,
        attempts=0,
    )


def _exporter(
    store: FakeStore, handler: Callable[[httpx.Request], httpx.Response], clock: Callable[[], float] = lambda: 0.0
) -> LensLedgerExporter:
    client: Final = httpx.AsyncClient(base_url="http://lens", transport=httpx.MockTransport(handler))
    return LensLedgerExporter(store, client, SETTINGS, clock)


def _results(*results: dict[str, object]) -> httpx.Response:
    return httpx.Response(200, json={"version": 1, "results": list(results)})


@pytest.mark.asyncio
async def test_each_event_is_acknowledged_parked_or_retried_by_its_own_result() -> None:
    store: Final = FakeStore(events=(_event("stored"), _event("rejected"), _event("missing")))
    sent: Final[list[object]] = []  # mutable-ok: captures the posted batch

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return _results(
            {"event_id": "stored", "status": "stored"},
            {"event_id": "rejected", "status": "rejected", "reason": "missing_event_time"},
        )

    assert await _exporter(store, handler).export_events() == 1
    assert sent == [
        {
            "version": 1,
            "events": [
                {"event_id": "stored", "record": {"request_id": "stored"}},
                {"event_id": "rejected", "record": {"request_id": "rejected"}},
                {"event_id": "missing", "record": {"request_id": "missing"}},
            ],
        }
    ]
    assert store.acknowledged == ["stored"]
    assert store.parked == [("rejected", "Rejected")]
    assert store.retried == [("missing", "MissingResult")]


@pytest.mark.parametrize(
    "respond",
    [
        lambda request: httpx.Response(503),
        lambda request: httpx.Response(422, json={"code": "unsupported_version"}),
        lambda request: httpx.Response(200, content=b"not json"),
    ],
    ids=["unavailable", "older_lens", "invalid_response"],
)
@pytest.mark.asyncio
async def test_transient_failures_retry_the_whole_claim_without_acknowledging(
    respond: Callable[[httpx.Request], httpx.Response],
) -> None:
    store: Final = FakeStore(events=(_event("a"), _event("b")))

    assert await _exporter(store, respond).export_events() == 0
    assert store.acknowledged == []
    assert [event_id for event_id, _ in store.retried] == ["a", "b"]


@pytest.mark.asyncio
async def test_network_errors_retry_the_claim() -> None:
    store: Final = FakeStore(events=(_event("a"),))

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    assert await _exporter(store, handler).export_events() == 0
    assert store.retried == [("a", "ConnectError")]


@pytest.mark.asyncio
async def test_a_refused_batch_is_split_so_one_bad_event_cannot_block_the_rest() -> None:
    store: Final = FakeStore(events=(_event("good"), _event("bad")))

    def handler(request: httpx.Request) -> httpx.Response:
        events = json.loads(request.content)["events"]
        if len(events) > 1 or events[0]["event_id"] == "bad":
            return httpx.Response(400)
        return _results({"event_id": "good", "status": "stored"})

    assert await _exporter(store, handler).export_events() == 1
    assert store.acknowledged == ["good"]
    assert store.parked == [("bad", "HTTP 400")]


@pytest.mark.asyncio
async def test_undecodable_events_are_parked_and_never_posted() -> None:
    store: Final = FakeStore(events=(ClaimedEvent(event_id="broken", attempts=0, event="{"),))

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("nothing should be posted")

    assert await _exporter(store, handler).export_events() == 0
    assert store.parked == [("broken", "UndecodableEvent")]


@pytest.mark.parametrize(
    ("status", "acknowledged", "parked", "retried"),
    [
        (204, ["e/messages"], [], []),
        (409, ["e/messages"], [], []),
        (422, [], [("e/messages", "Rejected")], []),
        (503, [], [], [("e/messages", "HTTP 503")]),
    ],
    ids=["stored", "store_disabled", "integrity_failure", "unavailable"],
)
@pytest.mark.asyncio
async def test_payload_uploads_settle_independently_of_costs(
    status: int, acknowledged: list[str], parked: list[tuple[str, str]], retried: list[tuple[str, str]]
) -> None:
    store: Final = FakeStore(events=(_event("cost"),), payloads=(_payload("e"),))
    bodies: Final[list[object]] = []  # mutable-ok: captures uploads

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(status)

    await _exporter(store, handler).export_payloads()

    assert store.acknowledged == acknowledged
    assert store.parked == parked
    assert store.retried == retried
    assert store.events == (_event("cost"),)
    assert bodies == [
        {
            "version": 1,
            "event_id": "e",
            "field": "messages",
            "team_id": "team",
            "api_key": "hash",
            "event_time_ms": 1,
            "content": "[]",
            "bytes": 2,
            "sha256": "0" * 64,
        }
    ]


@pytest.mark.asyncio
async def test_cleanup_runs_once_per_interval_on_the_injected_clock() -> None:
    store: Final = FakeStore()
    now: Final = [0.0]  # mutable-ok: controlled clock
    exporter: Final = _exporter(store, lambda request: _results(), clock=lambda: now[0])

    await exporter.cleanup_due()
    await exporter.cleanup_due()
    now[0] = 60.0
    await exporter.cleanup_due()

    assert store.cleanups == [2, 2]


@pytest.mark.asyncio
async def test_run_drains_one_pass_and_stops_when_signalled() -> None:
    store: Final = FakeStore(events=(_event("a"),))
    stop: Final = asyncio.Event()

    def handler(request: httpx.Request) -> httpx.Response:
        stop.set()
        return _results({"event_id": "a", "status": "stored"})

    await asyncio.wait_for(_exporter(store, handler).run(stop), timeout=5)

    assert store.acknowledged == ["a"]
    assert store.cleanups == [2]


@pytest.mark.parametrize(
    ("response", "expected_inline"),
    [
        (httpx.Response(200, json={"ledger": {"versions": [1], "payload_store": True, "inline_bytes": 4096, "preview_bytes": 512}}), 4096),
        (httpx.Response(200, json={"storage_ready": True}), None),
        (httpx.Response(200, json={"ledger": {"versions": [2]}}), None),
        (httpx.Response(503), None),
    ],
    ids=["ledger_v1", "older_lens", "newer_only", "unavailable"],
)
@pytest.mark.asyncio
async def test_ledger_export_activates_only_against_a_lens_that_advertises_version_one(
    response: httpx.Response, expected_inline: int | None
) -> None:
    client: Final = httpx.AsyncClient(base_url="http://lens", transport=httpx.MockTransport(lambda request: response))

    capabilities: Final = await ledger_capabilities(client)

    assert (capabilities.inline_bytes if capabilities is not None else None) == expected_inline
