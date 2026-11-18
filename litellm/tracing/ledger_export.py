"""Drains the durable Lens ledger outbox.

Cost events and payload bodies run in separate loops, so a slow payload store never delays accounting
export. A row leaves the outbox only when Lens acknowledges it.
"""

import asyncio
import json
import os
import socket
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from functools import reduce
from typing import Final, Protocol, TypeAlias

import httpx
from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.repositories.lens_export_repository import ClaimedEvent, ClaimedPayload
from litellm.tracing.types import LedgerCapabilities, LedgerResults, LensStatus

LEDGER_VERSION: Final = 1
_STATUS_BYTES: Final = 64 * 1024
_RESULT_BYTES: Final = 16 * 1024 * 1024
_PERMANENT_BATCH: Final = frozenset({400, 413})
_REJECTED_UPLOAD: Final = frozenset({400, 413, 422})
_JSON_HEADERS: Final = {"Content-Type": "application/json"}
_RECORD: Final = TypeAdapter(dict[str, object])
UNREACHABLE_CAPABILITIES: Final = LedgerCapabilities(
    versions=(LEDGER_VERSION,), payload_store=True, inline_bytes=1024 * 1024, preview_bytes=1024 * 1024
)


class CostStore(Protocol):
    async def claim_events(self, limit: int, lease_seconds: int) -> tuple[ClaimedEvent, ...]: ...

    async def acknowledge_events(self, event_ids: Sequence[str]) -> int: ...

    async def retry_events(self, event_ids: Sequence[str], error: str) -> int: ...

    async def park_events(self, event_ids: Sequence[str], error: str) -> int: ...

    async def release(self) -> int: ...


class PayloadStore(Protocol):
    async def claim_payloads(self, limit: int, lease_seconds: int, max_bytes: int) -> tuple[ClaimedPayload, ...]: ...

    async def acknowledge_payloads(self, payloads: Sequence[ClaimedPayload]) -> int: ...

    async def retry_payloads(self, payloads: Sequence[ClaimedPayload], error: str) -> int: ...

    async def park_payloads(self, payloads: Sequence[ClaimedPayload], error: str) -> int: ...

    async def release(self) -> int: ...


def _positive(environ: Mapping[str, str], name: str, default: int) -> int:
    value: Final = environ.get(name, "")
    return int(value) if value.isdigit() and int(value) > 0 else default


@dataclass(frozen=True, slots=True)
class ExportSettings:
    batch_size: int = 100
    lease_seconds: int = 120
    cost_idle_seconds: float = 0.25
    upload_idle_seconds: float = 2.0
    cost_batch_bytes: int = 8 * 1024 * 1024
    upload_batch_bytes: int = 32 * 1024 * 1024
    probe_seconds: float = 30.0
    stop_seconds: float = 5.0

    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ) -> "ExportSettings":
        return cls(
            batch_size=min(_positive(environ, "LITELLM_LENS_EXPORT_BATCH_SIZE", 100), 1000),
            lease_seconds=_positive(environ, "LITELLM_LENS_EXPORT_LEASE_SECONDS", 120),
            cost_idle_seconds=max(_positive(environ, "LITELLM_LENS_EXPORT_IDLE_MS", 250), 50) / 1000,
            upload_batch_bytes=_positive(environ, "LITELLM_LENS_UPLOAD_BATCH_BYTES", 32 * 1024 * 1024),
        )


def exporter_owner(role: str) -> str:
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}:{role}"


@dataclass(frozen=True, slots=True)
class Supported:
    capabilities: LedgerCapabilities


@dataclass(frozen=True, slots=True)
class Unsupported:
    """Lens answered its status endpoint without ledger version 1: an older server, not an outage."""


@dataclass(frozen=True, slots=True)
class Unreachable:
    reason: str


Probe: TypeAlias = Supported | Unsupported | Unreachable


async def probe_lens(client: httpx.AsyncClient) -> Probe:
    try:
        response: Final = await client.get("internal/status", timeout=5)
    except httpx.HTTPError as error:
        return Unreachable(type(error).__name__)
    if response.status_code != 200:
        return Unreachable(f"HTTP {response.status_code}")
    if len(response.content) > _STATUS_BYTES:
        return Unreachable("StatusTooLarge")
    try:
        ledger: Final = LensStatus.model_validate_json(response.content).ledger
    except ValidationError:
        return Unreachable("InvalidStatus")
    if ledger is None or LEDGER_VERSION not in ledger.versions:
        return Unsupported()
    return Supported(ledger)


def _encode(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()


def _decoded(event: ClaimedEvent) -> Mapping[str, object] | None:
    try:
        return _RECORD.validate_json(event.event)
    except ValidationError:
        return None


_Claimed: TypeAlias = tuple[ClaimedEvent, Mapping[str, object]]


def _event_bytes(batch: Sequence[_Claimed]) -> int:
    return sum(len(event.event.encode()) for event, _ in batch)


def _batches(events: Sequence[_Claimed], max_bytes: int) -> tuple[tuple[_Claimed, ...], ...]:
    def add(batches: tuple[tuple[_Claimed, ...], ...], item: _Claimed) -> tuple[tuple[_Claimed, ...], ...]:
        if batches and _event_bytes((*batches[-1], item)) <= max_bytes:
            return (*batches[:-1], (*batches[-1], item))
        return (*batches, (item,))

    return reduce(add, events, ())


@dataclass(frozen=True, slots=True)
class _Results:
    results: LedgerResults


@dataclass(frozen=True, slots=True)
class _Failed:
    error: str
    permanent: bool


class CostRunner:
    def __init__(self, store: CostStore, client: httpx.AsyncClient, settings: ExportSettings) -> None:
        self._store: Final = store
        self._client: Final = client
        self._settings: Final = settings

    async def _post(self, events: Sequence[tuple[ClaimedEvent, Mapping[str, object]]]) -> _Results | _Failed:
        body: Final = {
            "version": LEDGER_VERSION,
            "events": [{"event_id": event.event_id, "record": record} for event, record in events],
        }
        try:
            response: Final = await self._client.post(
                "internal/ledger", content=_encode(body), headers=_JSON_HEADERS, timeout=30
            )
        except httpx.HTTPError as error:
            return _Failed(type(error).__name__, permanent=False)
        if response.status_code != 200:
            return _Failed(f"HTTP {response.status_code}", permanent=response.status_code in _PERMANENT_BATCH)
        if len(response.content) > _RESULT_BYTES:
            return _Failed("ResponseTooLarge", permanent=False)
        try:
            return _Results(LedgerResults.model_validate_json(response.content))
        except ValidationError:
            return _Failed("InvalidResponse", permanent=False)

    async def _settle(self, events: Sequence[tuple[ClaimedEvent, Mapping[str, object]]]) -> int:
        outcome: Final = await self._post(events)
        ids: Final = tuple(event.event_id for event, _ in events)
        match outcome:
            case _Failed(error=error, permanent=True) if len(events) == 1:
                await self._store.park_events(ids, error)
                verbose_proxy_logger.error("Lens refused ledger event %s (%s); it is parked", ids[0], error)
                return 0
            case _Failed(permanent=True):
                return sum([await self._settle((event,)) for event in events])
            case _Failed(error=error):
                await self._store.retry_events(ids, error)
                return 0
            case _Results(results=results):
                statuses: Final = {result.event_id: result.status for result in results.results}
                stored: Final = tuple(event_id for event_id in ids if statuses.get(event_id) == "stored")
                rejected: Final = tuple(event_id for event_id in ids if statuses.get(event_id) == "rejected")
                missing: Final = tuple(event_id for event_id in ids if event_id not in statuses)
                await self._store.acknowledge_events(stored)
                await self._store.park_events(rejected, "Rejected")
                await self._store.retry_events(missing, "MissingResult")
                if rejected:
                    verbose_proxy_logger.error("Lens rejected %d ledger events; they are parked", len(rejected))
                return len(stored)

    async def step(self) -> int:
        claimed: Final = await self._store.claim_events(self._settings.batch_size, self._settings.lease_seconds)
        decoded: Final = tuple((event, _decoded(event)) for event in claimed)
        await self._store.park_events(
            tuple(event.event_id for event, record in decoded if record is None), "UndecodableEvent"
        )
        readable: Final = tuple((event, record) for event, record in decoded if record is not None)
        return sum([await self._settle(batch) for batch in _batches(readable, self._settings.cost_batch_bytes)])


class UploadRunner:
    """Uploads committed bodies. A 409 means Lens has no payload store: the body is parked only once a probe has
    positively reported that, and is otherwise retried, so a restart without the store never drops it."""

    def __init__(
        self,
        store: PayloadStore,
        client: httpx.AsyncClient,
        settings: ExportSettings,
        store_disabled: Callable[[], bool],
    ) -> None:
        self._store: Final = store
        self._client: Final = client
        self._settings: Final = settings
        self._store_disabled: Final = store_disabled

    async def _upload(self, payload: ClaimedPayload) -> int | str:
        try:
            response: Final = await self._client.post(
                "internal/ledger/payloads",
                content=_encode({"version": LEDGER_VERSION, **payload.model_dump(exclude={"attempts"})}),
                headers=_JSON_HEADERS,
                timeout=60,
            )
        except httpx.HTTPError as error:
            return type(error).__name__
        return response.status_code

    async def step(self) -> int:
        claimed: Final = await self._store.claim_payloads(
            self._settings.batch_size, self._settings.lease_seconds, self._settings.upload_batch_bytes
        )
        outcomes: Final = tuple([(payload, await self._upload(payload)) for payload in claimed])
        uploaded: Final = tuple(payload for payload, outcome in outcomes if outcome == 204)
        disabled: Final = tuple(payload for payload, outcome in outcomes if outcome == 409 and self._store_disabled())
        rejected: Final = tuple(
            (payload, outcome)
            for payload, outcome in outcomes
            if isinstance(outcome, int) and outcome in _REJECTED_UPLOAD
        )
        failed: Final = tuple(
            (payload, outcome)
            for payload, outcome in outcomes
            if outcome != 204
            and not (outcome == 409 and self._store_disabled())
            and not (isinstance(outcome, int) and outcome in _REJECTED_UPLOAD)
        )
        await self._store.acknowledge_payloads(uploaded)
        await self._store.park_payloads(disabled, "PayloadStoreDisabled")
        for payload, outcome in rejected:
            await self._store.park_payloads((payload,), f"HTTP {outcome}")
        for payload, outcome in failed:
            await self._store.retry_payloads((payload,), outcome if isinstance(outcome, str) else f"HTTP {outcome}")
        if disabled or rejected:
            verbose_proxy_logger.error(
                "Lens did not store %d payload bodies (%d with no payload store); they are parked for requeue",
                len(disabled) + len(rejected),
                len(disabled),
            )
        return len(uploaded)


async def _idle(stop: asyncio.Event, seconds: float) -> None:
    with suppress(asyncio.TimeoutError):
        await asyncio.wait_for(stop.wait(), timeout=seconds)


async def _attempt(step: Callable[[], Awaitable[int]]) -> int:
    try:
        return await step()
    except asyncio.CancelledError:
        raise
    except Exception as error:  # noqa: BLE001  # the outbox is durable; a failed pass retries next tick
        verbose_proxy_logger.warning("Lens ledger export pass failed, retrying: %s", type(error).__name__)
        return 0


async def run_loop(
    step: Callable[[], Awaitable[int]],
    release: Callable[[], Awaitable[int]],
    *,
    stop: asyncio.Event,
    active: Callable[[], bool],
    idle_seconds: float,
) -> None:
    """Runs `step` until stopped; on stop or cancellation it releases this runner's leases."""
    try:
        while not stop.is_set():
            if not active() or await _attempt(step) == 0:
                await _idle(stop, idle_seconds)
    finally:
        with suppress(Exception, asyncio.CancelledError):
            await asyncio.wait_for(release(), timeout=2)
