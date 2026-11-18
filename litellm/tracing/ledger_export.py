"""Drains the durable Lens ledger outbox: cost events and payload bodies are acknowledged independently."""

import asyncio
import json
import os
import socket
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Final, Protocol

import httpx
from pydantic import ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.repositories.lens_export_repository import ClaimedEvent, ClaimedPayload
from litellm.tracing.types import LedgerCapabilities, LedgerResults, LensStatus

LEDGER_VERSION: Final = 1
_STATUS_BYTES: Final = 64 * 1024
_RESULT_BYTES: Final = 16 * 1024 * 1024


class LensExportStore(Protocol):
    async def claim_events(self, limit: int, lease_seconds: int) -> tuple[ClaimedEvent, ...]: ...

    async def acknowledge_events(self, event_ids: Sequence[str]) -> int: ...

    async def retry_events(self, event_ids: Sequence[str], error: str, max_attempts: int) -> int: ...

    async def park_events(self, event_ids: Sequence[str], error: str) -> int: ...

    async def claim_payloads(self, limit: int, lease_seconds: int) -> tuple[ClaimedPayload, ...]: ...

    async def acknowledge_payloads(self, payloads: Sequence[ClaimedPayload]) -> int: ...

    async def retry_payloads(self, payloads: Sequence[ClaimedPayload], error: str, max_attempts: int) -> int: ...

    async def park_payloads(self, payloads: Sequence[ClaimedPayload], error: str) -> int: ...

    async def cleanup(self, retention_days: int) -> int: ...


def _positive(environ: Mapping[str, str], name: str, default: int) -> int:
    value: Final = environ.get(name, "")
    return int(value) if value.isdigit() and int(value) > 0 else default


@dataclass(frozen=True, slots=True)
class ExportSettings:
    batch_size: int = 100
    lease_seconds: int = 120
    max_attempts: int = 12
    retention_days: int = 7
    idle_seconds: float = 2.0
    cleanup_seconds: float = 3600.0

    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ) -> "ExportSettings":
        return cls(
            batch_size=min(_positive(environ, "LITELLM_LENS_EXPORT_BATCH_SIZE", 100), 1000),
            lease_seconds=_positive(environ, "LITELLM_LENS_EXPORT_LEASE_SECONDS", 120),
            max_attempts=_positive(environ, "LITELLM_LENS_EXPORT_MAX_ATTEMPTS", 12),
            retention_days=_positive(environ, "LITELLM_LENS_EXPORT_RETENTION_DAYS", 7),
        )


def exporter_owner() -> str:
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


async def ledger_capabilities(client: httpx.AsyncClient) -> LedgerCapabilities | None:
    try:
        response: Final = await client.get("/internal/status", timeout=5)
    except httpx.HTTPError:
        return None
    if response.status_code != 200 or len(response.content) > _STATUS_BYTES:
        return None
    try:
        ledger: Final = LensStatus.model_validate_json(response.content).ledger
    except ValidationError:
        return None
    return ledger if ledger is not None and LEDGER_VERSION in ledger.versions else None


@dataclass(frozen=True, slots=True)
class _Results:
    results: LedgerResults


@dataclass(frozen=True, slots=True)
class _Failed:
    error: str
    permanent: bool


def _decoded(event: ClaimedEvent) -> Mapping[str, object] | None:
    try:
        record: Final = json.loads(event.event)
    except ValueError:
        return None
    return record if isinstance(record, dict) else None


class LensLedgerExporter:
    def __init__(
        self,
        store: LensExportStore,
        client: httpx.AsyncClient,
        settings: ExportSettings,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._store: Final = store
        self._client: Final = client
        self._settings: Final = settings
        self._clock: Final = clock
        self._next_cleanup = clock()  # rebind-ok: periodic cleanup deadline

    async def _post(self, events: Sequence[tuple[ClaimedEvent, Mapping[str, object]]]) -> _Results | _Failed:
        body: Final = {
            "version": LEDGER_VERSION,
            "events": [{"event_id": event.event_id, "record": record} for event, record in events],
        }
        try:
            response: Final = await self._client.post("/internal/ledger", json=body, timeout=30)
        except httpx.HTTPError as error:
            return _Failed(type(error).__name__, permanent=False)
        if response.status_code != 200:
            return _Failed(f"HTTP {response.status_code}", permanent=response.status_code in (400, 413))
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
                return 0
            case _Failed(permanent=True):
                return sum([await self._settle((event,)) for event in events])
            case _Failed(error=error):
                await self._store.retry_events(ids, error, self._settings.max_attempts)
                return 0
            case _Results(results=results):
                statuses: Final = {result.event_id: result for result in results.results}
                stored: Final = tuple(i for i in ids if i in statuses and statuses[i].status == "stored")
                rejected: Final = tuple(i for i in ids if i in statuses and statuses[i].status == "rejected")
                missing: Final = tuple(i for i in ids if i not in statuses)
                await self._store.acknowledge_events(stored)
                await self._store.park_events(rejected, "Rejected")
                await self._store.retry_events(missing, "MissingResult", self._settings.max_attempts)
                if rejected:
                    verbose_proxy_logger.error("Lens rejected %d ledger events; they are parked", len(rejected))
                return len(stored)

    async def export_events(self) -> int:
        claimed: Final = await self._store.claim_events(self._settings.batch_size, self._settings.lease_seconds)
        decoded: Final = tuple((event, _decoded(event)) for event in claimed)
        undecodable: Final = tuple(event.event_id for event, record in decoded if record is None)
        await self._store.park_events(undecodable, "UndecodableEvent")
        readable: Final = tuple((event, record) for event, record in decoded if record is not None)
        return await self._settle(readable) if readable else 0

    async def _upload(self, payload: ClaimedPayload) -> str | None:
        try:
            response: Final = await self._client.post(
                "/internal/ledger/payloads",
                json={"version": LEDGER_VERSION, **payload.model_dump(exclude={"attempts"})},
                timeout=60,
            )
        except httpx.HTTPError as error:
            return type(error).__name__
        return None if response.status_code == 204 else f"HTTP {response.status_code}"

    async def export_payloads(self) -> int:
        claimed: Final = await self._store.claim_payloads(self._settings.batch_size, self._settings.lease_seconds)
        outcomes: Final = tuple([(payload, await self._upload(payload)) for payload in claimed])
        uploaded: Final = tuple(payload for payload, error in outcomes if error is None)
        disabled: Final = tuple(payload for payload, error in outcomes if error == "HTTP 409")
        permanent: Final = tuple(payload for payload, error in outcomes if error in ("HTTP 400", "HTTP 422"))
        failed: Final = tuple(
            (payload, error)
            for payload, error in outcomes
            if error is not None and error not in ("HTTP 409", "HTTP 400", "HTTP 422")
        )
        await self._store.acknowledge_payloads(uploaded + disabled)
        await self._store.park_payloads(permanent, "Rejected")
        for payload, error in failed:
            await self._store.retry_payloads((payload,), error, self._settings.max_attempts)
        return len(uploaded)

    async def cleanup_due(self) -> int:
        if self._clock() < self._next_cleanup:
            return 0
        self._next_cleanup = self._clock() + self._settings.cleanup_seconds
        return await self._store.cleanup(self._settings.retention_days)

    async def run_once(self) -> int:
        moved: Final = await self.export_events() + await self.export_payloads()
        await self.cleanup_due()
        return moved

    async def run(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                moved = await self.run_once()  # rebind-ok: loop progress
            except Exception as error:  # noqa: BLE001  # the outbox is durable; a failed pass retries next tick
                verbose_proxy_logger.warning("Lens ledger export pass failed, retrying: %s", type(error).__name__)
                moved = 0  # rebind-ok: loop progress
            if moved == 0:
                with suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(stop.wait(), timeout=self._settings.idle_seconds)
