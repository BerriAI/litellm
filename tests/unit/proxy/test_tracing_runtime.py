import asyncio
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Final

import httpx
import pytest

import litellm
from litellm.proxy.tracing_runtime import LedgerExportOptions, manage_tracing
from litellm.tracing.exporter import LensExporter
from litellm.tracing.ledger_export import UNREACHABLE_CAPABILITIES, ExportSettings
from litellm.tracing.types import LedgerCapabilities

LEDGER_STATUS: Final = httpx.Response(
    200, json={"ledger": {"versions": [1], "payload_store": False, "inline_bytes": 4096, "preview_bytes": 4096}}
)
OLD_LENS: Final = httpx.Response(200, json={"storage_ready": True, "protocol_version": 7})
OUTAGE: Final = httpx.Response(503)
FAST: Final = ExportSettings(probe_seconds=0.01, cost_idle_seconds=0.01, upload_idle_seconds=0.01, stop_seconds=0.2)


@dataclass
class RecordingOutbox:
    statements: list[str] = field(default_factory=list)  # mutable-ok: records statements for assertions
    hang: asyncio.Event | None = None

    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]:
        if self.hang is not None:
            await self.hang.wait()
        return []

    async def execute_raw(self, query: str, *args: object) -> int:
        self.statements.append(query)
        return 0


@dataclass
class RecordingDatabase:
    outbox: RecordingOutbox

    @asynccontextmanager
    async def tx(self, *, timeout: timedelta) -> AsyncIterator[RecordingOutbox]:
        yield self.outbox


def _legacy_registered() -> bool:
    return any(isinstance(callback, LensExporter) for callback in litellm.callbacks)


@dataclass
class Publications:
    values: list[LedgerCapabilities | None] = field(default_factory=list)  # mutable-ok: records publications
    changed: asyncio.Event = field(default_factory=asyncio.Event)

    def publish(self, capabilities: LedgerCapabilities | None) -> None:
        self.values.append(capabilities)
        self.changed.set()

    async def until(self, expected: LedgerCapabilities | None, count: int = 1) -> None:
        async def reached() -> None:
            while self.values.count(expected) < count:
                self.changed.clear()
                await self.changed.wait()

        await asyncio.wait_for(reached(), timeout=5)


def _options(publications: Publications, outbox: RecordingOutbox, *, enabled: bool = True) -> LedgerExportOptions:
    return LedgerExportOptions(
        enabled=enabled,
        database=RecordingDatabase(outbox),  # pyright: ignore[reportArgumentType]  # recording outbox fake
        publish=publications.publish,
        settings=FAST,
    )


def _factory(responses: list[httpx.Response]) -> Callable[[object], httpx.AsyncClient]:
    def respond(request: httpx.Request) -> httpx.Response:
        return responses[0] if len(responses) == 1 else responses.pop(0)

    def client_factory(connection: object) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url="http://lens.test", transport=httpx.MockTransport(respond))

    return client_factory


@pytest.fixture
def lens_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_LENS_URL", "http://lens.test")
    monkeypatch.setenv("LITELLM_LENS_SERVICE_TOKEN", "test-service-token-with-32-characters")


@pytest.mark.asyncio
async def test_an_outage_at_boot_keeps_the_durable_ledger_as_the_only_producer(lens_env: None) -> None:
    publications: Final = Publications()

    async with manage_tracing(
        True, client_factory=_factory([OUTAGE]), ledger=_options(publications, RecordingOutbox())
    ):
        await publications.until(UNREACHABLE_CAPABILITIES, count=2)
        assert not _legacy_registered()

    assert None not in publications.values


@pytest.mark.asyncio
async def test_a_positively_identified_older_lens_hands_production_to_the_callback_and_back(lens_env: None) -> None:
    publications: Final = Publications()
    supported: Final = LedgerCapabilities(versions=(1,), payload_store=False, inline_bytes=4096, preview_bytes=4096)

    async with manage_tracing(
        True, client_factory=_factory([OLD_LENS, LEDGER_STATUS]), ledger=_options(publications, RecordingOutbox())
    ):
        await publications.until(None)
        await publications.until(supported)
        assert not _legacy_registered()

    assert publications.values[0] == UNREACHABLE_CAPABILITIES


@pytest.mark.asyncio
async def test_an_older_lens_with_the_flag_on_keeps_the_callback_registered(lens_env: None) -> None:
    publications: Final = Publications()

    async with manage_tracing(
        True, client_factory=_factory([OLD_LENS]), ledger=_options(publications, RecordingOutbox())
    ):
        await publications.until(None, count=2)
        assert _legacy_registered()

    assert not _legacy_registered()


@pytest.mark.asyncio
async def test_with_the_flag_off_the_callback_produces_and_committed_rows_still_drain(lens_env: None) -> None:
    publications: Final = Publications()
    outbox: Final = RecordingOutbox()

    async with manage_tracing(
        True, client_factory=_factory([LEDGER_STATUS]), ledger=_options(publications, outbox, enabled=False)
    ):
        await publications.until(None, count=2)
        assert _legacy_registered()

    assert set(publications.values) == {None}


@pytest.mark.asyncio
async def test_shutdown_cancels_a_stuck_runner_and_releases_its_leases(lens_env: None) -> None:
    publications: Final = Publications()
    outbox: Final = RecordingOutbox(hang=asyncio.Event())

    async def lifecycle() -> None:
        async with manage_tracing(
            True, client_factory=_factory([LEDGER_STATUS]), ledger=_options(publications, outbox)
        ):
            await publications.until(UNREACHABLE_CAPABILITIES)

    await asyncio.wait_for(lifecycle(), timeout=5)

    assert sum("SET lease_owner = NULL" in statement for statement in outbox.statements) == 4


@pytest.mark.asyncio
async def test_bodies_parked_while_the_store_was_off_are_requeued_when_it_returns(lens_env: None) -> None:
    publications: Final = Publications()
    outbox: Final = RecordingOutbox()
    with_store: Final = LedgerCapabilities(versions=(1,), payload_store=True, inline_bytes=4096, preview_bytes=512)
    store_on: Final = httpx.Response(
        200, json={"ledger": {"versions": [1], "payload_store": True, "inline_bytes": 4096, "preview_bytes": 512}}
    )

    async with manage_tracing(
        True, client_factory=_factory([LEDGER_STATUS, store_on]), ledger=_options(publications, outbox)
    ):
        await publications.until(with_store)

    requeues: Final = [statement for statement in outbox.statements if "SET parked_at = NULL" in statement]
    assert len(requeues) == 1
    assert "LiteLLM_LensLedgerPayload" in requeues[0]
