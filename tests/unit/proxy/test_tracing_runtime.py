from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Final

import httpx
import pytest

import litellm
from litellm.proxy.tracing_runtime import LedgerExportOptions, manage_tracing
from litellm.tracing.exporter import LensExporter
from litellm.tracing.ledger_export import ExportSettings
from litellm.tracing.types import LedgerCapabilities

LEDGER_STATUS: Final = {"ledger": {"versions": [1], "payload_store": False, "inline_bytes": 4096, "preview_bytes": 4096}}


class EmptyOutbox:
    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]:
        return []

    async def execute_raw(self, query: str, *args: object) -> int:
        return 0


class EmptyDatabase:
    @asynccontextmanager
    async def tx(self, *, timeout: timedelta) -> AsyncIterator[EmptyOutbox]:
        yield EmptyOutbox()


def _legacy_registered() -> bool:
    return any(isinstance(callback, LensExporter) for callback in litellm.callbacks)


@pytest.mark.parametrize(
    ("flag", "status", "legacy", "published"),
    [
        (True, LEDGER_STATUS, False, 4096),
        (True, {"storage_ready": True}, True, None),
        (False, LEDGER_STATUS, True, None),
    ],
    ids=["ledger_active", "lens_without_ledger", "flag_off_drains_only"],
)
@pytest.mark.asyncio
async def test_each_request_has_exactly_one_lens_producer(
    monkeypatch: pytest.MonkeyPatch,
    flag: bool,
    status: Mapping[str, object],
    legacy: bool,
    published: int | None,
) -> None:
    monkeypatch.setenv("LITELLM_LENS_URL", "http://lens.test")
    monkeypatch.setenv("LITELLM_LENS_SERVICE_TOKEN", "test-service-token-with-32-characters")
    publications: Final[list[LedgerCapabilities | None]] = []  # mutable-ok: records published contracts

    def client_factory(connection: object) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url="http://lens.test",
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json=status)),
        )

    options: Final = LedgerExportOptions(
        enabled=flag,
        database=EmptyDatabase(),  # pyright: ignore[reportArgumentType]  # empty outbox fake
        publish=publications.append,
        settings=ExportSettings(),
    )

    async with manage_tracing(True, client_factory=client_factory, ledger=options):
        assert _legacy_registered() is legacy
        active: Final = publications[0] if publications else None
        assert (active.inline_bytes if active is not None else None) == published

    assert not _legacy_registered()
    assert publications[-1:] in ([], [None])
