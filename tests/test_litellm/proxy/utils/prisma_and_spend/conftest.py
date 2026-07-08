"""Shared fixtures for tests/test_litellm/proxy/utils/prisma_and_spend/.

All fixtures used by PR2 test files live here. Do NOT add fixtures inside
individual test files; if a fixture is missing, add it here and update the
Notion plan.

The PrismaClient is exercised against a fully-mocked Prisma stack: the
``prisma.Prisma`` constructor and the writer/reader wrappers are patched
before PrismaClient.__init__ runs so the init code paths execute without
needing a generated Prisma client or a real database.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from email.message import EmailMessage
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))


VOLATILE_KEYS = frozenset(
    {
        "created_at",
        "updated_at",
        "checked_at",
        "started_at",
        "request_id",
        "id",
        "token",
        "expires",
        "expires_at",
        "litellm_call_id",
        "created",
        "spend",
        "last_refreshed_at",
        "startTime",
        "endTime",
        "salt",
    }
)


def normalize(data: Any, volatile: frozenset = VOLATILE_KEYS) -> Any:
    """Recursively replace values for volatile keys with '<VOLATILE>'."""
    if isinstance(data, dict):
        return {
            k: ("<VOLATILE>" if k in volatile else normalize(v, volatile))
            for k, v in data.items()
        }
    if isinstance(data, list):
        return [normalize(v, volatile) for v in data]
    return data


_PRISMA_TABLES: List[str] = [
    "litellm_verificationtoken",
    "litellm_teamtable",
    "litellm_usertable",
    "litellm_endusertable",
    "litellm_organizationtable",
    "litellm_proxymodeltable",
    "litellm_modeltable",
    "litellm_budgettable",
    "litellm_spendlogs",
    "litellm_config",
    "litellm_usernotifications",
    "litellm_healthchecktable",
    "litellm_dailyuserspend",
    "litellm_dailyteamspend",
    "litellm_dailytagspend",
    "litellm_managed_object_table",
    "litellm_credentialstable",
    "litellm_mcpservertable",
    "litellm_audit_log",
    "litellm_invitationlink",
    "litellm_session_token_table",
    "litellm_passthrough_endpoint_table",
    "litellm_cron_job",
    "litellm_passthrough_logs",
    "litellm_promptstable",
    "litellm_guardrailstable",
    "litellm_managed_files",
    "litellm_mcpusercredentials",
    "litellm_objectpermissiontable",
    "litellm_organizationmembership",
]


def _make_table_mock() -> MagicMock:
    table = MagicMock()
    table.find_unique = AsyncMock(return_value=None)
    table.find_many = AsyncMock(return_value=[])
    table.find_first = AsyncMock(return_value=None)
    table.create = AsyncMock()
    table.create_many = AsyncMock()
    table.update = AsyncMock()
    table.update_many = AsyncMock()
    table.upsert = AsyncMock()
    table.delete = AsyncMock()
    table.delete_many = AsyncMock()
    table.count = AsyncMock(return_value=0)
    table.group_by = AsyncMock(return_value=[])
    table.aggregate = AsyncMock(return_value={})
    return table


@pytest.fixture
def mock_prisma_client() -> MagicMock:
    """Bare ``db`` mock with all common LiteLLM_* tables stubbed.

    Override individual return values in a test::

        mock_prisma_client.db.litellm_usertable.find_unique.return_value = user
    """
    client = MagicMock(name="MockPrismaClient")
    client.db = MagicMock(name="MockPrismaDB")
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    client.health_check = AsyncMock(return_value=[{"?column?": 1}])
    client.proxy_logging_obj = MagicMock()
    client.proxy_logging_obj.failure_handler = AsyncMock()
    client.spend_log_transactions = []
    client._spend_log_transactions_lock = asyncio.Lock()
    client.jsonify_object = lambda data: dict(data)
    client.db.is_connected = MagicMock(return_value=False)
    client.db.connect = AsyncMock()
    client.db.disconnect = AsyncMock()
    client.db.query_raw = AsyncMock(return_value=[{"?column?": 1}])
    client.db.execute_raw = AsyncMock()
    client.db.tx = MagicMock()
    client.db.batch_ = MagicMock()
    for table_name in _PRISMA_TABLES:
        setattr(client.db, table_name, _make_table_mock())
    return client


@pytest.fixture
def mock_dual_cache() -> MagicMock:
    """In-memory DualCache stand-in.

    Sync and async get/set wired against a private dict. Override or read
    ``cache._store`` directly in a test for assertion convenience.
    """
    cache = MagicMock(name="MockDualCache")
    cache._store: Dict[str, Any] = {}

    def _sync_get(key: str, **_: Any) -> Any:
        return cache._store.get(key)

    def _sync_set(key: str, value: Any, **_: Any) -> None:
        cache._store[key] = value

    async def _async_get(key: str, **_: Any) -> Any:
        return cache._store.get(key)

    async def _async_set(key: str, value: Any, **_: Any) -> None:
        cache._store[key] = value

    async def _async_delete(key: str, **_: Any) -> None:
        cache._store.pop(key, None)

    cache.get_cache = MagicMock(side_effect=_sync_get)
    cache.set_cache = MagicMock(side_effect=_sync_set)
    cache.async_get_cache = AsyncMock(side_effect=_async_get)
    cache.async_set_cache = AsyncMock(side_effect=_async_set)
    cache.async_delete_cache = AsyncMock(side_effect=_async_delete)
    return cache


@pytest.fixture
def patched_prisma_import(monkeypatch: pytest.MonkeyPatch) -> Iterator[MagicMock]:
    """Replace ``prisma.Prisma`` and ``PrismaWrapper`` so PrismaClient.__init__
    runs without a generated client. Yields the fake Prisma instance.

    ``prisma`` raises RuntimeError (not AttributeError) for the missing
    ``Prisma`` attribute, so ``monkeypatch.setattr`` can't probe it; assign
    directly and restore in teardown.
    """
    import prisma as _prisma_pkg
    import litellm.proxy.utils as _utils_mod

    fake_prisma = MagicMock(name="FakePrisma")
    fake_prisma.is_connected = MagicMock(return_value=False)
    fake_prisma.connect = AsyncMock()
    fake_prisma.disconnect = AsyncMock()

    fake_prisma_factory = MagicMock(name="FakePrismaFactory", return_value=fake_prisma)
    had_prisma_attr = "Prisma" in _prisma_pkg.__dict__
    previous_prisma_attr = _prisma_pkg.__dict__.get("Prisma")
    _prisma_pkg.Prisma = fake_prisma_factory  # type: ignore[attr-defined]

    fake_wrapper = MagicMock(name="FakePrismaWrapper")
    fake_wrapper.is_connected = MagicMock(return_value=False)
    fake_wrapper.connect = AsyncMock()
    fake_wrapper.disconnect = AsyncMock()
    fake_wrapper.query_raw = AsyncMock(return_value=[{"?column?": 1}])

    def _fake_wrapper_ctor(*args: Any, **kwargs: Any) -> MagicMock:
        return fake_wrapper

    monkeypatch.setattr(_utils_mod, "PrismaWrapper", _fake_wrapper_ctor)
    fake_prisma.__wrapper__ = fake_wrapper
    try:
        yield fake_prisma
    finally:
        if had_prisma_attr:
            _prisma_pkg.Prisma = previous_prisma_attr  # type: ignore[attr-defined]
        else:
            try:
                del _prisma_pkg.Prisma  # type: ignore[attr-defined]
            except AttributeError:
                pass


@pytest.fixture
def prisma_client(
    patched_prisma_import: MagicMock,
    mock_prisma_client: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    """Wired ``PrismaClient`` whose ``db`` attribute is the table mock.

    The init runs through the real code path (testing the constructor's
    config-attribute setup) and is then snapped to the easier-to-assert
    table mock for downstream behavior pinning.
    """
    monkeypatch.delenv("DATABASE_URL_READ_REPLICA", raising=False)
    monkeypatch.delenv("IAM_TOKEN_DB_AUTH", raising=False)
    from litellm.proxy.utils import PrismaClient

    proxy_logging_obj = MagicMock(name="MockProxyLogging")
    proxy_logging_obj.failure_handler = AsyncMock()
    pc = PrismaClient(
        database_url="postgresql://test:test@localhost:5432/test",
        proxy_logging_obj=proxy_logging_obj,
    )
    pc.db = mock_prisma_client.db
    return pc


@dataclass
class FakeClock:
    """Monotonic-time controller for the spend monitor loop.

    Tests advance time via ``clock.advance(seconds)`` while asyncio.sleep
    is replaced with a clock-driven no-op.
    """

    now: float = 0.0
    sleep_calls: List[float] = field(default_factory=list)

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def time(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        self.now += seconds


@pytest.fixture
def fake_clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    """Install a controllable clock + asyncio.sleep replacement."""
    clock = FakeClock()
    monkeypatch.setattr("time.time", clock.time)
    monkeypatch.setattr("time.monotonic", clock.time)

    async def _fast_sleep(seconds: float, *_: Any, **__: Any) -> None:
        clock.sleep_calls.append(seconds)
        clock.now += seconds

    monkeypatch.setattr("asyncio.sleep", _fast_sleep)
    return clock


@pytest.fixture
def make_spend_log_row() -> Callable[..., Dict[str, Any]]:
    """Factory for fake LiteLLM_SpendLogs rows."""

    def _make(
        request_id: str = "req-1",
        spend: float = 0.01,
        model: str = "gpt-4o-mini",
        **overrides: Any,
    ) -> Dict[str, Any]:
        row = {
            "request_id": request_id,
            "spend": spend,
            "model": model,
            "user": "user-1",
            "team_id": "team-1",
            "api_key": "hashed-key",
            "startTime": "2026-06-02T00:00:00Z",
            "endTime": "2026-06-02T00:00:01Z",
            "metadata": {},
        }
        row.update(overrides)
        return row

    return _make


@dataclass
class _SentMessage:
    from_addr: Optional[str]
    to_addrs: Any
    subject: Optional[str]
    body: Optional[str]
    starttls_called: bool
    login_args: Optional[tuple]


@dataclass
class InMemorySMTP:
    """Captures outbound SMTP traffic for ``send_email`` tests."""

    sent: List[_SentMessage] = field(default_factory=list)
    raise_on_send: Optional[Exception] = None

    def server_factory(self) -> Callable[..., Any]:
        outer = self

        class _Conn:
            def __init__(self) -> None:
                self._starttls_called = False
                self._login_args: Optional[tuple] = None

            def __enter__(self) -> "_Conn":
                return self

            def __exit__(self, *exc: Any) -> None:
                return None

            def starttls(self, **kwargs: Any) -> None:
                self._starttls_called = True

            def login(self, user: str, password: str) -> None:
                self._login_args = (user, password)

            def send_message(
                self,
                msg: EmailMessage,
                from_addr: Optional[str] = None,
                to_addrs: Any = None,
            ) -> None:
                if outer.raise_on_send is not None:
                    raise outer.raise_on_send
                body = ""
                for part in msg.walk():
                    if part.get_content_type() == "text/html":
                        body = part.get_payload(decode=False) or ""
                        break
                outer.sent.append(
                    _SentMessage(
                        from_addr=from_addr,
                        to_addrs=to_addrs,
                        subject=msg["Subject"],
                        body=body,
                        starttls_called=self._starttls_called,
                        login_args=self._login_args,
                    )
                )

        def _factory(*args: Any, **kwargs: Any) -> _Conn:
            return _Conn()

        return _factory


@pytest.fixture
def in_memory_smtp(monkeypatch: pytest.MonkeyPatch) -> InMemorySMTP:
    """Patch ``smtplib.SMTP`` and ``smtplib.SMTP_SSL`` to capture sends in memory.

    Override ``smtp.raise_on_send`` to test the SMTP error path.
    """
    smtp = InMemorySMTP()
    factory = smtp.server_factory()
    monkeypatch.setattr("smtplib.SMTP", factory)
    monkeypatch.setattr("smtplib.SMTP_SSL", factory)
    return smtp
