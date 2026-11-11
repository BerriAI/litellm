"""Coordination between planned Prisma engine restarts and reconnect paths.

Covers the fix for https://github.com/BerriAI/litellm/issues/29176 — an RDS
IAM token refresh recreates the Prisma client (killing the query-engine
subprocess), and the engine-death watcher / in-flight transport-error
retries must not treat that planned restart as a crash and recreate the
client a second time.

A planned restart is also invisible to the database itself, so a ``SELECT 1``
health probe that races the kill/connect window fails with a transport error
against the engine's local HTTP port. That failure must not be reported as a
``db_exceptions`` DB failure, or every IAM refresh cycle raises a false alarm.

Symbols pinned here:
  - ``PrismaWrapper._expected_engine_deaths``
  - ``PrismaWrapper._engine_generation``
  - ``PrismaWrapper.engine_generation``
  - ``PrismaWrapper.wait_for_planned_engine_replacement``
  - ``PrismaWrapper.on_engine_replaced``
  - ``PrismaWrapper.recreate_prisma_client`` (expected_generation guard)
  - ``PrismaWrapper._safe_refresh_token`` (refresh coalescing)
  - ``RoutingPrismaWrapper.recreate_prisma_client`` (guard forwarding)
  - ``PrismaClient.health_check`` (planned-replacement alert suppression)
  - ``PrismaClient._probe_target_wrapper``
  - ``PrismaClient._probe_answers_now``
  - ``PrismaClient._planned_engine_replacement_absorbed``
"""

import asyncio
import os
import signal
import sys
import urllib.parse
from datetime import datetime, timedelta
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock, call, patch

import httpx
import pytest
from prisma import Prisma as GeneratedPrisma
from prisma.engine.errors import EngineConnectionError


from litellm.proxy.db.prisma_client import PrismaWrapper
from litellm.proxy.utils import PrismaClient


@pytest.fixture(autouse=True)
def mock_prisma_binary():
    """Mock prisma.Prisma to avoid requiring generated Prisma binaries for unit tests."""
    mock_module = MagicMock()
    # Production code isinstance-checks against this, which a bare MagicMock
    # attribute cannot satisfy.
    mock_module.engine.errors.EngineConnectionError = EngineConnectionError
    with patch.dict(sys.modules, {"prisma": mock_module}):
        yield mock_module


def _make_wrapper(engine_pid: int = 111, iam: bool = False) -> PrismaWrapper:
    prisma = GeneratedPrisma(use_dotenv=False)
    engine = MagicMock()
    engine.process.pid = engine_pid
    setattr(prisma, "_Prisma__engine", engine)
    return PrismaWrapper(original_prisma=prisma, iam_token_db_auth=iam)


def _make_prisma_client(db: Any) -> PrismaClient:
    """A ``PrismaClient`` whose ``db`` is a real wrapper and whose alerting
    hook is observable."""
    proxy_logging_obj = MagicMock()
    proxy_logging_obj.failure_handler = AsyncMock()
    client = PrismaClient(
        database_url="postgresql://user:pass@localhost:5432/db",
        proxy_logging_obj=proxy_logging_obj,
    )
    client.db = db
    client._db_watchdog_reconnect_timeout_seconds = 5.0
    return client


_real_asyncio_sleep = asyncio.sleep


async def _yield_to_loop(times: int = 10) -> None:
    """Let already-scheduled tasks make progress.

    Bound to the real ``asyncio.sleep`` at import time: the tests below patch
    ``asyncio.sleep`` to skip the SIGTERM/SIGKILL grace, and an ``AsyncMock``
    stand-in never yields to the event loop, which would silently leave every
    background task un-started and the assertions vacuous.
    """
    for _ in range(times):
        await _real_asyncio_sleep(0)


async def _await_health_check_reports() -> None:
    """Await the fire-and-forget reporting tasks ``health_check()`` scheduled.

    Selected by coroutine qualname rather than by draining every pending task,
    so an unrelated background task can never make these assertions pass by
    accident.
    """
    reports = [
        task
        for task in asyncio.all_tasks()
        if getattr(task.get_coro(), "__qualname__", "")
        == "PrismaClient._report_health_check_failure"
    ]
    if reports:
        await asyncio.gather(*reports, return_exceptions=True)


def _fails_then_answers(error: Exception, failures: int = 3) -> Any:
    """Raise ``error`` for the first ``failures`` probes, then answer.

    ``health_check`` retries up to three times, so this exhausts the retries and
    still lets the confirmation probe that decides suppression succeed. Without
    that, a test would report for the wrong reason: the confirmation probe would
    fail too, masking whether the error type was classified at all.
    """
    seen: List[int] = []

    async def _query_raw(_sql: str) -> Any:
        seen.append(1)
        if len(seen) <= failures:
            raise error
        return [{"?column?": 1}]

    return _query_raw


def _blocking_replacement(gate: asyncio.Event, fail: bool = False) -> MagicMock:
    """A replacement Prisma whose ``connect()`` parks until ``gate`` is set.

    Holds ``_reconnection_lock`` open for as long as the test needs, which is
    how a health probe is made to fail *while* a planned replacement is in
    flight rather than after it.
    """

    async def _connect(*_: Any, **__: Any) -> None:
        await gate.wait()
        if fail:
            raise ConnectionRefusedError("database is down")

    return MagicMock(connect=AsyncMock(side_effect=_connect))


def _token_db_url(created: datetime, expires_in: int = 900) -> str:
    """Build a DATABASE_URL whose password is a parseable RDS IAM token."""
    token = (
        f"host/?X-Amz-Date={created.strftime('%Y%m%dT%H%M%SZ')}"
        f"&X-Amz-Expires={expires_in}&X-Amz-Signature=abc"
    )
    quoted = urllib.parse.quote(token, safe="")
    return f"postgresql://user:{quoted}@host:5432/db"


def test_wrapper_instruments_generated_prisma_engine() -> None:
    prisma = GeneratedPrisma(use_dotenv=False)
    engine = MagicMock()
    setattr(prisma, "_Prisma__engine", engine)

    wrapper = PrismaWrapper(original_prisma=prisma, iam_token_db_auth=False)

    assert wrapper._active_drain_tracker is not None
    assert prisma._engine._engine is engine


async def _wait_for_retirements(wrapper: PrismaWrapper) -> None:
    await asyncio.gather(*tuple(wrapper._retirement_tasks))


@pytest.mark.asyncio
async def test_recreate_marks_old_engine_pid_as_expected_death(mock_prisma_binary):
    """The watcher must be able to tell a planned kill from a crash."""
    wrapper = _make_wrapper(engine_pid=111)
    mock_prisma_binary.Prisma.return_value = MagicMock(connect=AsyncMock())

    with (
        patch("os.kill"),
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        await wrapper.recreate_prisma_client("postgresql://new")

    assert 111 in wrapper._expected_engine_deaths


@pytest.mark.asyncio
async def test_recreate_increments_engine_generation(mock_prisma_binary):
    wrapper = _make_wrapper(engine_pid=111)
    mock_prisma_binary.Prisma.return_value = MagicMock(connect=AsyncMock())

    assert wrapper._engine_generation == 0
    with (
        patch("os.kill"),
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        await wrapper.recreate_prisma_client("postgresql://new")

    assert wrapper._engine_generation == 1


@pytest.mark.asyncio
async def test_recreate_skips_when_expected_generation_is_stale(mock_prisma_binary):
    """A reconnect that observed a failure before another path already
    recreated the client must not recreate (and kill the fresh engine) again."""
    wrapper = _make_wrapper(engine_pid=111)
    old_prisma = wrapper._original_prisma
    wrapper._engine_generation = 3

    with (
        patch("os.kill") as mock_kill,
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        recreated = await wrapper.recreate_prisma_client(
            "postgresql://new", expected_generation=2
        )

    pinned = {
        "recreated": recreated,
        "prisma_constructed": mock_prisma_binary.Prisma.call_count,
        "killed": mock_kill.call_count,
        "client_unchanged": wrapper._original_prisma is old_prisma,
        "generation": wrapper._engine_generation,
    }
    assert pinned == {
        "recreated": False,
        "prisma_constructed": 0,
        "killed": 0,
        "client_unchanged": True,
        "generation": 3,
    }


@pytest.mark.asyncio
async def test_recreate_proceeds_when_expected_generation_matches(mock_prisma_binary):
    wrapper = _make_wrapper(engine_pid=111)
    wrapper._engine_generation = 3
    mock_prisma_binary.Prisma.return_value = MagicMock(connect=AsyncMock())

    with (
        patch("os.kill"),
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        recreated = await wrapper.recreate_prisma_client(
            "postgresql://new", expected_generation=3
        )

    assert recreated is True
    assert wrapper._engine_generation == 4


@pytest.mark.asyncio
async def test_concurrent_guarded_recreates_only_recreate_once(mock_prisma_binary):
    """Two racing reconnect paths that both observed generation 0 must result
    in exactly one engine recreate (the loser sees the bumped generation)."""
    wrapper = _make_wrapper(engine_pid=111)
    mock_prisma_binary.Prisma.return_value = MagicMock(connect=AsyncMock())

    with (
        patch("os.kill"),
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        results = await asyncio.gather(
            wrapper.recreate_prisma_client("postgresql://new", expected_generation=0),
            wrapper.recreate_prisma_client("postgresql://new", expected_generation=0),
        )

    pinned = {
        "results": sorted(results),
        "prisma_constructed": mock_prisma_binary.Prisma.call_count,
        "generation": wrapper._engine_generation,
    }
    assert pinned == {
        "results": [False, True],
        "prisma_constructed": 1,
        "generation": 1,
    }


@pytest.mark.asyncio
async def test_on_engine_replaced_invoked_after_successful_recreate(
    mock_prisma_binary,
):
    """PrismaClient hooks this to re-arm the engine watcher on the new PID."""
    wrapper = _make_wrapper(engine_pid=111)
    mock_prisma_binary.Prisma.return_value = MagicMock(connect=AsyncMock())
    hook = MagicMock()
    wrapper.on_engine_replaced = hook

    with (
        patch("os.kill"),
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        await wrapper.recreate_prisma_client("postgresql://new")

    assert hook.call_count == 1


@pytest.mark.asyncio
async def test_on_engine_replaced_not_invoked_when_recreate_skipped(
    mock_prisma_binary,
):
    wrapper = _make_wrapper(engine_pid=111)
    wrapper._engine_generation = 5
    hook = MagicMock()
    wrapper.on_engine_replaced = hook

    await wrapper.recreate_prisma_client("postgresql://new", expected_generation=1)

    assert hook.call_count == 0


@pytest.mark.asyncio
async def test_safe_refresh_token_skips_when_token_still_fresh(
    mock_prisma_binary, monkeypatch
):
    """Stacked refresh triggers (e.g. __getattr__ scheduling a refresh task
    that runs after the proactive loop already refreshed) must coalesce
    instead of killing the freshly-spawned engine again."""
    wrapper = _make_wrapper(engine_pid=111, iam=True)
    monkeypatch.setenv(
        "DATABASE_URL", _token_db_url(created=datetime.utcnow(), expires_in=900)
    )
    wrapper.get_rds_iam_token = MagicMock(return_value="postgresql://fresh")

    await wrapper._safe_refresh_token()

    pinned = {
        "token_minted": wrapper.get_rds_iam_token.call_count,
        "prisma_constructed": mock_prisma_binary.Prisma.call_count,
    }
    assert pinned == {"token_minted": 0, "prisma_constructed": 0}


@pytest.mark.asyncio
async def test_safe_refresh_token_refreshes_when_token_expired(
    mock_prisma_binary, monkeypatch
):
    wrapper = _make_wrapper(engine_pid=111, iam=True)
    expired = datetime.utcnow() - timedelta(seconds=1200)
    monkeypatch.setenv("DATABASE_URL", _token_db_url(created=expired, expires_in=900))
    wrapper.get_rds_iam_token = MagicMock(return_value="postgresql://fresh")
    mock_prisma_binary.Prisma.return_value = MagicMock(connect=AsyncMock())

    with (
        patch("os.kill"),
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        await wrapper._safe_refresh_token()
        await _wait_for_retirements(wrapper)

    pinned = {
        "token_minted": wrapper.get_rds_iam_token.call_count,
        "prisma_constructed": mock_prisma_binary.Prisma.call_count,
    }
    assert pinned == {"token_minted": 1, "prisma_constructed": 1}


@pytest.mark.asyncio
async def test_safe_refresh_connects_replacement_before_swapping_and_killing(
    mock_prisma_binary, monkeypatch
):
    wrapper = _make_wrapper(engine_pid=111, iam=True)
    old_prisma = wrapper._original_prisma
    expired = datetime.utcnow() - timedelta(seconds=1200)
    monkeypatch.setenv("DATABASE_URL", _token_db_url(created=expired, expires_in=900))
    wrapper.get_rds_iam_token = MagicMock(return_value="postgresql://fresh")
    replacement_prisma = MagicMock()

    async def connect_replacement() -> None:
        assert wrapper._original_prisma is old_prisma
        assert mock_kill.call_count == 0

    replacement_prisma.connect = AsyncMock(side_effect=connect_replacement)
    replacement_prisma.is_connected = MagicMock(return_value=True)
    replacement_prisma._engine = MagicMock()
    replacement_prisma._engine.process.pid = 222
    mock_prisma_binary.Prisma.return_value = replacement_prisma

    def observe_kill(pid: int, sent_signal: int) -> None:
        assert wrapper._original_prisma is replacement_prisma

    with (
        patch("os.kill", side_effect=observe_kill) as mock_kill,
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        await wrapper._safe_refresh_token()
        await _wait_for_retirements(wrapper)

    assert wrapper._original_prisma is replacement_prisma
    mock_kill.assert_any_call(111, signal.SIGTERM)
    assert wrapper._engine_generation == 1


@pytest.mark.asyncio
async def test_safe_refresh_failure_keeps_old_client_and_token(
    mock_prisma_binary, monkeypatch
):
    wrapper = _make_wrapper(engine_pid=111, iam=True)
    old_prisma = wrapper._original_prisma
    expired = datetime.utcnow() - timedelta(seconds=1200)
    previous_db_url = _token_db_url(created=expired, expires_in=900)
    monkeypatch.setenv("DATABASE_URL", previous_db_url)

    def mint_token() -> str:
        monkeypatch.setenv("DATABASE_URL", "postgresql://fresh")
        return "postgresql://fresh"

    wrapper.get_rds_iam_token = MagicMock(side_effect=mint_token)
    replacement_prisma = MagicMock(connect=AsyncMock(side_effect=RuntimeError("database unavailable")))
    mock_prisma_binary.Prisma.return_value = replacement_prisma

    with patch("os.kill") as mock_kill:
        with pytest.raises(RuntimeError, match="database unavailable"):
            await wrapper._safe_refresh_token()

    assert wrapper._original_prisma is old_prisma
    assert os.environ["DATABASE_URL"] == previous_db_url
    assert wrapper._engine_generation == 0
    mock_kill.assert_not_called()


@pytest.mark.asyncio
async def test_safe_refresh_waits_for_active_query_before_retiring_old_engine(
    mock_prisma_binary, monkeypatch
):
    wrapper = _make_wrapper(engine_pid=111, iam=True)
    old_engine = wrapper._original_prisma._engine
    query_started = asyncio.Event()
    release_query = asyncio.Event()

    async def run_query(content: str, *, tx_id: object | None) -> dict[str, str]:
        query_started.set()
        await release_query.wait()
        return {"status": "complete"}

    old_engine._engine.query = AsyncMock(side_effect=run_query)
    query_task = asyncio.create_task(old_engine.query("query", tx_id=None))
    await query_started.wait()

    expired = datetime.utcnow() - timedelta(seconds=1200)
    monkeypatch.setenv("DATABASE_URL", _token_db_url(created=expired, expires_in=900))
    wrapper.get_rds_iam_token = MagicMock(return_value="postgresql://fresh")
    replacement_prisma = MagicMock(connect=AsyncMock())
    replacement_prisma._engine = MagicMock()
    replacement_prisma._engine.process.pid = 222
    mock_prisma_binary.Prisma.return_value = replacement_prisma

    with (
        patch("os.kill") as mock_kill,
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        await wrapper._safe_refresh_token()
        assert mock_kill.call_count == 0
        assert not query_task.done()
        release_query.set()
        await query_task
        await _wait_for_retirements(wrapper)

    mock_kill.assert_any_call(111, signal.SIGTERM)


@pytest.mark.asyncio
async def test_safe_refresh_waits_for_open_transaction_before_retiring_old_engine(
    mock_prisma_binary, monkeypatch
):
    wrapper = _make_wrapper(engine_pid=111, iam=True)
    old_engine = wrapper._original_prisma._engine
    old_engine._engine.start_transaction = AsyncMock(return_value="transaction-1")
    old_engine._engine.commit_transaction = AsyncMock()
    transaction_id = await old_engine.start_transaction(content="transaction")

    expired = datetime.utcnow() - timedelta(seconds=1200)
    monkeypatch.setenv("DATABASE_URL", _token_db_url(created=expired, expires_in=900))
    wrapper.get_rds_iam_token = MagicMock(return_value="postgresql://fresh")
    replacement_prisma = MagicMock(connect=AsyncMock())
    replacement_prisma._engine = MagicMock()
    replacement_prisma._engine.process.pid = 222
    mock_prisma_binary.Prisma.return_value = replacement_prisma

    with (
        patch("os.kill") as mock_kill,
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        await wrapper._safe_refresh_token()
        assert mock_kill.call_count == 0
        await old_engine.commit_transaction(transaction_id)
        await _wait_for_retirements(wrapper)

    mock_kill.assert_any_call(111, signal.SIGTERM)


@pytest.mark.asyncio
async def test_retirement_kills_old_engine_when_drain_never_completes(
    mock_prisma_binary, monkeypatch
):
    """A leaked drain count (e.g. a transaction whose owner was hard-cancelled
    before commit/rollback) must not keep the replaced engine alive forever."""
    wrapper = _make_wrapper(engine_pid=111, iam=True)
    monkeypatch.setattr(wrapper, "ENGINE_RETIREMENT_DRAIN_TIMEOUT_SECONDS", 0.05)
    old_engine = wrapper._original_prisma._engine
    old_engine._engine.start_transaction = AsyncMock(return_value="transaction-1")
    await old_engine.start_transaction(content="transaction")

    expired = datetime.utcnow() - timedelta(seconds=1200)
    monkeypatch.setenv("DATABASE_URL", _token_db_url(created=expired, expires_in=900))
    wrapper.get_rds_iam_token = MagicMock(return_value="postgresql://fresh")
    replacement_prisma = MagicMock(connect=AsyncMock())
    replacement_prisma._engine = MagicMock()
    replacement_prisma._engine.process.pid = 222
    mock_prisma_binary.Prisma.return_value = replacement_prisma

    with (
        patch("os.kill") as mock_kill,
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        await wrapper._safe_refresh_token()
        await asyncio.wait_for(_wait_for_retirements(wrapper), timeout=2)

    mock_kill.assert_any_call(111, signal.SIGTERM)


@pytest.mark.asyncio
async def test_safe_refresh_cancellation_restores_token_and_cleans_replacement(
    mock_prisma_binary, monkeypatch
):
    wrapper = _make_wrapper(engine_pid=111, iam=True)
    old_prisma = wrapper._original_prisma
    expired = datetime.utcnow() - timedelta(seconds=1200)
    previous_db_url = _token_db_url(created=expired, expires_in=900)
    monkeypatch.setenv("DATABASE_URL", previous_db_url)
    connect_started = asyncio.Event()
    block_connect = asyncio.Event()

    def mint_token() -> str:
        monkeypatch.setenv("DATABASE_URL", "postgresql://fresh")
        return "postgresql://fresh"

    async def connect_replacement() -> None:
        connect_started.set()
        await block_connect.wait()

    wrapper.get_rds_iam_token = MagicMock(side_effect=mint_token)
    replacement_prisma = MagicMock(connect=AsyncMock(side_effect=connect_replacement))
    replacement_prisma._engine = MagicMock()
    replacement_prisma._engine.process.pid = 222
    mock_prisma_binary.Prisma.return_value = replacement_prisma

    with (
        patch("os.kill") as mock_kill,
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        refresh_task = asyncio.create_task(wrapper._safe_refresh_token())
        await connect_started.wait()
        refresh_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await refresh_task
        await _wait_for_retirements(wrapper)

    assert wrapper._original_prisma is old_prisma
    assert os.environ["DATABASE_URL"] == previous_db_url
    assert wrapper._engine_generation == 0
    assert mock_kill.call_args_list == [
        call(222, signal.SIGTERM),
        call(222, signal.SIGKILL),
    ]


@pytest.mark.asyncio
async def test_repeated_safe_refreshes_retire_each_replaced_engine(
    mock_prisma_binary, monkeypatch
):
    wrapper = _make_wrapper(engine_pid=111, iam=True)
    replacement_clients = []
    for engine_pid in (222, 333):
        replacement_prisma = MagicMock(connect=AsyncMock())
        replacement_prisma.is_connected = MagicMock(return_value=True)
        replacement_prisma._engine = MagicMock()
        replacement_prisma._engine.process.pid = engine_pid
        replacement_clients.append(replacement_prisma)
    mock_prisma_binary.Prisma.side_effect = replacement_clients
    wrapper.get_rds_iam_token = MagicMock(side_effect=("postgresql://first", "postgresql://second"))
    expired = datetime.utcnow() - timedelta(seconds=1200)

    with (
        patch("os.kill") as mock_kill,
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        for _ in range(2):
            monkeypatch.setenv("DATABASE_URL", _token_db_url(created=expired, expires_in=900))
            await wrapper._safe_refresh_token()
        await _wait_for_retirements(wrapper)

    assert mock_kill.call_args_list == [
        call(111, signal.SIGTERM),
        call(111, signal.SIGKILL),
        call(222, signal.SIGTERM),
        call(222, signal.SIGKILL),
    ]
    assert wrapper._original_prisma is replacement_clients[-1]
    assert wrapper._engine_generation == 2


@pytest.mark.asyncio
async def test_safe_refresh_token_refreshes_when_token_unparseable(
    mock_prisma_binary, monkeypatch
):
    """Unparseable tokens follow the fallback-interval path and must always
    refresh — skipping here would mean never refreshing at all."""
    wrapper = _make_wrapper(engine_pid=111, iam=True)
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:plainpass@host:5432/db")
    wrapper.get_rds_iam_token = MagicMock(return_value="postgresql://fresh")
    mock_prisma_binary.Prisma.return_value = MagicMock(connect=AsyncMock())

    with (
        patch("os.kill"),
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        await wrapper._safe_refresh_token()
        await _wait_for_retirements(wrapper)

    assert wrapper.get_rds_iam_token.call_count == 1


@pytest.mark.asyncio
async def test_routing_recreate_skips_reader_when_writer_generation_stale(
    mock_prisma_binary, monkeypatch
):
    from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper

    monkeypatch.setenv("DATABASE_URL_READ_REPLICA", "postgresql://reader")
    writer = _make_wrapper(engine_pid=111)
    reader = _make_wrapper(engine_pid=222)
    writer._engine_generation = 2
    reader.recreate_prisma_client = AsyncMock()
    routing = RoutingPrismaWrapper(writer=writer, reader=reader)

    recreated = await routing.recreate_prisma_client(
        "postgresql://new", expected_generation=1
    )

    pinned = {
        "recreated": recreated,
        "reader_recreated": reader.recreate_prisma_client.await_count,
        "writer_prisma_constructed": mock_prisma_binary.Prisma.call_count,
    }
    assert pinned == {
        "recreated": False,
        "reader_recreated": 0,
        "writer_prisma_constructed": 0,
    }


@pytest.mark.asyncio
async def test_routing_recreate_recreates_both_when_generation_matches(
    mock_prisma_binary, monkeypatch
):
    from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper

    monkeypatch.setenv("DATABASE_URL_READ_REPLICA", "postgresql://reader")
    writer = _make_wrapper(engine_pid=111)
    reader = _make_wrapper(engine_pid=222)
    reader.recreate_prisma_client = AsyncMock()
    routing = RoutingPrismaWrapper(writer=writer, reader=reader)
    mock_prisma_binary.Prisma.return_value = MagicMock(connect=AsyncMock())

    with (
        patch("os.kill"),
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        recreated = await routing.recreate_prisma_client(
            "postgresql://new", expected_generation=0
        )

    pinned = {
        "recreated": recreated,
        "reader_recreated": reader.recreate_prisma_client.await_count,
    }
    assert pinned == {"recreated": True, "reader_recreated": 1}


@pytest.mark.asyncio
async def test_recreate_caps_expected_engine_deaths_set(mock_prisma_binary):
    """The planned-death set is bounded. Stale PIDs accrue when a death
    callback early-returns on PID mismatch (watcher already re-armed on the new
    engine), so a recreate clears the set once it grows past the cap, then
    records only the current old PID."""
    wrapper = _make_wrapper(engine_pid=111)
    mock_prisma_binary.Prisma.return_value = MagicMock(connect=AsyncMock())
    # Seed with stale PIDs at the cap so the next recreate triggers the clear.
    wrapper._expected_engine_deaths = set(range(1000, 1064))
    assert len(wrapper._expected_engine_deaths) >= 64

    with (
        patch("os.kill"),
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        await wrapper.recreate_prisma_client("postgresql://new")

    assert wrapper._expected_engine_deaths == {111}


@pytest.mark.asyncio
async def test_wait_for_planned_engine_replacement_returns_once_recreate_settles(
    mock_prisma_binary,
):
    wrapper = _make_wrapper(engine_pid=111)
    gate = asyncio.Event()
    mock_prisma_binary.Prisma.return_value = _blocking_replacement(gate)

    with patch("os.kill"), patch("asyncio.sleep", new_callable=AsyncMock):
        recreate = asyncio.create_task(
            wrapper.recreate_prisma_client("postgresql://new")
        )
        await _yield_to_loop()
        assert wrapper._reconnection_lock.locked() is True

        waiter = asyncio.create_task(wrapper.wait_for_planned_engine_replacement(5.0))
        await _yield_to_loop()
        blocked_while_in_flight = not waiter.done()

        gate.set()
        await recreate
        await waiter

    assert {
        "blocked_while_in_flight": blocked_while_in_flight,
        "generation": wrapper.engine_generation,
    } == {"blocked_while_in_flight": True, "generation": 1}


@pytest.mark.asyncio
async def test_wait_for_planned_engine_replacement_gives_up_at_timeout(
    mock_prisma_binary,
):
    """A replacement that never settles must not stall the caller forever; the
    caller then sees an unchanged generation and reports the failure."""
    wrapper = _make_wrapper(engine_pid=111)
    gate = asyncio.Event()
    mock_prisma_binary.Prisma.return_value = _blocking_replacement(gate)

    with patch("os.kill"), patch("asyncio.sleep", new_callable=AsyncMock):
        recreate = asyncio.create_task(
            wrapper.recreate_prisma_client("postgresql://new")
        )
        await _yield_to_loop()
        assert wrapper._reconnection_lock.locked() is True

        await asyncio.wait_for(
            wrapper.wait_for_planned_engine_replacement(0.05), timeout=5.0
        )
        gave_up_with_replacement_still_in_flight = not recreate.done()

        gate.set()
        await recreate

    assert gave_up_with_replacement_still_in_flight is True


@pytest.mark.asyncio
async def test_health_check_does_not_alert_when_probe_races_a_completed_replacement(
    mock_prisma_binary,
):
    """The reported bug: an IAM-refresh engine recreate makes a concurrent
    readiness probe fail transiently, and that failure was alerting as a DB
    exception on every refresh cycle.

    The reporting task is drained while the replacement is still in flight,
    which is when it runs in production; a decision taken at that instant sees
    an engine generation that has not moved yet.
    """
    wrapper = _make_wrapper(engine_pid=111)
    client = _make_prisma_client(wrapper)
    wrapper.query_raw = AsyncMock(
        side_effect=[
            httpx.ConnectError("All connection attempts failed"),
            [{"?column?": 1}],
            [{"?column?": 1}],
        ]
    )
    gate = asyncio.Event()
    mock_prisma_binary.Prisma.return_value = _blocking_replacement(gate)

    with patch("os.kill"), patch("asyncio.sleep", new_callable=AsyncMock):
        recreate = asyncio.create_task(
            wrapper.recreate_prisma_client("postgresql://new")
        )
        await _yield_to_loop()
        assert wrapper._reconnection_lock.locked() is True

        probe_result = await client.health_check()

        drain = asyncio.create_task(_await_health_check_reports())
        await _yield_to_loop()
        alerts_while_replacement_in_flight = (
            client.proxy_logging_obj.failure_handler.await_count
        )

        gate.set()
        await recreate
        await drain

    assert {
        "probe_result": probe_result,
        "probe_attempts": wrapper.query_raw.await_count,
        "alerts_while_in_flight": alerts_while_replacement_in_flight,
        "alerts": client.proxy_logging_obj.failure_handler.await_count,
    } == {
        "probe_result": [{"?column?": 1}],
        "probe_attempts": 3,
        "alerts_while_in_flight": 0,
        "alerts": 0,
    }


@pytest.mark.asyncio
async def test_health_check_alerts_when_a_completed_replacement_still_cannot_reach_the_database(
    mock_prisma_binary,
):
    """``Prisma.connect()`` polls the query engine's own ``/status`` endpoint
    rather than round-tripping to the database, so a replacement can complete
    against a database that is still unreachable. The engine generation alone
    must not be enough to stay silent."""
    wrapper = _make_wrapper(engine_pid=111)
    client = _make_prisma_client(wrapper)
    wrapper.query_raw = AsyncMock(
        side_effect=httpx.ConnectError("All connection attempts failed")
    )
    gate = asyncio.Event()
    mock_prisma_binary.Prisma.return_value = _blocking_replacement(gate)

    with patch("os.kill"), patch("asyncio.sleep", new_callable=AsyncMock):
        recreate = asyncio.create_task(
            wrapper.recreate_prisma_client("postgresql://new")
        )
        await _yield_to_loop()
        assert wrapper._reconnection_lock.locked() is True

        with pytest.raises(httpx.ConnectError):
            await client.health_check()

        gate.set()
        await recreate
        await _await_health_check_reports()

    assert {
        "replacement_completed": wrapper.engine_generation,
        "alerted": client.proxy_logging_obj.failure_handler.await_count > 0,
    } == {"replacement_completed": 1, "alerted": True}


@pytest.mark.asyncio
async def test_health_check_alerts_when_the_replacement_never_completes(
    mock_prisma_binary,
):
    """A real outage also has a replacement in flight, but it fails, so the
    engine generation never advances and the probe failure must still alert."""
    wrapper = _make_wrapper(engine_pid=111)
    client = _make_prisma_client(wrapper)
    wrapper.query_raw = AsyncMock(
        side_effect=httpx.ConnectError("All connection attempts failed")
    )
    gate = asyncio.Event()
    mock_prisma_binary.Prisma.return_value = _blocking_replacement(gate, fail=True)

    with patch("os.kill"), patch("asyncio.sleep", new_callable=AsyncMock):
        recreate = asyncio.create_task(
            wrapper.recreate_prisma_client("postgresql://new")
        )
        await _yield_to_loop()
        assert wrapper._reconnection_lock.locked() is True

        with pytest.raises(httpx.ConnectError):
            await client.health_check()

        gate.set()
        with pytest.raises(ConnectionRefusedError):
            await recreate
        await _await_health_check_reports()

    call_types: List[str] = [
        c.kwargs["call_type"]
        for c in client.proxy_logging_obj.failure_handler.await_args_list
    ]
    assert {
        "generation": wrapper.engine_generation,
        "alerted": len(call_types) > 0,
        "call_types": set(call_types),
    } == {"generation": 0, "alerted": True, "call_types": {"health_check"}}


@pytest.mark.asyncio
async def test_health_check_alerts_for_non_connection_errors_during_a_replacement(
    mock_prisma_binary,
):
    """Suppression is scoped to transport failures. A query the database itself
    rejected is a real defect and must alert even mid-replacement, and even
    though the database is plainly reachable a moment later."""
    wrapper = _make_wrapper(engine_pid=111)
    client = _make_prisma_client(wrapper)
    wrapper.query_raw = AsyncMock(side_effect=_fails_then_answers(ValueError("malformed SELECT")))
    gate = asyncio.Event()
    mock_prisma_binary.Prisma.return_value = _blocking_replacement(gate)

    with patch("os.kill"), patch("asyncio.sleep", new_callable=AsyncMock):
        recreate = asyncio.create_task(
            wrapper.recreate_prisma_client("postgresql://new")
        )
        await _yield_to_loop()
        assert wrapper._reconnection_lock.locked() is True

        with pytest.raises(ValueError, match='malformed SELECT'):
            await client.health_check()

        gate.set()
        await recreate
        await _await_health_check_reports()

    assert {
        "generation": wrapper.engine_generation,
        "alerted": client.proxy_logging_obj.failure_handler.await_count > 0,
    } == {"generation": 1, "alerted": True}


@pytest.mark.asyncio
async def test_health_check_alerts_for_a_transient_failure_with_no_engine_replacement(
    mock_prisma_binary,
):
    """Suppression is scoped to failures a planned replacement explains. A
    transport blip that self-heals with no engine replacement at all still
    alerts, so the gate cannot be widened into silencing every failure whose
    database happens to answer a moment later."""
    wrapper = _make_wrapper(engine_pid=111)
    client = _make_prisma_client(wrapper)
    wrapper.query_raw = AsyncMock(
        side_effect=[
            httpx.ConnectError("All connection attempts failed"),
            [{"?column?": 1}],
            [{"?column?": 1}],
        ]
    )

    probe_result = await client.health_check()
    await _await_health_check_reports()

    assert {
        "probe_result": probe_result,
        "generation": wrapper.engine_generation,
        "alerted": client.proxy_logging_obj.failure_handler.await_count > 0,
    } == {"probe_result": [{"?column?": 1}], "generation": 0, "alerted": True}


@pytest.mark.asyncio
async def test_health_probe_stays_on_its_target_when_reader_availability_flips(
    mock_prisma_binary, monkeypatch
):
    """Routing is re-resolved on every attribute access, so a reader that
    recovers mid-call would otherwise send the probe to a different engine than
    the one whose generation is being checked, and blame the wrong replacement.
    The probe follows the wrapper it was handed."""
    from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper

    monkeypatch.setenv("DATABASE_URL_READ_REPLICA", "postgresql://reader")
    writer = _make_wrapper(engine_pid=111)
    reader = _make_wrapper(engine_pid=222)
    routing = RoutingPrismaWrapper(writer=writer, reader=reader)
    client = _make_prisma_client(routing)
    writer.query_raw = AsyncMock(return_value=[{"writer": 1}])
    reader.query_raw = AsyncMock(return_value=[{"reader": 1}])

    routing._reader_unavailable = False
    target = client._probe_target_wrapper()
    routing._reader_unavailable = True
    result = await client._run_health_probe(target)

    assert {
        "target_is_reader": target is reader,
        "result": result,
        "reader_probes": reader.query_raw.await_count,
        "writer_probes": writer.query_raw.await_count,
    } == {
        "target_is_reader": True,
        "result": [{"reader": 1}],
        "reader_probes": 1,
        "writer_probes": 0,
    }


@pytest.mark.asyncio
async def test_health_check_consults_the_reader_wrapper_under_read_replica_routing(
    mock_prisma_binary, monkeypatch
):
    """``query_raw`` is routed to the reader, so a reader-side planned
    replacement is the one that explains a probe failure."""
    from litellm.proxy.db.routing_prisma_wrapper import RoutingPrismaWrapper

    monkeypatch.setenv("DATABASE_URL_READ_REPLICA", "postgresql://reader")
    writer = _make_wrapper(engine_pid=111)
    reader = _make_wrapper(engine_pid=222)
    routing = RoutingPrismaWrapper(writer=writer, reader=reader)
    client = _make_prisma_client(routing)
    reader.query_raw = AsyncMock(
        side_effect=[
            httpx.ConnectError("All connection attempts failed"),
            [{"?column?": 1}],
            [{"?column?": 1}],
        ]
    )
    gate = asyncio.Event()
    mock_prisma_binary.Prisma.return_value = _blocking_replacement(gate)

    with patch("os.kill"), patch("asyncio.sleep", new_callable=AsyncMock):
        recreate = asyncio.create_task(
            reader.recreate_prisma_client("postgresql://new-reader")
        )
        await _yield_to_loop()
        assert reader._reconnection_lock.locked() is True

        probe_result = await client.health_check()

        drain = asyncio.create_task(_await_health_check_reports())
        await _yield_to_loop()
        alerts_while_replacement_in_flight = (
            client.proxy_logging_obj.failure_handler.await_count
        )

        gate.set()
        await recreate
        await drain

    assert {
        "probe_target_is_the_reader": client._probe_target_wrapper() is reader,
        "writer_generation": writer.engine_generation,
        "reader_generation": reader.engine_generation,
        "probe_result": probe_result,
        "alerts_while_in_flight": alerts_while_replacement_in_flight,
        "alerts": client.proxy_logging_obj.failure_handler.await_count,
    } == {
        "probe_target_is_the_reader": True,
        "writer_generation": 0,
        "reader_generation": 1,
        "probe_result": [{"?column?": 1}],
        "alerts_while_in_flight": 0,
        "alerts": 0,
    }
