"""Coordination between planned Prisma engine restarts and reconnect paths.

Covers the fix for https://github.com/BerriAI/litellm/issues/29176 — an RDS
IAM token refresh recreates the Prisma client (killing the query-engine
subprocess), and the engine-death watcher / in-flight transport-error
retries must not treat that planned restart as a crash and recreate the
client a second time.

Symbols pinned here:
  - ``PrismaWrapper._expected_engine_deaths``
  - ``PrismaWrapper._engine_generation``
  - ``PrismaWrapper.on_engine_replaced``
  - ``PrismaWrapper.recreate_prisma_client`` (expected_generation guard)
  - ``PrismaWrapper._safe_refresh_token`` (refresh coalescing)
  - ``RoutingPrismaWrapper.recreate_prisma_client`` (guard forwarding)
"""

import asyncio
import os
import sys
import urllib.parse
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

from litellm.proxy.db.prisma_client import PrismaWrapper


@pytest.fixture(autouse=True)
def mock_prisma_binary():
    """Mock prisma.Prisma to avoid requiring generated Prisma binaries for unit tests."""
    mock_module = MagicMock()
    with patch.dict(sys.modules, {"prisma": mock_module}):
        yield mock_module


def _make_wrapper(engine_pid: int = 111, iam: bool = False) -> PrismaWrapper:
    mock_prisma = MagicMock()
    mock_prisma.connect = AsyncMock()
    mock_prisma.is_connected = MagicMock(return_value=True)
    mock_prisma._engine = MagicMock()
    mock_prisma._engine.process.pid = engine_pid
    return PrismaWrapper(original_prisma=mock_prisma, iam_token_db_auth=iam)


def _token_db_url(created: datetime, expires_in: int = 900) -> str:
    """Build a DATABASE_URL whose password is a parseable RDS IAM token."""
    token = (
        f"host/?X-Amz-Date={created.strftime('%Y%m%dT%H%M%SZ')}"
        f"&X-Amz-Expires={expires_in}&X-Amz-Signature=abc"
    )
    quoted = urllib.parse.quote(token, safe="")
    return f"postgresql://user:{quoted}@host:5432/db"


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

    pinned = {
        "token_minted": wrapper.get_rds_iam_token.call_count,
        "prisma_constructed": mock_prisma_binary.Prisma.call_count,
    }
    assert pinned == {"token_minted": 1, "prisma_constructed": 1}


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
