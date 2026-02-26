"""
Tests for PrismaClient engine memory reclamation: RSS monitoring and periodic restart.

Covers:
- _read_pid_rss_mb reads /proc/<pid>/status correctly
- _read_pid_rss_mb returns None when file is missing or malformed
- Memory reclaim loop triggers restart when RSS exceeds threshold
- Memory reclaim loop triggers restart when engine age exceeds max
- Memory reclaim loop skips restart when both thresholds are healthy
- _perform_engine_memory_reclaim disconnects and reconnects the engine
- _perform_engine_memory_reclaim handles missing DATABASE_URL gracefully
- start/stop lifecycle management
- Feature disabled via env var
"""

import asyncio
import os
import sys
import time
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

from litellm.proxy.utils import PrismaClient, ProxyLogging


@pytest.fixture(autouse=True)
def mock_prisma_binary():
    """Mock prisma.Prisma to avoid requiring generated Prisma binaries for unit tests."""
    mock_module = MagicMock()
    with patch.dict(sys.modules, {"prisma": mock_module}):
        yield


@pytest.fixture
def mock_proxy_logging():
    proxy_logging = AsyncMock(spec=ProxyLogging)
    proxy_logging.failure_handler = AsyncMock()
    return proxy_logging


@pytest.fixture
def engine_client(mock_proxy_logging) -> PrismaClient:
    client = PrismaClient(
        database_url="mock://test", proxy_logging_obj=mock_proxy_logging
    )
    client.db = MagicMock()
    client.db.recreate_prisma_client = AsyncMock()
    client.db.disconnect = AsyncMock(return_value=None)
    client.db.connect = AsyncMock(return_value=None)
    client.db.query_raw = AsyncMock(return_value=[{"result": 1}])
    return client


# ---------------------------------------------------------------------------
# _read_pid_rss_mb
# ---------------------------------------------------------------------------


def test_read_pid_rss_mb_parses_vmrss_correctly(engine_client):
    """should parse VmRSS line from /proc/<pid>/status and return MB."""
    proc_status = (
        "Name:\tquery-engine\n"
        "VmPeak:\t4096000 kB\n"
        "VmSize:\t3500000 kB\n"
        "VmRSS:\t2048000 kB\n"
        "VmData:\t1000000 kB\n"
    )
    with patch("builtins.open", mock_open(read_data=proc_status)):
        result = PrismaClient._read_pid_rss_mb(1234)
    assert result == pytest.approx(2048000 / 1024.0)  # 2000.0 MB


def test_read_pid_rss_mb_returns_none_when_file_missing(engine_client):
    """should return None when /proc/<pid>/status does not exist."""
    with patch("builtins.open", side_effect=OSError("No such file")):
        result = PrismaClient._read_pid_rss_mb(99999)
    assert result is None


def test_read_pid_rss_mb_returns_none_when_no_vmrss_line(engine_client):
    """should return None when VmRSS is not in the status file."""
    proc_status = "Name:\tquery-engine\nVmPeak:\t4096000 kB\n"
    with patch("builtins.open", mock_open(read_data=proc_status)):
        result = PrismaClient._read_pid_rss_mb(1234)
    assert result is None


def test_read_pid_rss_mb_returns_none_on_malformed_line(engine_client):
    """should return None when VmRSS line is malformed."""
    proc_status = "VmRSS:\tmalformed\n"
    with patch("builtins.open", mock_open(read_data=proc_status)):
        result = PrismaClient._read_pid_rss_mb(1234)
    assert result is None


# ---------------------------------------------------------------------------
# _perform_engine_memory_reclaim
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_perform_engine_memory_reclaim_restarts_engine(engine_client):
    """should disconnect/reconnect the engine and reset started_at timestamp."""
    engine_client._engine_pid = 1234
    engine_client._engine_started_at = time.time() - 7200
    engine_client._start_engine_watcher = AsyncMock()
    engine_client._stop_engine_watcher = MagicMock()

    mock_engine = MagicMock()
    mock_engine.process = MagicMock()
    mock_engine.process.pid = 5678
    engine_client.db._original_prisma = MagicMock()
    engine_client.db._original_prisma._engine = mock_engine

    with patch.dict(os.environ, {"DATABASE_URL": "postgresql://test"}):
        await engine_client._perform_engine_memory_reclaim()

    engine_client._stop_engine_watcher.assert_called_once()
    engine_client.db.recreate_prisma_client.assert_awaited_once_with(
        "postgresql://test"
    )
    engine_client._start_engine_watcher.assert_awaited_once()
    assert engine_client._engine_confirmed_dead is False
    assert engine_client._engine_started_at > time.time() - 5


@pytest.mark.asyncio
async def test_perform_engine_memory_reclaim_handles_missing_db_url(engine_client):
    """should log error and return without crashing when DATABASE_URL is unset."""
    engine_client._engine_pid = 1234
    engine_client._stop_engine_watcher = MagicMock()

    with patch.dict(os.environ, {}, clear=True):
        await engine_client._perform_engine_memory_reclaim()

    engine_client.db.recreate_prisma_client.assert_not_awaited()


@pytest.mark.asyncio
async def test_perform_engine_memory_reclaim_handles_reconnect_failure(engine_client):
    """should not crash when recreate_prisma_client raises."""
    engine_client._engine_pid = 1234
    engine_client._stop_engine_watcher = MagicMock()
    engine_client.db.recreate_prisma_client = AsyncMock(
        side_effect=RuntimeError("connection refused")
    )

    with patch.dict(os.environ, {"DATABASE_URL": "postgresql://test"}):
        await engine_client._perform_engine_memory_reclaim()

    engine_client.db.recreate_prisma_client.assert_awaited_once()


# ---------------------------------------------------------------------------
# _engine_memory_reclaim_loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_reclaim_loop_triggers_on_rss_exceeding_limit(engine_client):
    """should trigger reclaim when engine RSS exceeds memory limit."""
    engine_client._engine_memory_limit_mb = 1024
    engine_client._engine_max_age_seconds = 99999
    engine_client._engine_memory_check_interval_seconds = 0
    engine_client._engine_started_at = time.time()

    call_count = 0

    async def fake_sleep(seconds):
        nonlocal call_count
        call_count += 1
        if call_count > 1:
            raise asyncio.CancelledError()

    engine_client._perform_engine_memory_reclaim = AsyncMock()

    with (
        patch.object(engine_client, "_get_engine_pid", return_value=1234),
        patch.object(
            PrismaClient, "_read_pid_rss_mb", return_value=2048.0
        ),
        patch("asyncio.sleep", side_effect=fake_sleep),
    ):
        await engine_client._engine_memory_reclaim_loop()

    engine_client._perform_engine_memory_reclaim.assert_awaited_once()


@pytest.mark.asyncio
async def test_memory_reclaim_loop_triggers_on_max_age_exceeded(engine_client):
    """should trigger reclaim when engine age exceeds max_age."""
    engine_client._engine_memory_limit_mb = 99999
    engine_client._engine_max_age_seconds = 60
    engine_client._engine_memory_check_interval_seconds = 0
    engine_client._engine_started_at = time.time() - 120

    call_count = 0

    async def fake_sleep(seconds):
        nonlocal call_count
        call_count += 1
        if call_count > 1:
            raise asyncio.CancelledError()

    engine_client._perform_engine_memory_reclaim = AsyncMock()

    with (
        patch.object(engine_client, "_get_engine_pid", return_value=1234),
        patch.object(PrismaClient, "_read_pid_rss_mb", return_value=512.0),
        patch("asyncio.sleep", side_effect=fake_sleep),
    ):
        await engine_client._engine_memory_reclaim_loop()

    engine_client._perform_engine_memory_reclaim.assert_awaited_once()


@pytest.mark.asyncio
async def test_memory_reclaim_loop_skips_when_healthy(engine_client):
    """should not trigger reclaim when both RSS and age are within limits."""
    engine_client._engine_memory_limit_mb = 4096
    engine_client._engine_max_age_seconds = 7200
    engine_client._engine_memory_check_interval_seconds = 0
    engine_client._engine_started_at = time.time()

    call_count = 0

    async def fake_sleep(seconds):
        nonlocal call_count
        call_count += 1
        if call_count > 1:
            raise asyncio.CancelledError()

    engine_client._perform_engine_memory_reclaim = AsyncMock()

    with (
        patch.object(engine_client, "_get_engine_pid", return_value=1234),
        patch.object(PrismaClient, "_read_pid_rss_mb", return_value=512.0),
        patch("asyncio.sleep", side_effect=fake_sleep),
    ):
        await engine_client._engine_memory_reclaim_loop()

    engine_client._perform_engine_memory_reclaim.assert_not_awaited()


@pytest.mark.asyncio
async def test_memory_reclaim_loop_skips_when_no_engine_pid(engine_client):
    """should skip check when engine PID is unknown."""
    engine_client._engine_memory_check_interval_seconds = 0

    call_count = 0

    async def fake_sleep(seconds):
        nonlocal call_count
        call_count += 1
        if call_count > 1:
            raise asyncio.CancelledError()

    engine_client._perform_engine_memory_reclaim = AsyncMock()

    with (
        patch.object(engine_client, "_get_engine_pid", return_value=0),
        patch("asyncio.sleep", side_effect=fake_sleep),
    ):
        await engine_client._engine_memory_reclaim_loop()

    engine_client._perform_engine_memory_reclaim.assert_not_awaited()


@pytest.mark.asyncio
async def test_memory_reclaim_loop_handles_exception_gracefully(engine_client):
    """should catch exceptions in the loop and continue."""
    engine_client._engine_memory_check_interval_seconds = 0

    call_count = 0

    async def fake_sleep(seconds):
        nonlocal call_count
        call_count += 1
        if call_count > 2:
            raise asyncio.CancelledError()

    with (
        patch.object(
            engine_client,
            "_get_engine_pid",
            side_effect=RuntimeError("unexpected"),
        ),
        patch("asyncio.sleep", side_effect=fake_sleep),
    ):
        await engine_client._engine_memory_reclaim_loop()


# ---------------------------------------------------------------------------
# start/stop lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_engine_memory_reclaim_task_creates_task(engine_client):
    """should create a background task when enabled."""
    engine_client._engine_memory_reclaim_enabled = True
    assert engine_client._engine_memory_reclaim_task is None

    await engine_client.start_engine_memory_reclaim_task()

    assert engine_client._engine_memory_reclaim_task is not None
    engine_client._engine_memory_reclaim_task.cancel()
    try:
        await engine_client._engine_memory_reclaim_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_start_engine_memory_reclaim_task_noop_when_disabled(engine_client):
    """should not create a task when disabled."""
    engine_client._engine_memory_reclaim_enabled = False

    await engine_client.start_engine_memory_reclaim_task()

    assert engine_client._engine_memory_reclaim_task is None


@pytest.mark.asyncio
async def test_start_engine_memory_reclaim_task_noop_when_already_running(
    engine_client,
):
    """should not create a duplicate task."""
    engine_client._engine_memory_reclaim_enabled = True
    sentinel = MagicMock()
    engine_client._engine_memory_reclaim_task = sentinel

    await engine_client.start_engine_memory_reclaim_task()

    assert engine_client._engine_memory_reclaim_task is sentinel


@pytest.mark.asyncio
async def test_stop_engine_memory_reclaim_task_cancels(engine_client):
    """should cancel the running task."""
    engine_client._engine_memory_reclaim_enabled = True

    await engine_client.start_engine_memory_reclaim_task()
    assert engine_client._engine_memory_reclaim_task is not None

    await engine_client.stop_engine_memory_reclaim_task()
    assert engine_client._engine_memory_reclaim_task is None


@pytest.mark.asyncio
async def test_stop_engine_memory_reclaim_task_noop_when_not_running(engine_client):
    """should be a no-op when no task is running."""
    assert engine_client._engine_memory_reclaim_task is None
    await engine_client.stop_engine_memory_reclaim_task()
    assert engine_client._engine_memory_reclaim_task is None


# ---------------------------------------------------------------------------
# Constructor defaults
# ---------------------------------------------------------------------------


def test_default_config_values(engine_client):
    """should have sensible defaults for memory reclaim configuration."""
    assert engine_client._engine_memory_reclaim_enabled is True
    assert engine_client._engine_memory_limit_mb == 2048
    assert engine_client._engine_max_age_seconds == 3600
    assert engine_client._engine_memory_check_interval_seconds == 60


def test_config_from_env_vars(mock_proxy_logging):
    """should read configuration from environment variables."""
    env = {
        "PRISMA_ENGINE_MEMORY_RECLAIM_ENABLED": "false",
        "PRISMA_ENGINE_MEMORY_LIMIT_MB": "4096",
        "PRISMA_ENGINE_MAX_AGE_SECONDS": "7200",
        "PRISMA_ENGINE_MEMORY_CHECK_INTERVAL_SECONDS": "120",
    }
    with patch.dict(os.environ, env):
        client = PrismaClient(
            database_url="mock://test", proxy_logging_obj=mock_proxy_logging
        )
    assert client._engine_memory_reclaim_enabled is False
    assert client._engine_memory_limit_mb == 4096
    assert client._engine_max_age_seconds == 7200
    assert client._engine_memory_check_interval_seconds == 120
