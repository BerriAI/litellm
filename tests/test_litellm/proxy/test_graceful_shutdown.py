"""
Tests for graceful shutdown behavior of the LiteLLM proxy.

Covers:
- GracefulShutdownManager state machine
- /health/readiness 503 during shutdown
- /health/liveliness and /health/liveness 503 during shutdown
- /health/drain endpoint behavior
- Drain waits for in-flight counter to reach zero
- Drain respects GRACEFUL_SHUTDOWN_TIMEOUT
- Observability log events during drain
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("../../.."))

import litellm.proxy.health_endpoints._health_endpoints as _health_endpoints_module
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.health_endpoints._health_endpoints import (
    health_drain,
    health_liveliness,
    health_readiness,
)
from litellm.proxy.middleware.in_flight_requests_middleware import (
    InFlightRequestsMiddleware,
)
from litellm.proxy.shutdown.graceful_shutdown_manager import GracefulShutdownManager

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_shutdown_state():
    """Ensure GracefulShutdownManager state is clean between tests."""
    GracefulShutdownManager.reset()
    yield
    GracefulShutdownManager.reset()


@pytest.fixture(autouse=True)
def reset_in_flight():
    """Reset InFlightRequestsMiddleware counter between tests."""
    InFlightRequestsMiddleware._in_flight = 0
    yield
    InFlightRequestsMiddleware._in_flight = 0


# ---------------------------------------------------------------------------
# GracefulShutdownManager unit tests
# ---------------------------------------------------------------------------


def test_initial_state_is_not_shutting_down():
    assert GracefulShutdownManager.is_shutting_down() is False


def test_start_shutdown_sets_flag():
    GracefulShutdownManager.start_shutdown()
    assert GracefulShutdownManager.is_shutting_down() is True


def test_start_shutdown_is_idempotent():
    GracefulShutdownManager.start_shutdown()
    GracefulShutdownManager.start_shutdown()  # second call should not error
    assert GracefulShutdownManager.is_shutting_down() is True


def test_reset_clears_flag():
    GracefulShutdownManager.start_shutdown()
    GracefulShutdownManager.reset()
    assert GracefulShutdownManager.is_shutting_down() is False


@pytest.mark.asyncio
async def test_wait_for_drain_returns_immediately_when_no_in_flight():
    """If there are no in-flight requests, drain completes instantly."""
    GracefulShutdownManager.start_shutdown()
    drained = await GracefulShutdownManager.wait_for_drain(timeout=5)
    assert drained == 0


@pytest.mark.asyncio
async def test_wait_for_drain_waits_for_counter_to_reach_zero():
    """Drain polls until the in-flight counter reaches zero."""
    GracefulShutdownManager.start_shutdown()
    InFlightRequestsMiddleware._in_flight = 2

    async def _release_after_delay():
        await asyncio.sleep(0.2)
        InFlightRequestsMiddleware._in_flight = 0

    asyncio.create_task(_release_after_delay())
    drained = await GracefulShutdownManager.wait_for_drain(timeout=5)
    assert drained == 2
    assert InFlightRequestsMiddleware.get_count() == 0


@pytest.mark.asyncio
async def test_wait_for_drain_respects_timeout():
    """Drain exits after timeout even if requests are still in flight."""
    GracefulShutdownManager.start_shutdown()
    InFlightRequestsMiddleware._in_flight = 5

    drained = await GracefulShutdownManager.wait_for_drain(timeout=0.3)
    # Requests never finished, so drained == 0
    assert drained == 0
    # Counter is still non-zero
    assert InFlightRequestsMiddleware.get_count() == 5


@pytest.mark.asyncio
async def test_wait_for_drain_uses_env_timeout(monkeypatch):
    """GRACEFUL_SHUTDOWN_TIMEOUT env var controls the default timeout."""
    monkeypatch.setenv("GRACEFUL_SHUTDOWN_TIMEOUT", "0.2")
    GracefulShutdownManager.start_shutdown()
    InFlightRequestsMiddleware._in_flight = 1

    import time

    start = time.monotonic()
    await GracefulShutdownManager.wait_for_drain()  # no explicit timeout → uses env
    elapsed = time.monotonic() - start

    assert elapsed >= 0.2
    assert elapsed < 2.0  # should not hang


@pytest.mark.asyncio
async def test_wait_for_drain_emits_started_log(caplog):
    """start_shutdown emits a structured log line."""
    import logging

    with caplog.at_level(logging.INFO, logger="LiteLLM Proxy"):
        GracefulShutdownManager.start_shutdown()

    assert any("graceful_shutdown_started" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_wait_for_drain_emits_complete_log(caplog):
    """wait_for_drain emits a shutdown_complete structured log line."""
    import logging

    GracefulShutdownManager.start_shutdown()

    with caplog.at_level(logging.INFO, logger="LiteLLM Proxy"):
        await GracefulShutdownManager.wait_for_drain(timeout=1)

    assert any("graceful_shutdown_complete" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Health endpoint behavior during shutdown
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_readiness_returns_503_during_shutdown():
    """/health/readiness must return 503 when the proxy is shutting down."""
    from fastapi import Response

    GracefulShutdownManager.start_shutdown()

    response = Response()
    result = await health_readiness(response=response)

    assert response.status_code == 503
    assert result["status"] == "shutting_down"


@pytest.mark.asyncio
async def test_health_readiness_returns_200_when_not_shutting_down():
    """/health/readiness must return 200 during normal operation."""
    from fastapi import Response

    response = Response()
    with patch("litellm.proxy.proxy_server.prisma_client", None):
        result = await health_readiness(response=response)

    assert response.status_code == 200
    assert result["status"] == "healthy"


@pytest.mark.asyncio
async def test_health_readiness_shutdown_takes_priority_over_db_check():
    """
    During shutdown the 503 is returned immediately without checking the DB,
    so the endpoint stays fast even if DB is unreachable.
    """
    from fastapi import Response

    mock_prisma = MagicMock()
    mock_prisma.health_check = AsyncMock(side_effect=RuntimeError("DB unreachable"))

    GracefulShutdownManager.start_shutdown()

    response = Response()
    with patch("litellm.proxy.proxy_server.prisma_client", mock_prisma):
        result = await health_readiness(response=response)

    # DB health_check must NOT have been called
    mock_prisma.health_check.assert_not_called()
    assert response.status_code == 503
    assert result["status"] == "shutting_down"


@pytest.mark.asyncio
async def test_health_liveliness_returns_503_during_shutdown():
    """/health/liveliness must return 503 when the proxy is shutting down."""
    from fastapi import Response

    GracefulShutdownManager.start_shutdown()

    response = Response()
    result = await health_liveliness(response=response)

    assert response.status_code == 503
    assert result["status"] == "shutting_down"


@pytest.mark.asyncio
async def test_health_liveliness_returns_200_when_not_shutting_down():
    """/health/liveliness returns the legacy payload during normal operation."""
    from fastapi import Response

    response = Response()
    result = await health_liveliness(response=response)

    assert response.status_code == 200
    assert result == "I'm alive!"


def test_health_liveness_returns_503_via_test_client(monkeypatch):
    """/health/liveness returns 503 over HTTP when shutting down."""
    app = FastAPI()
    app.include_router(_health_endpoints_module.router)
    client = TestClient(app, raise_server_exceptions=False)

    GracefulShutdownManager.start_shutdown()

    response = client.get("/health/liveness")
    assert response.status_code == 503


def test_health_liveliness_returns_503_via_test_client(monkeypatch):
    """/health/liveliness returns 503 over HTTP when shutting down."""
    app = FastAPI()
    app.include_router(_health_endpoints_module.router)
    client = TestClient(app, raise_server_exceptions=False)

    GracefulShutdownManager.start_shutdown()

    response = client.get("/health/liveliness")
    assert response.status_code == 503


# ---------------------------------------------------------------------------
# /health/drain endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_drain_sets_shutdown_flag():
    """/health/drain must trigger the shutdown flag."""
    assert GracefulShutdownManager.is_shutting_down() is False
    await health_drain()
    assert GracefulShutdownManager.is_shutting_down() is True


@pytest.mark.asyncio
async def test_health_drain_returns_drained_count():
    """/health/drain reports how many requests were drained."""
    InFlightRequestsMiddleware._in_flight = 0
    result = await health_drain()
    assert result["status"] == "drained"
    assert result["in_flight_requests"] == 0
    assert "drained_requests" in result


@pytest.mark.asyncio
async def test_health_drain_waits_for_in_flight_requests():
    """/health/drain blocks until real in-flight requests complete.

    _in_flight is pre-set to 2 to simulate the drain endpoint itself (1,
    subtracted via deduct=1) plus one real in-flight request (1).  Once
    the real request finishes the adjusted count reaches 0 and drain returns.
    """
    InFlightRequestsMiddleware._in_flight = 2  # drain(1) + real in-flight(1)

    async def _release():
        await asyncio.sleep(0.15)
        InFlightRequestsMiddleware._in_flight = (
            1  # real request done; drain still running
        )

    asyncio.create_task(_release())
    result = await health_drain()

    assert result["status"] == "drained"


def test_health_drain_accessible_via_http(monkeypatch):
    """/health/drain responds over HTTP when auth dependency is satisfied."""
    app = FastAPI()
    app.include_router(_health_endpoints_module.router)

    async def _no_auth():
        return None

    app.dependency_overrides[user_api_key_auth] = _no_auth
    client = TestClient(app)

    response = client.get("/health/drain")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "drained"


def test_health_drain_does_not_self_count_via_http(monkeypatch):
    """drain subtracts itself from in-flight so it completes without exhausting the timeout.

    Without the deduct=1 fix the drain endpoint counts itself as an in-flight
    request, the poll loop never reaches zero, and the call blocks for the full
    GRACEFUL_SHUTDOWN_TIMEOUT before returning.
    """
    import time

    monkeypatch.setenv("GRACEFUL_SHUTDOWN_TIMEOUT", "2")

    app = FastAPI()
    app.add_middleware(InFlightRequestsMiddleware)
    app.include_router(_health_endpoints_module.router)

    async def _no_auth():
        return None

    app.dependency_overrides[user_api_key_auth] = _no_auth
    client = TestClient(app)

    start = time.monotonic()
    response = client.get("/health/drain")
    elapsed = time.monotonic() - start

    assert response.status_code == 200
    assert response.json()["status"] == "drained"
    assert elapsed < 1.0, f"drain self-counted: blocked {elapsed:.2f}s (expected < 1s)"


# ---------------------------------------------------------------------------
# Behavior matrix summary
# ---------------------------------------------------------------------------
#
# | Scenario                          | /readiness | /liveliness | /liveness |
# |-----------------------------------|------------|-------------|-----------|
# | Normal operation                  | 200        | 200         | 200       |
# | is_shutting_down = True           | 503        | 503         | 503       |
# | DB disconnected, not shutting down| 503        | 200         | 200       |
# | DB disconnected + shutting down   | 503*       | 503         | 503       |
# | In-flight > 0, draining           | 503        | 503         | 503       |
#
# * Returns 503 immediately without checking DB (optimises shutdown path)
