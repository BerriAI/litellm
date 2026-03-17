"""
Tests for shared aiohttp session auto-recovery.

When the shared session closes (e.g. network interruption, idle timeout),
add_shared_session_to_data should recreate it instead of permanently
falling back to per-request connections.

Fixes: https://github.com/BerriAI/litellm/issues/23806
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_add_shared_session_attaches_open_session():
    """When the shared session is open, it should be attached to data."""
    from litellm.proxy.route_llm_request import add_shared_session_to_data

    mock_session = MagicMock()
    mock_session.closed = False

    with patch("litellm.proxy.proxy_server.shared_aiohttp_session", mock_session):
        data = {}
        await add_shared_session_to_data(data)
        assert data["shared_session"] is mock_session


@pytest.mark.asyncio
async def test_add_shared_session_recreates_closed_session():
    """When the shared session is closed, it should be recreated."""
    import litellm.proxy.route_llm_request as route_module
    from litellm.proxy import proxy_server as proxy_server_module
    from litellm.proxy.route_llm_request import add_shared_session_to_data

    # Reset the module-level lock so each test uses the current event loop
    route_module._shared_session_lock = None

    closed_session = MagicMock()
    closed_session.closed = True

    new_session = MagicMock()
    new_session.closed = False

    with patch.object(
        proxy_server_module,
        "shared_aiohttp_session",
        closed_session,
    ):
        with patch.object(
            proxy_server_module,
            "_initialize_shared_aiohttp_session",
            new_callable=AsyncMock,
            return_value=new_session,
        ) as mock_init:
            data = {}
            await add_shared_session_to_data(data)

            mock_init.assert_called_once()
            assert data["shared_session"] is new_session
            assert proxy_server_module.shared_aiohttp_session is new_session


@pytest.mark.asyncio
async def test_add_shared_session_handles_recreation_failure():
    """When recreation fails, data should not contain shared_session."""
    import litellm.proxy.route_llm_request as route_module
    from litellm.proxy import proxy_server as proxy_server_module
    from litellm.proxy.route_llm_request import add_shared_session_to_data

    # Reset the module-level lock so each test uses the current event loop
    route_module._shared_session_lock = None

    closed_session = MagicMock()
    closed_session.closed = True

    with patch.object(
        proxy_server_module,
        "shared_aiohttp_session",
        closed_session,
    ):
        with patch.object(
            proxy_server_module,
            "_initialize_shared_aiohttp_session",
            new_callable=AsyncMock,
            return_value=None,
        ):
            data = {}
            await add_shared_session_to_data(data)
            assert "shared_session" not in data


@pytest.mark.asyncio
async def test_add_shared_session_handles_recreation_exception():
    """When _initialize_shared_aiohttp_session raises, data should not contain shared_session."""
    import litellm.proxy.route_llm_request as route_module
    from litellm.proxy import proxy_server as proxy_server_module
    from litellm.proxy.route_llm_request import add_shared_session_to_data

    # Reset the module-level lock so each test uses the current event loop
    route_module._shared_session_lock = None

    closed_session = MagicMock()
    closed_session.closed = True

    with patch.object(
        proxy_server_module,
        "shared_aiohttp_session",
        closed_session,
    ):
        with patch.object(
            proxy_server_module,
            "_initialize_shared_aiohttp_session",
            new_callable=AsyncMock,
            side_effect=RuntimeError("connection pool exhausted"),
        ):
            data = {}
            await add_shared_session_to_data(data)
            # Should gracefully handle exception — no shared_session attached
            assert "shared_session" not in data


@pytest.mark.asyncio
async def test_add_shared_session_no_session_available():
    """When no session was ever created, data should not contain shared_session."""
    from litellm.proxy.route_llm_request import add_shared_session_to_data

    with patch("litellm.proxy.proxy_server.shared_aiohttp_session", None):
        data = {}
        await add_shared_session_to_data(data)
        assert "shared_session" not in data


@pytest.mark.asyncio
async def test_add_shared_session_concurrent_recreation_uses_lock():
    """When multiple coroutines detect a closed session concurrently,
    only one should recreate it (double-checked locking via asyncio.Lock)."""
    import litellm.proxy.route_llm_request as route_module
    from litellm.proxy import proxy_server as proxy_server_module
    from litellm.proxy.route_llm_request import add_shared_session_to_data

    # Reset the module-level lock so each test is isolated
    route_module._shared_session_lock = None

    closed_session = MagicMock()
    closed_session.closed = True

    new_session = MagicMock()
    new_session.closed = False

    call_count = 0

    async def mock_init():
        nonlocal call_count
        call_count += 1
        # Simulate some async work
        await asyncio.sleep(0.01)
        return new_session

    with patch.object(
        proxy_server_module,
        "shared_aiohttp_session",
        closed_session,
    ):
        with patch.object(
            proxy_server_module,
            "_initialize_shared_aiohttp_session",
            side_effect=mock_init,
        ):
            # Launch 5 concurrent calls
            results = [{} for _ in range(5)]
            await asyncio.gather(*(add_shared_session_to_data(d) for d in results))

            # Only 1 coroutine should have called _initialize (the rest see the
            # re-checked session as open under the lock)
            assert call_count == 1, f"Expected 1 init call, got {call_count}"
            # All should have the new session
            for d in results:
                assert d.get("shared_session") is new_session
