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

    with patch(
        "litellm.proxy.proxy_server.shared_aiohttp_session", mock_session
    ):
        data = {}
        await add_shared_session_to_data(data)
        assert data["shared_session"] is mock_session


@pytest.mark.asyncio
async def test_add_shared_session_recreates_closed_session():
    """When the shared session is closed, it should be recreated."""
    from litellm.proxy import proxy_server as proxy_server_module
    from litellm.proxy.route_llm_request import add_shared_session_to_data

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
    from litellm.proxy import proxy_server as proxy_server_module
    from litellm.proxy.route_llm_request import add_shared_session_to_data

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
async def test_add_shared_session_no_session_available():
    """When no session was ever created, data should not contain shared_session."""
    from litellm.proxy.route_llm_request import add_shared_session_to_data

    with patch(
        "litellm.proxy.proxy_server.shared_aiohttp_session", None
    ):
        data = {}
        await add_shared_session_to_data(data)
        assert "shared_session" not in data
