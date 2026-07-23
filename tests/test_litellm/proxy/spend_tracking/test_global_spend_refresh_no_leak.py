"""Regression test for the Prisma connection leak in global_spend_refresh.

The REFRESH MATERIALIZED VIEW branch builds a dedicated PrismaClient with a long
timeout (the refresh can take a while on large spend tables) and connects it, but
it never disconnected the client. Every /global/spend/refresh call therefore
leaked a DB connection until the pool was exhausted, causing 500s on all auth.

The fix wraps the refresh in try/finally so the dedicated client is always
disconnected. This test locks that in: the client is disconnected both on the
success path and when the refresh query raises.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_global_spend_refresh_disconnects_client_on_success(monkeypatch):
    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy.spend_tracking.spend_management_endpoints import (
        global_spend_refresh,
    )

    # Singleton gate + materialized-view probe both succeed.
    fake_singleton = MagicMock()
    fake_singleton.db = MagicMock()
    fake_singleton.db.query_raw = AsyncMock(
        return_value=[{"relname": "MonthlyGlobalSpend", "relkind": "m"}]
    )
    monkeypatch.setattr(proxy_server, "prisma_client", fake_singleton)
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost:5432/db")

    # The dedicated refresh client: connect + query succeed.
    fake_client = MagicMock()
    fake_client.db = MagicMock()
    fake_client.db.connect = AsyncMock()
    fake_client.db.query_raw = AsyncMock(return_value=None)
    fake_client.db.disconnect = AsyncMock()

    with patch(
        "litellm.proxy.utils.PrismaClient", return_value=fake_client
    ):
        result = await global_spend_refresh()

    assert result["status"] == "success"
    # The leak fix: the dedicated client is always disconnected.
    fake_client.db.disconnect.assert_awaited_once()


@pytest.mark.asyncio
async def test_global_spend_refresh_disconnects_client_on_failure(monkeypatch):
    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy.spend_tracking.spend_management_endpoints import (
        global_spend_refresh,
    )

    fake_singleton = MagicMock()
    fake_singleton.db = MagicMock()
    fake_singleton.db.query_raw = AsyncMock(
        return_value=[{"relname": "MonthlyGlobalSpend", "relkind": "m"}]
    )
    monkeypatch.setattr(proxy_server, "prisma_client", fake_singleton)
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost:5432/db")

    # The dedicated refresh client: the REFRESH query raises (e.g. timeout).
    fake_client = MagicMock()
    fake_client.db = MagicMock()
    fake_client.db.connect = AsyncMock()
    fake_client.db.query_raw = AsyncMock(side_effect=Exception("refresh timed out"))
    fake_client.db.disconnect = AsyncMock()

    with patch(
        "litellm.proxy.utils.PrismaClient", return_value=fake_client
    ):
        result = await global_spend_refresh()

    assert result["status"] == "failure"
    # Even when the refresh fails, the client must not leak.
    fake_client.db.disconnect.assert_awaited_once()
