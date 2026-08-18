"""Pin ``_cache_user_row``.

Symbols pinned here:
  - ``_cache_user_row``
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.utils import _cache_user_row


@pytest.mark.asyncio
async def test_cache_user_row_caches_on_miss(
    mock_dual_cache: Any,
) -> None:
    user_row = SimpleNamespace(
        user_id="u1", spend=2.5, max_budget=10.0, name="Alice"
    )
    user_row.model_dump_json = MagicMock(
        return_value='{"user_id":"u1","spend":2.5,"max_budget":10.0,"name":"Alice"}'
    )
    db = MagicMock()
    db.get_data = AsyncMock(return_value=user_row)

    result = await _cache_user_row("u1", mock_dual_cache, db)
    cache_key = "u1_user_api_key_user_id"
    pinned = {
        "result": result,
        "cache_value": mock_dual_cache._store[cache_key],
        "get_calls": mock_dual_cache.get_cache.call_count,
        "set_calls": mock_dual_cache.set_cache.call_count,
        "db_called": db.get_data.await_count,
    }
    assert pinned == {
        "result": None,
        "cache_value": '{"user_id":"u1","spend":2.5,"max_budget":10.0,"name":"Alice"}',
        "get_calls": 1,
        "set_calls": 1,
        "db_called": 1,
    }


@pytest.mark.asyncio
async def test_cache_user_row_skips_db_on_cache_hit(
    mock_dual_cache: Any,
) -> None:
    cache_key = "u-hit_user_api_key_user_id"
    mock_dual_cache._store[cache_key] = "cached-blob"
    db = MagicMock()
    db.get_data = AsyncMock(return_value=None)
    result = await _cache_user_row("u-hit", mock_dual_cache, db)
    assert result is None
    assert db.get_data.await_count == 0


@pytest.mark.asyncio
async def test_cache_user_row_skips_set_when_user_row_lacks_model_dump_json(
    mock_dual_cache: Any,
) -> None:
    user_row = SimpleNamespace(user_id="u2", spend=1.0)
    db = MagicMock()
    db.get_data = AsyncMock(return_value=user_row)
    await _cache_user_row("u2", mock_dual_cache, db)
    assert mock_dual_cache._store == {}
    assert mock_dual_cache.set_cache.call_count == 0


@pytest.mark.asyncio
async def test_cache_user_row_propagates_db_error(
    mock_dual_cache: Any,
) -> None:
    db = MagicMock()
    db.get_data = AsyncMock(side_effect=RuntimeError("db down"))
    with pytest.raises(RuntimeError, match="db down"):
        await _cache_user_row("u3", mock_dual_cache, db)
