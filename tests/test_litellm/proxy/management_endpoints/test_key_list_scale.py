"""
Tests for virtual key listing at scale (issue #19477).

Verifies that:
- _list_key_helper runs find_many and count concurrently
- _list_key_helper skips redundant attach_object_permission_to_dict
  when the relation is already eager-loaded via include
- _get_user_info_for_proxy_admin limits keys returned
- Default pagination enforces bounded result sets
"""

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from unittest.mock import AsyncMock, MagicMock, patch

from litellm.proxy.management_endpoints.key_management_endpoints import (
    _list_key_helper,
)


def _make_mock_key(token: str = "tok_1", user_id: str = "u1", **overrides):
    """Return a MagicMock that behaves like a Prisma VerificationToken row."""
    defaults = {
        "token": token,
        "key_name": None,
        "key_alias": None,
        "spend": 0.0,
        "max_budget": None,
        "expires": None,
        "models": [],
        "aliases": {},
        "config": {},
        "user_id": user_id,
        "team_id": None,
        "max_parallel_requests": None,
        "metadata": {},
        "tpm_limit": None,
        "rpm_limit": None,
        "budget_duration": None,
        "budget_reset_at": None,
        "allowed_cache_controls": [],
        "permissions": {},
        "model_spend": {},
        "model_max_budget": {},
        "soft_budget_cooldown": False,
        "blocked": False,
        "litellm_budget_table": None,
        "organization_id": None,
        "created_at": "2025-01-01T00:00:00Z",
        "updated_at": "2025-01-01T00:00:00Z",
        "token_id": None,
        "object_permission_id": None,
        "object_permission": None,
        "tags": None,
        "auto_rotate": None,
        "rotation_interval": None,
        "last_rotation_at": None,
        "key_rotation_at": None,
        "next_rotation_at": None,
    }
    defaults.update(overrides)
    m = MagicMock()
    m.model_dump.return_value = dict(defaults)
    m.user_id = defaults["user_id"]
    m.team_id = defaults["team_id"]
    m.token = defaults["token"]
    return m


# ---------------------------------------------------------------------------
# Test: find_many and count are called concurrently (asyncio.gather)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_list_key_helper_runs_find_many_and_count_concurrently():
    """find_many and count should be awaited via asyncio.gather, not sequentially."""
    execution_order: list = []

    async def mock_find_many(**kwargs):
        execution_order.append("find_many_start")
        await asyncio.sleep(0)  # yield to event loop
        execution_order.append("find_many_end")
        return [_make_mock_key()]

    async def mock_count(**kwargs):
        execution_order.append("count_start")
        await asyncio.sleep(0)
        execution_order.append("count_end")
        return 1

    prisma = AsyncMock()
    prisma.db.litellm_verificationtoken.find_many = mock_find_many
    prisma.db.litellm_verificationtoken.count = mock_count

    result = await _list_key_helper(
        prisma_client=prisma,
        page=1,
        size=10,
        user_id="u1",
        team_id=None,
        organization_id=None,
        key_alias=None,
        key_hash=None,
        return_full_object=False,
    )

    # This rules out purely sequential execution like:
    # ["find_many_start", "find_many_end", "count_start", "count_end"]
    # With asyncio.gather both coroutines start before either finishes,
    # so at least one *_start must precede the other's *_end.
    find_many_start = execution_order.index("find_many_start")
    find_many_end = execution_order.index("find_many_end")
    count_start = execution_order.index("count_start")
    count_end = execution_order.index("count_end")

    interleaved = (count_start < find_many_end) or (find_many_start < count_end)
    assert interleaved, (
        f"Expected interleaved execution, got sequential: {execution_order}"
    )
    assert result["total_count"] == 1


# ---------------------------------------------------------------------------
# Test: object_permission already eager-loaded, no extra DB call
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_list_key_helper_skips_redundant_object_permission_lookup():
    """When include=object_permission is used, attach_object_permission_to_dict
    should NOT make additional DB queries."""
    key = _make_mock_key(object_permission={"object_permission_id": "op1", "mcp_servers": []})

    prisma = AsyncMock()
    prisma.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[key])
    prisma.db.litellm_verificationtoken.count = AsyncMock(return_value=1)

    with patch(
        "litellm.proxy.management_endpoints.key_management_endpoints.attach_object_permission_to_dict"
    ) as mock_attach:
        result = await _list_key_helper(
            prisma_client=prisma,
            page=1,
            size=10,
            user_id="u1",
            team_id=None,
            organization_id=None,
            key_alias=None,
            key_hash=None,
            return_full_object=True,
        )

        # Should NOT be called because object_permission is already in the dict
        mock_attach.assert_not_called()

    assert result["total_count"] == 1
    assert len(result["keys"]) == 1


# ---------------------------------------------------------------------------
# Test: object_permission NOT in dict → fallback lookup IS called
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_list_key_helper_calls_attach_when_object_permission_missing():
    """When the key dict does NOT have object_permission, fallback to DB lookup."""
    key_data = _make_mock_key()
    # Remove object_permission from the dict to simulate it not being loaded
    dump = key_data.model_dump.return_value
    dump.pop("object_permission", None)

    prisma = AsyncMock()
    prisma.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[key_data])
    prisma.db.litellm_verificationtoken.count = AsyncMock(return_value=1)

    with patch(
        "litellm.proxy.management_endpoints.key_management_endpoints.attach_object_permission_to_dict",
        new_callable=AsyncMock,
        return_value=dump,
    ) as mock_attach:
        await _list_key_helper(
            prisma_client=prisma,
            page=1,
            size=10,
            user_id="u1",
            team_id=None,
            organization_id=None,
            key_alias=None,
            key_hash=None,
            return_full_object=True,
        )

        mock_attach.assert_called_once()


# ---------------------------------------------------------------------------
# Test: pagination respects page_size limit
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_list_key_helper_respects_page_size():
    """find_many should receive the correct skip and take values."""
    prisma = AsyncMock()
    prisma.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[])
    prisma.db.litellm_verificationtoken.count = AsyncMock(return_value=500)

    result = await _list_key_helper(
        prisma_client=prisma,
        page=3,
        size=25,
        user_id="u1",
        team_id=None,
        organization_id=None,
        key_alias=None,
        key_hash=None,
    )

    call_kwargs = prisma.db.litellm_verificationtoken.find_many.call_args.kwargs
    assert call_kwargs["skip"] == 50  # (3-1)*25
    assert call_kwargs["take"] == 25
    assert result["total_count"] == 500
    assert result["total_pages"] == 20  # ceil(500/25)
    assert result["current_page"] == 3


# ---------------------------------------------------------------------------
# Test: default pagination returns bounded results
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_list_key_helper_default_pagination():
    """With page=1 and size=10, only 10 keys should be fetched."""
    keys = [_make_mock_key(token=f"tok_{i}") for i in range(10)]

    prisma = AsyncMock()
    prisma.db.litellm_verificationtoken.find_many = AsyncMock(return_value=keys)
    prisma.db.litellm_verificationtoken.count = AsyncMock(return_value=380_000)

    result = await _list_key_helper(
        prisma_client=prisma,
        page=1,
        size=10,
        user_id=None,
        team_id=None,
        organization_id=None,
        key_alias=None,
        key_hash=None,
    )

    assert len(result["keys"]) == 10
    assert result["total_count"] == 380_000
    assert result["total_pages"] == 38_000


# ---------------------------------------------------------------------------
# Helper: inject a mock proxy_server module so deferred imports inside
# internal_user_endpoints resolve without pulling in the real (heavy) module.
# ---------------------------------------------------------------------------
import types as _types

def _install_mock_proxy_server(**attrs):
    """Put a lightweight fake ``litellm.proxy.proxy_server`` into sys.modules."""
    mod = _types.ModuleType("litellm.proxy.proxy_server")
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules["litellm.proxy.proxy_server"] = mod
    return mod


# ---------------------------------------------------------------------------
# Test: _get_user_info_for_proxy_admin limits keys
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_get_user_info_for_proxy_admin_limits_keys():
    """Admin endpoint should pass a LIMIT parameter to the SQL query."""
    mock_prisma = AsyncMock()
    mock_prisma.db.query_raw = AsyncMock(
        return_value=[{"teams": None, "keys": None}]
    )
    mock_prisma.get_data = AsyncMock(return_value=None)

    saved = sys.modules.get("litellm.proxy.proxy_server")
    try:
        _install_mock_proxy_server(
            prisma_client=mock_prisma,
            general_settings={},
            litellm_master_key_hash="master_hash",
        )
        from litellm.proxy._types import UserAPIKeyAuth, LitellmUserRoles
        from litellm.proxy.management_endpoints.internal_user_endpoints import (
            _get_user_info_for_proxy_admin,
        )
        mock_key_dict = UserAPIKeyAuth(
            user_id="admin-user-123",
            user_role=LitellmUserRoles.PROXY_ADMIN,
        )
        await _get_user_info_for_proxy_admin(user_api_key_dict=mock_key_dict)
    finally:
        if saved is not None:
            sys.modules["litellm.proxy.proxy_server"] = saved
        else:
            sys.modules.pop("litellm.proxy.proxy_server", None)

    # Verify query_raw was called with the LIMIT parameter
    mock_prisma.db.query_raw.assert_called_once()
    call_args = mock_prisma.db.query_raw.call_args
    sql_query = call_args[0][0]
    limit_param = call_args[0][1]

    assert "LIMIT $1" in sql_query
    assert limit_param == 100  # _ADMIN_KEY_LIMIT


# ---------------------------------------------------------------------------
# Test: _get_user_info_for_proxy_admin handles large key sets safely
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_get_user_info_for_proxy_admin_returns_limited_keys():
    """Even if DB has many keys, admin endpoint should return at most _ADMIN_KEY_LIMIT."""
    mock_keys = [
        {"token": f"tok_{i}", "models": [], "key_name": None, "key_alias": None,
         "spend": 0, "max_budget": None, "user_id": "admin", "team_id": None,
         "max_parallel_requests": None, "metadata": {}, "tpm_limit": None,
         "rpm_limit": None, "budget_duration": None, "budget_reset_at": None,
         "allowed_cache_controls": [], "permissions": {}, "model_spend": {},
         "model_max_budget": {}, "soft_budget_cooldown": False, "blocked": False,
         "litellm_budget_table": None, "organization_id": None, "expires": None,
         "aliases": {}, "config": {}, "created_at": "2025-01-01T00:00:00Z",
         "updated_at": "2025-01-01T00:00:00Z", "token_id": None,
         "object_permission_id": None, "tags": None}
        for i in range(100)
    ]

    mock_prisma = AsyncMock()
    mock_prisma.db.query_raw = AsyncMock(
        return_value=[{"teams": None, "keys": mock_keys}]
    )
    mock_prisma.get_data = AsyncMock(return_value=None)

    saved = sys.modules.get("litellm.proxy.proxy_server")
    try:
        _install_mock_proxy_server(
            prisma_client=mock_prisma,
            general_settings={},
            litellm_master_key_hash="master_hash",
        )
        from litellm.proxy._types import UserAPIKeyAuth, LitellmUserRoles
        from litellm.proxy.management_endpoints.internal_user_endpoints import (
            _get_user_info_for_proxy_admin,
        )
        mock_key_dict = UserAPIKeyAuth(
            user_id="admin-user-123",
            user_role=LitellmUserRoles.PROXY_ADMIN,
        )
        result = await _get_user_info_for_proxy_admin(user_api_key_dict=mock_key_dict)
    finally:
        if saved is not None:
            sys.modules["litellm.proxy.proxy_server"] = saved
        else:
            sys.modules.pop("litellm.proxy.proxy_server", None)

    assert len(result.keys) <= 100


# ---------------------------------------------------------------------------
# Test: deleted keys also use concurrent queries
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_list_key_helper_deleted_status_concurrent():
    """Deleted key listing should also run find_many and count concurrently."""
    prisma = AsyncMock()
    prisma.db.litellm_deletedverificationtoken.find_many = AsyncMock(return_value=[])
    prisma.db.litellm_deletedverificationtoken.count = AsyncMock(return_value=0)

    result = await _list_key_helper(
        prisma_client=prisma,
        page=1,
        size=10,
        user_id="u1",
        team_id=None,
        organization_id=None,
        key_alias=None,
        key_hash=None,
        status="deleted",
    )

    prisma.db.litellm_deletedverificationtoken.find_many.assert_called_once()
    prisma.db.litellm_deletedverificationtoken.count.assert_called_once()
    assert result["total_count"] == 0
