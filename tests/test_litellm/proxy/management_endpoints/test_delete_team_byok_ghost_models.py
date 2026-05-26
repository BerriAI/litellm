"""
Regression tests for GH #22594 / LIT-2120: deleting a team-scoped BYOK model
must remove its **public** model name from `team.models`, even when the team's
`litellm_modeltable.model_aliases` map does not contain a reverse entry.

Before the fix in delete_model:
  - `model_params.model_name` (passed as `public_model_name` to
    `delete_team_model_alias`) is the INTERNAL unique name
    `model_name_{team_id}_{uuid}`, not the user-visible public name.
  - The alias lookup misses, `valid_team_model_aliases` is empty, and the
    public name remains in `team.models` -> "ghost" entry in /models.

After the fix:
  - We resolve `model_info.team_public_model_name` and use it for the alias
    lookup AND defensively drop it from `team.models` regardless of the
    alias-map state. We also refresh the team cache so the ghost doesn't
    linger for the TTL window.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy._types import (
    LiteLLM_ProxyModelTable,
    LitellmUserRoles,
    UserAPIKeyAuth,
)


_PS = "litellm.proxy.proxy_server"
_MM = "litellm.proxy.management_endpoints.model_management_endpoints"


def _build_db_row(model_id: str, team_id: str, public_name: str, internal_name: str):
    return LiteLLM_ProxyModelTable(
        model_id=model_id,
        model_name=internal_name,
        litellm_params={"model": "openai/my-fake-model", "api_key": "fake-key"},
        model_info={
            "id": model_id,
            "team_id": team_id,
            "team_public_model_name": public_name,
        },
        created_by="admin",
        updated_by="admin",
    )


def _build_mock_prisma(
    db_row: LiteLLM_ProxyModelTable,
    team_models: list,
    modeltable_rows: list,
):
    team_row = MagicMock()
    team_row.models = list(team_models)
    team_row.model_dump = MagicMock(
        return_value={"team_id": "team-xyz", "models": list(team_models)}
    )

    mock_prisma = MagicMock()
    mock_prisma.db = MagicMock()
    mock_prisma.db.litellm_modeltable = MagicMock()
    mock_prisma.db.litellm_modeltable.find_many = AsyncMock(return_value=modeltable_rows)
    mock_prisma.db.litellm_modeltable.update = AsyncMock(return_value=None)
    mock_prisma.db.litellm_teamtable = MagicMock()
    mock_prisma.db.litellm_teamtable.find_unique = AsyncMock(return_value=team_row)
    mock_prisma.db.litellm_teamtable.update = AsyncMock(return_value=team_row)
    mock_prisma.db.litellm_proxymodeltable = MagicMock()
    mock_prisma.db.litellm_proxymodeltable.find_unique = AsyncMock(return_value=db_row)
    mock_prisma.db.litellm_proxymodeltable.delete = AsyncMock(return_value=db_row)
    return mock_prisma, team_row


@pytest.mark.asyncio
async def test_delete_team_byok_model_removes_public_name_from_team_models():
    """BYOK models never populate `model_aliases`, so the alias-lookup path
    finds nothing. The fix must still strip the public name from `team.models`."""
    from litellm.proxy.management_endpoints.model_management_endpoints import (
        ModelInfoDelete,
        delete_model as delete_model_endpoint,
    )

    team_id = "team-ghost-1"
    public_name = "byok-public-model"
    internal_name = f"model_name_{team_id}_uuid-xyz"
    model_id = "ghost-model-id-1"

    db_row = _build_db_row(model_id, team_id, public_name, internal_name)
    mock_prisma, _ = _build_mock_prisma(
        db_row=db_row,
        team_models=[public_name, "always-allowed-model"],
        modeltable_rows=[],
    )

    admin = UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN)

    with (
        patch(f"{_PS}.prisma_client", mock_prisma),
        patch(f"{_PS}.llm_router", MagicMock()),
        patch(f"{_PS}.store_model_in_db", True),
        patch(f"{_PS}.premium_user", True),
        patch(f"{_PS}.proxy_logging_obj", MagicMock()),
        patch(f"{_PS}.user_api_key_cache", MagicMock()),
        patch(
            f"{_MM}.ModelManagementAuthChecks.can_user_make_model_call",
            new=AsyncMock(return_value=None),
        ),
        patch(f"{_MM}._refresh_cached_team", new=AsyncMock(return_value=None)),
    ):
        result = await delete_model_endpoint(
            model_info=ModelInfoDelete(id=model_id),
            user_api_key_dict=admin,
        )

    assert "deleted successfully" in result["message"]

    mock_prisma.db.litellm_teamtable.update.assert_awaited_once()
    update_call = mock_prisma.db.litellm_teamtable.update.await_args
    updated_models = update_call.kwargs["data"]["models"]
    assert public_name not in updated_models, (
        f"Ghost! public_name={public_name!r} should be stripped from team.models; got {updated_models!r}"
    )
    assert "always-allowed-model" in updated_models, "Unrelated models must be preserved"


@pytest.mark.asyncio
async def test_delete_team_byok_model_uses_team_public_name_for_alias_lookup():
    """The fix must pass `team_public_model_name` (not `model_name`) to
    `delete_team_model_alias` so the alias map IS hit when it's populated."""
    from litellm.proxy.management_endpoints.model_management_endpoints import (
        ModelInfoDelete,
        delete_model as delete_model_endpoint,
    )

    team_id = "team-ghost-2"
    public_name = "byok-public-with-alias"
    internal_name = f"model_name_{team_id}_uuid-abc"
    model_id = "ghost-model-id-2"

    modeltable_row = MagicMock()
    modeltable_row.id = 7
    modeltable_row.model_aliases = {"some-alias": public_name}
    modeltable_row.team = MagicMock()
    modeltable_row.team.team_id = team_id

    db_row = _build_db_row(model_id, team_id, public_name, internal_name)
    mock_prisma, _ = _build_mock_prisma(
        db_row=db_row,
        team_models=[public_name],
        modeltable_rows=[modeltable_row],
    )

    admin = UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN)

    with (
        patch(f"{_PS}.prisma_client", mock_prisma),
        patch(f"{_PS}.llm_router", MagicMock()),
        patch(f"{_PS}.store_model_in_db", True),
        patch(f"{_PS}.premium_user", True),
        patch(f"{_PS}.proxy_logging_obj", MagicMock()),
        patch(f"{_PS}.user_api_key_cache", MagicMock()),
        patch(
            f"{_MM}.ModelManagementAuthChecks.can_user_make_model_call",
            new=AsyncMock(return_value=None),
        ),
        patch(f"{_MM}._refresh_cached_team", new=AsyncMock(return_value=None)),
    ):
        await delete_model_endpoint(
            model_info=ModelInfoDelete(id=model_id),
            user_api_key_dict=admin,
        )

    mock_prisma.db.litellm_modeltable.update.assert_awaited_once()
    update_call = mock_prisma.db.litellm_teamtable.update.await_args
    updated_models = update_call.kwargs["data"]["models"]
    assert public_name not in updated_models


@pytest.mark.asyncio
async def test_delete_team_byok_model_triggers_team_cache_refresh():
    """After updating the team row, the team cache must be refreshed so the
    ghost doesn't linger for the in-memory TTL on subsequent /models calls."""
    from litellm.proxy.management_endpoints.model_management_endpoints import (
        ModelInfoDelete,
        delete_model as delete_model_endpoint,
    )

    team_id = "team-ghost-3"
    public_name = "byok-cache-refresh"
    internal_name = f"model_name_{team_id}_uuid-ddd"
    model_id = "ghost-model-id-3"

    db_row = _build_db_row(model_id, team_id, public_name, internal_name)
    mock_prisma, _ = _build_mock_prisma(
        db_row=db_row,
        team_models=[public_name],
        modeltable_rows=[],
    )

    refresh_mock = AsyncMock(return_value=None)
    admin = UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN)

    with (
        patch(f"{_PS}.prisma_client", mock_prisma),
        patch(f"{_PS}.llm_router", MagicMock()),
        patch(f"{_PS}.store_model_in_db", True),
        patch(f"{_PS}.premium_user", True),
        patch(f"{_PS}.proxy_logging_obj", MagicMock()),
        patch(f"{_PS}.user_api_key_cache", MagicMock()),
        patch(
            f"{_MM}.ModelManagementAuthChecks.can_user_make_model_call",
            new=AsyncMock(return_value=None),
        ),
        patch(f"{_MM}._refresh_cached_team", new=refresh_mock),
    ):
        await delete_model_endpoint(
            model_info=ModelInfoDelete(id=model_id),
            user_api_key_dict=admin,
        )

    refresh_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_delete_team_byok_model_resilient_to_cache_refresh_exception():
    """If team cache refresh raises, the delete should still report success
    (the DB row was already deleted; the cache will refresh on its own TTL)."""
    from litellm.proxy.management_endpoints.model_management_endpoints import (
        ModelInfoDelete,
        delete_model as delete_model_endpoint,
    )

    team_id = "team-ghost-4"
    public_name = "byok-flaky-cache"
    internal_name = f"model_name_{team_id}_uuid-eee"
    model_id = "ghost-model-id-4"

    db_row = _build_db_row(model_id, team_id, public_name, internal_name)
    mock_prisma, _ = _build_mock_prisma(
        db_row=db_row,
        team_models=[public_name],
        modeltable_rows=[],
    )

    admin = UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN)

    with (
        patch(f"{_PS}.prisma_client", mock_prisma),
        patch(f"{_PS}.llm_router", MagicMock()),
        patch(f"{_PS}.store_model_in_db", True),
        patch(f"{_PS}.premium_user", True),
        patch(f"{_PS}.proxy_logging_obj", MagicMock()),
        patch(f"{_PS}.user_api_key_cache", MagicMock()),
        patch(
            f"{_MM}.ModelManagementAuthChecks.can_user_make_model_call",
            new=AsyncMock(return_value=None),
        ),
        patch(
            f"{_MM}._refresh_cached_team",
            new=AsyncMock(side_effect=RuntimeError("redis is down")),
        ),
    ):
        result = await delete_model_endpoint(
            model_info=ModelInfoDelete(id=model_id),
            user_api_key_dict=admin,
        )

    assert "deleted successfully" in result["message"]
    mock_prisma.db.litellm_teamtable.update.assert_awaited_once()
