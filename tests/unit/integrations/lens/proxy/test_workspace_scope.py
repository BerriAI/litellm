from types import SimpleNamespace
from typing import TYPE_CHECKING, Final, cast
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from litellm.integrations.lens.generated.types import TraceScope
from litellm.integrations.lens.proxy.workspace_scope import (
    MAX_WORKSPACE_TRACE_KEYS,
    TraceWorkspaceFilters,
    load_workspace_key_hashes,
    workspace_trace_scope,
)

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient


@pytest.mark.asyncio
async def test_organization_and_project_match_current_key_assignments_with_team_fallback() -> None:
    teams: Final = AsyncMock(return_value=(SimpleNamespace(team_id="org-team"),))
    keys: Final = AsyncMock(return_value=(SimpleNamespace(token="hash-a"), SimpleNamespace(token="hash-b")))
    client: Final = cast(
        "PrismaClient",
        SimpleNamespace(
            db=SimpleNamespace(
                litellm_teamtable=SimpleNamespace(find_many=teams),
                litellm_verificationtoken=SimpleNamespace(find_many=keys),
            )
        ),
    )
    result: Final = await load_workspace_key_hashes(
        TraceWorkspaceFilters("org", "selected-team", "project", "selected-key"), prisma_client=client
    )
    assert result == ("hash-a", "hash-b")
    teams.assert_awaited_once_with(
        where={"organization_id": "org", "team_id": "selected-team"}, take=MAX_WORKSPACE_TRACE_KEYS + 1
    )
    keys.assert_awaited_once_with(
        where={
            "AND": (
                {
                    "OR": (
                        {"organization_id": "org"},
                        {"organization_id": None, "team_id": {"in": ("org-team",)}},
                    )
                },
                {"project_id": "project"},
                {"team_id": "selected-team"},
                {"token": "selected-key"},
            )
        },
        take=MAX_WORKSPACE_TRACE_KEYS + 1,
        order={"token": "asc"},
    )


@pytest.mark.asyncio
async def test_workspace_lookup_rejects_a_partial_result_instead_of_truncating() -> None:
    keys: Final = AsyncMock(return_value=(SimpleNamespace(token="hash"),) * (MAX_WORKSPACE_TRACE_KEYS + 1))
    client: Final = cast(
        "PrismaClient", SimpleNamespace(db=SimpleNamespace(litellm_verificationtoken=SimpleNamespace(find_many=keys)))
    )
    with pytest.raises(HTTPException) as error:
        await load_workspace_key_hashes(TraceWorkspaceFilters(project_id="project"), prisma_client=client)
    assert error.value.status_code == 413


@pytest.mark.asyncio
async def test_workspace_filter_without_a_database_fails_closed() -> None:
    with pytest.raises(HTTPException) as error:
        await load_workspace_key_hashes(TraceWorkspaceFilters(project_id="project"), prisma_client=None)
    assert error.value.status_code == 503


@pytest.mark.asyncio
async def test_an_empty_workspace_keeps_deny_all_and_the_authenticated_scope() -> None:
    async def lookup(filters: TraceWorkspaceFilters) -> tuple[str, ...]:
        return ()

    scope: Final[TraceScope] = {"all_teams": 0, "user_id": "reader", "team_ids": ("permitted-team",)}
    result: Final = await workspace_trace_scope(scope, TraceWorkspaceFilters(project_id="empty-project"), lookup)
    assert result == {
        **scope,
        "filter_team_id": "",
        "filter_api_key_hash": "",
        "filter_api_key_hashes_mode": "only",
        "filter_api_key_hashes": (),
    }
