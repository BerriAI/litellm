"""Proxy state and identities shared by the vector store access group tests."""

from collections.abc import Sequence
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.models.access_group import LiteLLM_AccessGroupTable
from litellm.proxy._types import (
    LiteLLM_ObjectPermissionTable,
    LiteLLM_TeamTableCachedObj,
    LitellmUserRoles,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_checks import cache_access_object
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.types.vector_stores import LiteLLM_ManagedVectorStore

TOKEN: Final = "hashed-key"
TEAM_ID: Final = "team-t"
OTHER_TEAM_ID: Final = "owner-team"


class ProxyState:
    """The proxy globals the checks read: a real cache in front of a database that holds no groups."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        group_reads: AsyncMock | None = None,
        team_reads: AsyncMock | None = None,
        store_reads: AsyncMock | None = None,
        cache: UserApiKeyCache | MagicMock | None = None,
    ) -> None:
        self.cache: Final = cache if cache is not None else UserApiKeyCache()
        self.group_reads: Final = group_reads if group_reads is not None else AsyncMock(return_value=None)
        self.prisma: Final = MagicMock(
            db=MagicMock(
                litellm_accessgrouptable=MagicMock(find_unique=self.group_reads),
                litellm_teamtable=MagicMock(find_unique=team_reads or AsyncMock(return_value=None)),
                litellm_managedvectorstorestable=MagicMock(find_unique=store_reads or AsyncMock(return_value=None)),
            )
        )
        monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", self.prisma)
        monkeypatch.setattr("litellm.proxy.proxy_server.user_api_key_cache", self.cache)

    async def add_group(
        self,
        group_id: str,
        stores: Sequence[str] = (),
        *,
        models: Sequence[str] = (),
        team_ids: Sequence[str] = (TEAM_ID,),
        key_ids: Sequence[str] = (TOKEN,),
    ) -> None:
        await cache_access_object(
            access_group_id=group_id,
            access_group_table=LiteLLM_AccessGroupTable(
                access_group_id=group_id,
                access_group_name=group_id,
                access_model_names=list(models),
                access_vector_store_ids=tuple(stores),
                assigned_team_ids=list(team_ids),
                assigned_key_ids=list(key_ids),
            ),
            user_api_key_cache=self.cache,
        )

    async def add_team(self, team: LiteLLM_TeamTableCachedObj) -> None:
        await self.cache.async_set_cache(key=f"team_id:{team.team_id}", value=team)


def own_permission(stores: Sequence[str] | None) -> LiteLLM_ObjectPermissionTable | None:
    if stores is None:
        return None
    return LiteLLM_ObjectPermissionTable(object_permission_id="own", vector_stores=list(stores))


def make_key(
    own_stores: Sequence[str] | None = None, groups: Sequence[str] = (), team_id: str | None = None
) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        token=TOKEN,
        team_id=team_id,
        access_group_ids=list(groups),
        object_permission_id="own" if own_stores is not None else None,
        object_permission=own_permission(own_stores),
        user_role=LitellmUserRoles.INTERNAL_USER,
    )


def make_team(own_stores: Sequence[str] | None = None, groups: Sequence[str] = ()) -> LiteLLM_TeamTableCachedObj:
    return LiteLLM_TeamTableCachedObj(
        team_id=TEAM_ID,
        access_group_ids=list(groups),
        object_permission_id="own" if own_stores is not None else None,
        object_permission=own_permission(own_stores),
    )


def team_owned_store(store_id: str) -> LiteLLM_ManagedVectorStore:
    return LiteLLM_ManagedVectorStore(vector_store_id=store_id, custom_llm_provider="openai", team_id=OTHER_TEAM_ID)
