"""
Keep the `models` allowlists on keys, teams, organizations, projects and users pointing at
deployment names that still exist.

Those allowlists store public model names, not ids, so a deployment rename that leaves them
alone denies the new name while the old entry grants a name nothing serves any more.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel

from litellm.proxy.common_utils.auth_cache_invalidation_pubsub import evict_and_broadcast
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.management_helpers.access_group_model_sync import RawExecutor, raw_executor, still_backed
from litellm.router import Router


class _TouchedRow(BaseModel):
    object_id: str
    team_alias: str | None = None


@dataclass(frozen=True, slots=True)
class _AllowlistTable:
    table: str
    returning: str
    cache_keys: Callable[[_TouchedRow], tuple[str, ...]]

    def replace_sql(self) -> str:
        return (
            f'UPDATE "{self.table}" SET "models" = array_replace(array_remove("models", $2), $1, $2) '
            f'WHERE $1 = ANY("models") RETURNING {self.returning}'
        )

    def append_sql(self) -> str:
        return (
            f'UPDATE "{self.table}" SET "models" = array_append("models", $2) '
            f'WHERE $1 = ANY("models") AND NOT ($2 = ANY("models")) RETURNING {self.returning}'
        )


def _team_cache_keys(row: _TouchedRow) -> tuple[str, ...]:
    return (f"team_id:{row.object_id}", *((f"team_alias:{row.team_alias}",) if row.team_alias else ()))


def _key_cache_keys(row: _TouchedRow) -> tuple[str, ...]:
    return (row.object_id,)


def _org_cache_keys(row: _TouchedRow) -> tuple[str, ...]:
    return (f"org_id:{row.object_id}", f"org_id:{row.object_id}:with_budget")


def _project_cache_keys(row: _TouchedRow) -> tuple[str, ...]:
    return (f"project_id:{row.object_id}",)


def _user_cache_keys(row: _TouchedRow) -> tuple[str, ...]:
    return (row.object_id,)


_ALLOWLIST_TABLES: Final = (
    _AllowlistTable("LiteLLM_TeamTable", '"team_id" AS object_id, "team_alias"', _team_cache_keys),
    _AllowlistTable("LiteLLM_VerificationToken", '"token" AS object_id', _key_cache_keys),
    _AllowlistTable("LiteLLM_OrganizationTable", '"organization_id" AS object_id', _org_cache_keys),
    _AllowlistTable("LiteLLM_ProjectTable", '"project_id" AS object_id', _project_cache_keys),
    _AllowlistTable("LiteLLM_UserTable", '"user_id" AS object_id', _user_cache_keys),
)


async def _rewrite_allowlist(
    executor: RawExecutor,
    allowlist: _AllowlistTable,
    sql: str,
    old_name: str,
    new_name: str,
    user_api_key_cache: UserApiKeyCache,
) -> None:
    touched_rows: Final = await executor.query_raw(sql, old_name, new_name)
    await evict_and_broadcast(
        tuple(cache_key for row in touched_rows for cache_key in allowlist.cache_keys(_TouchedRow.model_validate(row))),
        user_api_key_cache,
    )


async def sync_model_allowlists_for_renamed_model(
    prisma_client: object,
    *,
    model_id: str,
    old_name: str,
    new_name: str,
    llm_router: Router | None,
    user_api_key_cache: UserApiKeyCache,
) -> None:
    if old_name == new_name:
        return
    executor: Final = raw_executor(prisma_client)
    old_name_still_backed: Final = await still_backed(executor, llm_router, old_name, model_id)
    for allowlist in _ALLOWLIST_TABLES:
        await _rewrite_allowlist(
            executor,
            allowlist,
            allowlist.append_sql() if old_name_still_backed else allowlist.replace_sql(),
            old_name,
            new_name,
            user_api_key_cache,
        )
