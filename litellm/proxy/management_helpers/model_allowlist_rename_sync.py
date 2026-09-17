"""
Keep the `models` allowlists on keys, teams, organizations, projects and users pointing at
deployment names that still exist.

Those allowlists store public model names, not ids, so a deployment rename that leaves them
alone denies the new name while the old entry grants a name nothing serves any more.
"""

from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from pydantic import BaseModel

from litellm.proxy.common_utils.auth_cache_invalidation_pubsub import evict_and_broadcast
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.management_helpers.access_group_model_sync import raw_executor, still_backed
from litellm.router import Router


class _TouchedRow(BaseModel):
    kind: str
    object_id: str
    team_alias: str | None = None


@dataclass(frozen=True, slots=True)
class _AllowlistTable:
    kind: str
    table: str
    id_column: str
    cache_keys: Callable[[_TouchedRow], tuple[str, ...]]
    alias_column: str | None = None

    def update_cte(self, set_clause: str, where_clause: str) -> str:
        alias: Final = f'"{self.alias_column}"' if self.alias_column else "NULL::text"
        return (
            f'{self.kind}_rows AS (UPDATE "{self.table}" SET "models" = {set_clause} WHERE {where_clause} '
            f"RETURNING '{self.kind}' AS kind, \"{self.id_column}\" AS object_id, {alias} AS team_alias)"
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
    _AllowlistTable("team", "LiteLLM_TeamTable", "team_id", _team_cache_keys, alias_column="team_alias"),
    _AllowlistTable("key", "LiteLLM_VerificationToken", "token", _key_cache_keys),
    _AllowlistTable("org", "LiteLLM_OrganizationTable", "organization_id", _org_cache_keys),
    _AllowlistTable("project", "LiteLLM_ProjectTable", "project_id", _project_cache_keys),
    _AllowlistTable("user", "LiteLLM_UserTable", "user_id", _user_cache_keys),
)

_CACHE_KEYS_BY_KIND: Final = MappingProxyType({table.kind: table.cache_keys for table in _ALLOWLIST_TABLES})


def _rewrite_sql(set_clause: str, where_clause: str) -> str:
    """One statement touching every allowlist table, so the rewrite lands everywhere or nowhere."""
    ctes: Final = ", ".join(table.update_cte(set_clause, where_clause) for table in _ALLOWLIST_TABLES)
    rows: Final = " UNION ALL ".join(
        f"SELECT kind, object_id, team_alias FROM {table.kind}_rows" for table in _ALLOWLIST_TABLES
    )
    return f"WITH {ctes} {rows}"


_REPLACE_SQL: Final = _rewrite_sql('array_replace(array_remove("models", $2), $1, $2)', '$1 = ANY("models")')

_APPEND_SQL: Final = _rewrite_sql('array_append("models", $2)', '$1 = ANY("models") AND NOT ($2 = ANY("models"))')


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
    touched_rows: Final = await executor.query_raw(
        _APPEND_SQL if old_name_still_backed else _REPLACE_SQL, old_name, new_name
    )
    touched: Final = tuple(_TouchedRow.model_validate(row) for row in touched_rows)
    await evict_and_broadcast(
        tuple(cache_key for row in touched for cache_key in _CACHE_KEYS_BY_KIND[row.kind](row)),
        user_api_key_cache,
    )
