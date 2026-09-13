"""
Keep `litellm_accessgrouptable.access_model_names` pointing at deployment names that still exist.

Unified access groups store model names, not ids, so a deployment rename or delete that leaves
the arrays alone strands every group on a name nothing serves any more.
"""

from collections.abc import Sequence
from typing import Final, Protocol

from pydantic import BaseModel

from litellm.proxy.db.routing_prisma_wrapper import WriterPinnedClient
from litellm.proxy.management_helpers.access_group_team_sync import invalidate_access_group_caches
from litellm.repositories.table_repositories import AccessGroupRepository
from litellm.router import Router


class _TouchedGroupRow(BaseModel):
    access_group_id: str


class _DeploymentCountRow(BaseModel):
    deployment_count: int


class _RawExecutor(Protocol):
    async def query_raw(self, query: str, *args: str) -> Sequence[object]: ...


_BACKING_DEPLOYMENTS_SQL: Final = (
    'SELECT COUNT(*)::int AS deployment_count FROM "LiteLLM_ProxyModelTable" WHERE "model_name" = $1'
)

_REPLACE_MODEL_NAME_SQL: Final = (
    'UPDATE "LiteLLM_AccessGroupTable" '
    'SET "access_model_names" = array_replace(array_remove("access_model_names", $2), $1, $2) '
    'WHERE $1 = ANY("access_model_names") '
    'RETURNING "access_group_id"'
)

_APPEND_MODEL_NAME_SQL: Final = (
    'UPDATE "LiteLLM_AccessGroupTable" '
    'SET "access_model_names" = array_append("access_model_names", $2) '
    'WHERE $1 = ANY("access_model_names") AND NOT ($2 = ANY("access_model_names")) '
    'RETURNING "access_group_id"'
)

_REMOVE_MODEL_NAME_SQL: Final = (
    'UPDATE "LiteLLM_AccessGroupTable" '
    'SET "access_model_names" = array_remove("access_model_names", $1) '
    'WHERE $1 = ANY("access_model_names") '
    'RETURNING "access_group_id"'
)


def _raw_executor(prisma_client: object) -> _RawExecutor:
    db: Final = AccessGroupRepository(prisma_client).prisma_client.db  # pyright: ignore[reportAny]  # untyped Prisma client
    return WriterPinnedClient(db).db  # pyright: ignore[reportAny, reportReturnType]  # untyped Prisma client behind the pin


def _config_sourced_sibling(llm_router: Router, deployment_id: str, model_id: str) -> bool:
    if deployment_id == model_id:
        return False
    deployment: Final = llm_router.get_deployment(model_id=deployment_id)
    return deployment is not None and not deployment.model_info.db_model


def _served_by_a_config_deployment(llm_router: Router | None, model_name: str, model_id: str) -> bool:
    if llm_router is None:
        return False
    return any(
        _config_sourced_sibling(llm_router, deployment_id, model_id)
        for deployment_id in llm_router.get_model_ids(model_name=model_name)
    )


async def _still_backed(executor: _RawExecutor, llm_router: Router | None, model_name: str, model_id: str) -> bool:
    if _served_by_a_config_deployment(llm_router, model_name, model_id):
        return True
    count_rows: Final = await executor.query_raw(_BACKING_DEPLOYMENTS_SQL, model_name)
    return any(_DeploymentCountRow.model_validate(row).deployment_count > 0 for row in count_rows)


async def _rewrite_groups(executor: _RawExecutor, sql: str, *names: str) -> None:
    touched_rows: Final = await executor.query_raw(sql, *names)
    await invalidate_access_group_caches(
        tuple(_TouchedGroupRow.model_validate(row).access_group_id for row in touched_rows)
    )


async def sync_access_groups_for_renamed_model(
    prisma_client: object,
    *,
    model_id: str,
    old_name: str,
    new_name: str,
    llm_router: Router | None,
) -> None:
    if old_name == new_name:
        return
    executor: Final = _raw_executor(prisma_client)
    old_name_still_backed: Final = await _still_backed(executor, llm_router, old_name, model_id)
    await _rewrite_groups(
        executor, _APPEND_MODEL_NAME_SQL if old_name_still_backed else _REPLACE_MODEL_NAME_SQL, old_name, new_name
    )


async def sync_access_groups_for_deleted_model(
    prisma_client: object,
    *,
    model_id: str,
    model_name: str,
    llm_router: Router | None,
) -> None:
    executor: Final = _raw_executor(prisma_client)
    if await _still_backed(executor, llm_router, model_name, model_id):
        return
    await _rewrite_groups(executor, _REMOVE_MODEL_NAME_SQL, model_name)
