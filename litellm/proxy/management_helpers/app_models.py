from typing import Final

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from litellm.proxy._types import ProxyErrorTypes, ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import TeamNotFoundError, get_key_object, get_team_object
from litellm.proxy.db.db_lookup_gate import DBLookupDeadlineExceeded
from litellm.proxy.management_helpers.apps import AppStore
from litellm.rust_bridge._native import NativeDatabaseFailure
from litellm.types.proxy.management_endpoints.apps import AppFailure, AppResourceType


class _ResourceIdentity(BaseModel):
    model_config = ConfigDict(frozen=True)

    token: str | None = Field(default=None, repr=False)
    team_id: str | None = None


_ALIASES: Final = TypeAdapter(dict[str, str])


async def app_resource_models(
    store: AppStore, resource_type: AppResourceType, resource_id: str
) -> tuple[str, ...] | AppFailure:
    from litellm.proxy.proxy_server import accessible_model_names

    try:
        auth: Final = await _app_resource_auth(store, resource_type, resource_id)
        return auth if isinstance(auth, AppFailure) else await accessible_model_names(auth)
    except (HTTPException, ProxyException, NativeDatabaseFailure, DBLookupDeadlineExceeded) as error:
        missing: Final = isinstance(error, TeamNotFoundError) or (
            isinstance(error, ProxyException) and error.type == ProxyErrorTypes.token_not_found_in_db
        )
        if missing:
            return AppFailure(404, "App resource or its parent not found.")
        return AppFailure(503, "Unable to load app models. Try again.")


async def _app_resource_auth(
    store: AppStore, resource_type: AppResourceType, resource_id: str
) -> UserAPIKeyAuth | AppFailure:
    from litellm.proxy.proxy_server import prisma_client, proxy_logging_obj, user_api_key_cache

    table, identity = store.resource_table(resource_type)
    row: Final = await table.find_unique(where={identity: resource_id})
    if row is None:
        return AppFailure(404, "App resource not found.")
    resource: Final = _ResourceIdentity.model_validate(row.model_dump())
    if resource_type == AppResourceType.VIRTUAL_KEY:
        if resource.token is None:
            return AppFailure(404, "App key not found.")
        return await get_key_object(
            hashed_token=resource.token,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )
    team_id: Final = resource_id if resource_type == AppResourceType.TEAM else resource.team_id
    team: Final = (
        await get_team_object(
            team_id=team_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )
        if team_id is not None
        else None
    )
    aliases: Final[object] = (
        team.litellm_model_table.model_dump().get("model_aliases") if team and team.litellm_model_table else None
    )
    return UserAPIKeyAuth(
        team_id=team_id,
        project_id=resource_id if resource_type == AppResourceType.PROJECT else None,
        team_models=team.models if team else [],
        team_model_aliases=(
            _ALIASES.validate_json(aliases) if isinstance(aliases, str) else _ALIASES.validate_python(aliases or {})
        ),
    )
