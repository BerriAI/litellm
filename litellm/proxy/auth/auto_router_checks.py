from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Final

from pydantic import TypeAdapter, ValidationError

from litellm.litellm_core_utils.core_helpers import get_metadata_variable_name_from_kwargs

if TYPE_CHECKING:
    from litellm.router import Router

_MAPPING_ADAPTER: Final = TypeAdapter(Mapping[str, object])


def _mapping(value: object) -> Mapping[str, object] | None:
    try:
        return _MAPPING_ADAPTER.validate_python(value)
    except ValidationError:
        return None


async def authorize_member_auto_router_inference(
    *,
    deployment: Mapping[str, object] | None,
    request_kwargs: Mapping[str, object],
    llm_router: Router,
) -> None:
    if deployment is None:
        return
    model_info: Final = _mapping(deployment.get("model_info"))
    if model_info is None or model_info.get("member_auto_router") is not True:
        return

    from fastapi import HTTPException

    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.auth.auth_checks import (
        OrganizationNotFoundError,
        TeamNotFoundError,
        get_org_object,
        get_project_object,
        get_team_membership,
        get_team_object,
    )
    from litellm.proxy.management_helpers.auto_router_permissions import (
        MemberAutoRouterDependencyObjects,
        authorize_member_auto_router_dependencies,
        validate_member_auto_router_config,
    )

    metadata: Final = _mapping(request_kwargs.get(get_metadata_variable_name_from_kwargs(request_kwargs)))
    actor: Final = metadata.get("user_api_key_auth") if metadata is not None else None
    team_id: Final = model_info.get("team_id")
    if not isinstance(actor, UserAPIKeyAuth) or not isinstance(team_id, str) or not team_id:
        raise HTTPException(status_code=403, detail="Member auto-routers require authenticated team access")
    if actor.team_id != team_id and actor.user_role != LitellmUserRoles.PROXY_ADMIN:
        raise HTTPException(status_code=403, detail="This auto-router belongs to a different team")

    from litellm.proxy.proxy_server import prisma_client, proxy_logging_obj, user_api_key_cache

    if prisma_client is None:
        raise HTTPException(status_code=503, detail="Cannot verify auto-router model access without a database")
    try:
        team: Final = await get_team_object(
            team_id=team_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=actor.parent_otel_span,
            proxy_logging_obj=proxy_logging_obj,
        )
    except TeamNotFoundError as error:
        raise HTTPException(status_code=403, detail="The auto-router team no longer exists") from error
    if (
        actor.user_role != LitellmUserRoles.PROXY_ADMIN
        and actor.user_id is not None
        and (not actor.user_id or not any(member.user_id == actor.user_id for member in team.members_with_roles))
    ):
        raise HTTPException(status_code=403, detail="You are no longer a member of this auto-router's team")
    if team.blocked:
        raise HTTPException(status_code=403, detail="This auto router's team is blocked.")
    params: Final = _mapping(deployment.get("litellm_params"))
    if params is None:
        raise HTTPException(status_code=403, detail="The member auto-router configuration is invalid")
    raw_config: Final = _mapping(params.get("complexity_router_config"))
    if raw_config is None:
        raise HTTPException(status_code=403, detail="The member auto-router configuration is invalid")
    default_model: Final = params.get("complexity_router_default_model")
    config: Final = validate_member_auto_router_config(raw_config)
    membership: Final = (
        await get_team_membership(
            user_id=actor.user_id,
            team_id=team_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=actor.parent_otel_span,
            proxy_logging_obj=proxy_logging_obj,
        )
        if actor.user_id
        else None
    )
    try:
        organization: Final = (
            await get_org_object(
                org_id=team.organization_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=actor.parent_otel_span,
                proxy_logging_obj=proxy_logging_obj,
            )
            if team.organization_id
            else None
        )
    except OrganizationNotFoundError as error:
        raise HTTPException(status_code=403, detail="The auto router's organization is unavailable.") from error
    project: Final = (
        await get_project_object(
            project_id=actor.project_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
        )
        if actor.project_id
        else None
    )
    await authorize_member_auto_router_dependencies(
        config=config,
        default_model=default_model if isinstance(default_model, str) else None,
        user_api_key_dict=actor,
        team=team,
        prisma_client=None,
        llm_router=llm_router,
        dependency_objects=MemberAutoRouterDependencyObjects(
            membership=membership, organization=organization, project=project
        ),
    )
