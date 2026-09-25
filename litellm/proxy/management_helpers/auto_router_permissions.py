from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from typing_extensions import ReadOnly, TypedDict

from litellm.models.organization import LiteLLM_OrganizationTable
from litellm.models.project import LiteLLM_ProjectTable
from litellm.proxy._types import (
    UI_TEAM_ID,
    CommonProxyErrors,
    KeyManagementRoutes,
    LiteLLM_TeamMembership,
    LiteLLM_TeamTable,
    LitellmUserRoles,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_checks import (
    _check_team_member_model_access,  # pyright: ignore[reportPrivateUsage]  # shared membership authorization owner
    can_key_call_model,
    can_org_access_model,
    can_project_access_model,
    can_team_access_model,
)
from litellm.proxy.auth.team_grants import team_model_aliases
from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper
from litellm.repositories.organization_repository import OrganizationRepository
from litellm.repositories.prisma_protocols import DatabaseClient
from litellm.repositories.project_repository import ProjectRepository
from litellm.repositories.table_repositories import TeamMembershipRepository
from litellm.router import Router
from litellm.router_utils.auto_router_model_naming import classify_strategy_router_model, strategy_router_dependencies
from litellm.types.management_endpoints.auto_router_endpoints import RequestComplexityRouterConfig
from litellm.types.router import Deployment, updateDeployment

if TYPE_CHECKING:
    from prisma import types as prisma_types


class _MemberRouterThinking(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["enabled", "disabled", "adaptive"]
    budget_tokens: int | None = Field(default=None, gt=0, le=1_000_000)


class _MemberRouterGenerationParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reasoning_effort: str | None = None
    thinking: _MemberRouterThinking | None = None
    verbosity: Literal["low", "medium", "high"] | None = None
    max_tokens: int | None = Field(default=None, gt=0, le=1_000_000)
    max_completion_tokens: int | None = Field(default=None, gt=0, le=1_000_000)
    max_output_tokens: int | None = Field(default=None, gt=0, le=1_000_000)
    temperature: float | None = Field(default=None, ge=0, le=2, allow_inf_nan=False)
    top_p: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    frequency_penalty: float | None = Field(default=None, ge=-2, le=2, allow_inf_nan=False)
    presence_penalty: float | None = Field(default=None, ge=-2, le=2, allow_inf_nan=False)
    seed: int | None = None
    stop: str | tuple[str, ...] | None = None


class _MemberJevClassifierConfig(BaseModel):
    """The Jev classifier settings a team member may set. Credentials stay the proxy's own: a member-chosen
    api_base would receive the proxy's TYPESAFE_API_KEY, and a member-chosen api_key would be sent from the proxy."""

    model_config = ConfigDict(extra="forbid")

    provider: Literal["typesafe", "laya"] = "typesafe"
    model: str
    api_key: None = None
    api_base: None = None
    timeout_ms: int
    instructions: str | None = None
    circuit_breaker_enabled: bool
    circuit_breaker_cooldown_seconds: float


class _MemberComplexityRouterConfig(RequestComplexityRouterConfig):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class _RouterConfigSource(BaseModel):
    model: str | None = None
    complexity_router_config: Mapping[str, object] | None = None


class _MembershipKey(TypedDict):
    user_id: ReadOnly[str]
    team_id: ReadOnly[str]


class _MembershipWhere(TypedDict):
    user_id_team_id: ReadOnly[_MembershipKey]


@dataclass(frozen=True, slots=True)
class MemberAutoRouterDependencyObjects:
    membership: LiteLLM_TeamMembership | None
    organization: LiteLLM_OrganizationTable | None
    project: LiteLLM_ProjectTable | None


def authorize_member_auto_router_team(
    *, user_api_key_dict: UserAPIKeyAuth, team: LiteLLM_TeamTable, premium_user: bool
) -> None:
    if not premium_user:
        raise HTTPException(status_code=403, detail=CommonProxyErrors.not_premium_user.value)
    if (
        user_api_key_dict.user_role
        not in (LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.TEAM, LitellmUserRoles.ORG_ADMIN)
        or not user_api_key_dict.user_id
        or not any(member.user_id == user_api_key_dict.user_id for member in team.members_with_roles)
        or user_api_key_dict.team_id not in (None, UI_TEAM_ID, team.team_id)
        or team.blocked
        or KeyManagementRoutes.AUTO_ROUTER_MANAGE.value not in (team.team_member_permissions or ())
    ):
        raise HTTPException(status_code=403, detail="This team does not allow you to manage your own auto routers.")


def validate_member_auto_router_config(config: Mapping[str, object]) -> RequestComplexityRouterConfig:
    try:
        validated: Final = _MemberComplexityRouterConfig.model_validate(config)
        for entries in validated.tier_model_configs.values():
            for entry in entries:
                _MemberRouterGenerationParams.model_validate(entry.litellm_params)
        if validated.jev_classifier_config is not None:
            _MemberJevClassifierConfig.model_validate(validated.jev_classifier_config.model_dump())
        return validated
    except ValidationError as exc:
        location: Final = ".".join(str(part) for part in exc.errors()[0]["loc"])
        raise HTTPException(status_code=400, detail=f"Invalid member auto-router configuration at {location}.") from exc


async def authorize_member_auto_router_dependencies(
    *,
    config: RequestComplexityRouterConfig,
    default_model: str | None,
    user_api_key_dict: UserAPIKeyAuth,
    team: LiteLLM_TeamTable,
    prisma_client: DatabaseClient | None,
    llm_router: Router,
    dependency_objects: MemberAutoRouterDependencyObjects | None = None,
) -> None:
    from litellm.proxy.proxy_server import proxy_logging_obj, user_api_key_cache

    if team.blocked:
        raise HTTPException(status_code=403, detail="This auto router's team is blocked.")
    aliases: Final = team_model_aliases(team)
    alias_dict: Final = (
        dict(aliases) if aliases is not None else None  # mutable-ok: auth model and helpers require dict
    )
    scoped_actor: Final = user_api_key_dict.model_copy(
        update=MappingProxyType({"team_id": team.team_id, "team_models": team.models, "team_model_aliases": alias_dict})
    )
    objects: Final = (
        dependency_objects
        if dependency_objects is not None
        else await _load_member_auto_router_dependency_objects(
            user_api_key_dict=scoped_actor, team=team, prisma_client=prisma_client
        )
    )
    if team.organization_id and objects.organization is None:
        raise HTTPException(status_code=403, detail="The auto router's organization is unavailable.")
    if scoped_actor.project_id and (
        objects.project is None or objects.project.team_id != team.team_id or objects.project.blocked
    ):
        raise HTTPException(status_code=403, detail="The auto router's project is unavailable.")
    dependencies: Final = strategy_router_dependencies(
        MappingProxyType(
            {
                "model": "auto_router/complexity_router",
                "complexity_router_config": config.model_dump(exclude_none=True),
                "complexity_router_default_model": default_model,
            }
        )
    )
    for dependency, model, deployments in (
        (
            dependency,
            dependency.model_name,
            llm_router.get_model_list(model_name=dependency.model_name, team_id=team.team_id),
        )
        for dependency in dependencies
    ):
        if dependency.role != "evaluation" and (
            not deployments
            or any(
                classify_strategy_router_model(
                    _RouterConfigSource.model_validate(deployment["litellm_params"]).model or ""
                )
                is not None
                for deployment in deployments
            )
        ):
            raise HTTPException(status_code=400, detail=f"Auto-router target {model!r} must be a configured model.")
        await can_team_access_model(
            model=model,
            team_object=team,
            llm_router=llm_router,
            team_model_aliases=alias_dict,
            prisma_client=prisma_client,
        )
        await can_key_call_model(
            model=model,
            llm_model_list=None,
            valid_token=scoped_actor,
            llm_router=llm_router,
            prisma_client=prisma_client,
        )
        await _check_team_member_model_access(
            model=model,
            team_object=team,
            valid_token=scoped_actor,
            llm_router=llm_router,
            prisma_client=None,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
            team_membership=objects.membership,
            team_membership_loaded=True,
        )
        if objects.organization is not None:
            can_org_access_model(model=model, org_object=objects.organization, llm_router=llm_router)
        if objects.project is not None:
            can_project_access_model(model=model, project_object=objects.project, llm_router=llm_router)


async def _load_member_auto_router_dependency_objects(
    *, user_api_key_dict: UserAPIKeyAuth, team: LiteLLM_TeamTable, prisma_client: DatabaseClient | None
) -> MemberAutoRouterDependencyObjects:
    if prisma_client is None:
        raise HTTPException(status_code=503, detail="Cannot verify auto-router model access without a database")
    membership_where: Final[_MembershipWhere] = {
        "user_id_team_id": {"user_id": user_api_key_dict.user_id or "", "team_id": team.team_id}
    }
    membership_include: Final[prisma_types.LiteLLM_TeamMembershipInclude] = {"litellm_budget_table": True}
    membership_row: Final = (
        await TeamMembershipRepository(prisma_client).table.find_unique(
            where=membership_where, include=membership_include
        )
        if user_api_key_dict.user_id
        else None
    )
    membership: Final = (
        LiteLLM_TeamMembership.model_validate(membership_row.model_dump()) if membership_row is not None else None
    )
    organization: Final = (
        await OrganizationRepository(prisma_client).find_by_id(team.organization_id) if team.organization_id else None
    )
    if team.organization_id and organization is None:
        raise HTTPException(status_code=403, detail="The auto router's organization is unavailable.")
    project: Final = (
        await ProjectRepository(prisma_client).find_by_id(user_api_key_dict.project_id)
        if user_api_key_dict.project_id
        else None
    )
    return MemberAutoRouterDependencyObjects(membership=membership, organization=organization, project=project)


class StoredAutoRouterIdentity(BaseModel):
    created_by: str | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class MemberAutoRouterWrite:
    actor: UserAPIKeyAuth
    team_id: str
    model_id: str | None
    public_name: str
    updated_at: datetime | None
    config: RequestComplexityRouterConfig
    default_model: str | None


async def authorize_member_auto_router_write(
    *,
    incoming: Deployment | updateDeployment,
    existing: Deployment | None,
    user_api_key_dict: UserAPIKeyAuth,
    team: LiteLLM_TeamTable,
    premium_user: bool,
    prisma_client: DatabaseClient,
    llm_router: Router,
) -> MemberAutoRouterWrite:
    authorize_member_auto_router_team(user_api_key_dict=user_api_key_dict, team=team, premium_user=premium_user)
    stored: Final = StoredAutoRouterIdentity.model_validate(existing.model_dump()) if existing is not None else None
    if stored is not None and stored.created_by != user_api_key_dict.user_id:
        raise HTTPException(status_code=403, detail="Team members can update only their own auto routers.")
    params: Final = incoming.litellm_params
    if params is None or incoming.model_fields_set - frozenset({"model_name", "litellm_params", "model_info"}):
        raise HTTPException(status_code=403, detail="Team members may change only auto-router configuration.")
    if params.model_fields_set - frozenset({"model", "complexity_router_config", "complexity_router_default_model"}):
        raise HTTPException(status_code=403, detail="Team members may change only auto-router configuration.")
    info: Final = incoming.model_info
    if info is not None and (
        info.model_fields_set - frozenset({"id", "team_id"})
        or info.team_id not in (None, team.team_id)
        or (existing is not None and "id" in info.model_fields_set and info.id != existing.model_info.id)
    ):
        raise HTTPException(
            status_code=403, detail="Team members cannot change model ownership or administrative settings."
        )
    existing_model: Final = (
        decrypt_value_helper(existing.litellm_params.model, key="model", return_original_value=True)
        if existing is not None
        else None
    )
    effective_model: Final = params.model or existing_model
    if (
        not isinstance(effective_model, str)
        or classify_strategy_router_model(effective_model) != "complexity"
        or (existing is not None and effective_model != existing_model)
    ):
        raise HTTPException(status_code=403, detail="Team members may manage only complexity auto routers.")
    public_name: Final = (
        existing.model_info.team_public_model_name or existing.model_name
        if existing is not None
        else incoming.model_name
    )
    if (
        not public_name
        or public_name != public_name.strip()
        or any(character in public_name for character in "*?[]")
        or public_name.startswith("model_name_")
    ):
        raise HTTPException(
            status_code=400, detail="Choose a non-empty auto-router name without wildcards or internal prefixes."
        )
    if existing is not None and incoming.model_name not in (None, public_name, existing.model_name):
        raise HTTPException(status_code=403, detail="Team members cannot rename an auto router.")
    supplied_config: Final = _RouterConfigSource.model_validate(params.model_dump()).complexity_router_config
    raw_config: Final = (
        supplied_config
        if supplied_config is not None
        else _RouterConfigSource.model_validate(existing.litellm_params.model_dump()).complexity_router_config
        if existing is not None
        else None
    )
    if raw_config is None:
        raise HTTPException(status_code=400, detail="A complexity_router_config is required.")
    config: Final = validate_member_auto_router_config(raw_config)
    stored_default: Final = existing.litellm_params.complexity_router_default_model if existing is not None else None
    default_model: Final = (
        params.complexity_router_default_model
        if params.complexity_router_default_model is not None
        else decrypt_value_helper(stored_default, key="complexity_router_default_model", return_original_value=True)
        if stored_default is not None
        else None
    )
    await authorize_member_auto_router_dependencies(
        config=config,
        default_model=default_model,
        user_api_key_dict=user_api_key_dict,
        team=team,
        prisma_client=prisma_client,
        llm_router=llm_router,
    )
    return MemberAutoRouterWrite(
        actor=user_api_key_dict,
        team_id=team.team_id,
        model_id=existing.model_info.id if existing is not None else None,
        public_name=public_name,
        updated_at=stored.updated_at if stored is not None else None,
        config=config,
        default_model=default_model,
    )
