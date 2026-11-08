from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Final

from pydantic import BaseModel, Field, TypeAdapter

import litellm
from litellm._auth.failure import NativeAuthFailure, NativeAuthRuntimeError
from litellm._auth.fallback_access import GroupedModelScope
from litellm._auth.model_access import key_model_grants
from litellm._auth.resolved_models import ResolvedModelFacts, ResolvedTeamRead, resolved_models
from litellm._logging import verbose_proxy_logger
from litellm.constants import LITELLM_PROXY_MASTER_KEY_ALIAS
from litellm.proxy._types import (
    LiteLLM_EndUserTable,
    LiteLLM_ProjectTable,
    LiteLLM_TeamTableCachedObj,
    LiteLLM_UserTable,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.member_model_adapter import read_member_model_facts
from litellm.proxy.auth.model_access_adapter import object_model_grants, project_model_facts, raise_model_denial
from litellm.proxy.auth.personal_model_request_adapter import read_personal_model_request_facts
from litellm.proxy.auth.team_override_adapter import read_team_override_facts
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.utils import PrismaClient, ProxyLogging
from litellm.router_api import RegistryReader


class _Config(BaseModel, frozen=True):
    config: Mapping[str, object] = Field(default_factory=dict[str, object])


_METADATA: Final[TypeAdapter[Mapping[str, object] | None]] = TypeAdapter(Mapping[str, object] | None)


@dataclass(frozen=True, slots=True)
class ResolvedModelReading:
    facts: ResolvedModelFacts
    team: LiteLLM_TeamTableCachedObj | None = None
    owner: LiteLLM_UserTable | None = None


@dataclass(frozen=True, slots=True)
class _Source:
    model: str
    token: UserAPIKeyAuth
    router: RegistryReader | None
    prisma: PrismaClient | None
    cache: UserApiKeyCache
    logging: ProxyLogging
    team: LiteLLM_TeamTableCachedObj | None = None
    owner: LiteLLM_UserTable | None = None
    project: LiteLLM_ProjectTable | None = None
    customer: LiteLLM_EndUserTable | None = None


async def read_resolved_model_facts(
    model: str,
    valid_token: UserAPIKeyAuth,
    llm_router: RegistryReader | None,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
    proxy_logging_obj: ProxyLogging,
) -> ResolvedModelReading:
    facts: Final = ResolvedModelFacts(
        model=model,
        config_nonempty=bool(_Config.model_validate(valid_token, from_attributes=True).config),
        key_models=object_model_grants(valid_token),
        team_id=valid_token.team_id,
        user_id=valid_token.user_id,
        project_present=valid_token.project_id is not None,
        customer_present=valid_token.end_user_id is not None,
        database_present=prisma_client is not None,
        personal_enabled=bool(litellm.personal_key_model_access_from_teams),
        master=valid_token.api_key == LITELLM_PROXY_MASTER_KEY_ALIAS,
        role=valid_token.user_role,
    )
    return await _read(
        facts, _Source(model, valid_token, llm_router, prisma_client, user_api_key_cache, proxy_logging_obj)
    )


async def can_key_call_resolved_model(
    model: str, llm_model_list: Sequence[object] | None, valid_token: UserAPIKeyAuth, llm_router: RegistryReader | None
) -> None:
    from litellm.proxy.proxy_server import prisma_client, proxy_logging_obj, user_api_key_cache

    reading: Final = await read_resolved_model_facts(
        model, valid_token, llm_router, prisma_client, user_api_key_cache, proxy_logging_obj
    )
    decision: Final = resolved_models(reading.facts)
    if decision.kind == "Allow":
        return
    if decision.kind != "Deny" or decision.denial is None:
        raise NativeAuthRuntimeError("incomplete resolved model access facts")
    raise_model_denial(
        decision.denial,
        model,
        user_id=valid_token.user_id,
        team_id=reading.team.team_id if reading.team is not None else None,
        teams=reading.owner.teams if reading.owner is not None else None,
        member_models=decision.member_models,
    )


async def _read(facts: ResolvedModelFacts, source: _Source) -> ResolvedModelReading:
    from litellm.proxy.auth.auth_checks import (
        get_end_user_object,
        get_key_end_user_budget_id,
        get_models_from_access_groups,
        get_project_object,
        get_team_object,
        get_user_object,
        key_model_aliases_for_auth_check,
        team_model_aliases_for_auth_check,
    )

    decision: Final = resolved_models(facts)
    match decision.kind:
        case "Allow" | "Deny":
            return ResolvedModelReading(facts, source.team, source.owner)
        case "ReadKey":
            grants: Final = key_model_grants(
                object_model_grants(source.token), source.token.team_id, tuple(source.token.team_models or ())
            )
            key: Final = project_model_facts(
                source.model,
                source.router,
                grants,
                source.token.team_model_aliases,
                source.token.team_id,
                key_model_aliases_for_auth_check(source.token),
                "key",
            )
            return await _read(
                replace(facts, key=GroupedModelScope(key, tuple(source.token.access_group_ids or ()))), source
            )
        case "ReadKeyGroups":
            if facts.key is None:
                raise NativeAuthRuntimeError("resolved key group read without scope")
            models: Final = await get_models_from_access_groups(facts.key.group_ids)
            return await _read(replace(facts, key=replace(facts.key, group_models=tuple(models))), source)
        case "ReadTeamObject":
            if source.token.team_id is None:
                raise NativeAuthRuntimeError("resolved team read without identity")
            try:
                team: Final = await get_team_object(
                    team_id=source.token.team_id,
                    prisma_client=source.prisma,
                    user_api_key_cache=source.cache,
                    parent_otel_span=source.token.parent_otel_span,
                    proxy_logging_obj=source.logging,
                )
            except NativeAuthFailure:
                raise
            except Exception:  # noqa: BLE001  # legacy resolved-model team lookup falls back to the embedded token row
                return await _read(replace(facts, team_read=ResolvedTeamRead(failed=True)), source)
            return await _read(
                replace(facts, team_read=ResolvedTeamRead(found=team is not None)),  # pyright: ignore[reportUnnecessaryComparison]  # preserve the legacy missing-row result
                replace(source, team=team),
            )
        case "ReadEmbeddedTeam":
            embedded: Final = LiteLLM_TeamTableCachedObj(
                team_id=source.token.team_id,  # pyright: ignore[reportArgumentType]  # native selects this read only for a present team ID
                models=source.token.team_models,
                blocked=source.token.team_blocked,
                team_alias=source.token.team_alias,
                metadata=source.token.team_metadata,  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # legacy typed row validates embedded metadata
                object_permission_id=source.token.team_object_permission_id,
                object_permission=source.token.team_object_permission,
            )
            return await _read(replace(facts, embedded_ready=True), replace(source, team=embedded))
        case "ReadTeamScope":
            team_scope: Final = await read_team_override_facts(
                source.model,
                source.token,
                source.team,
                source.router,
                source.token.team_model_aliases,
                key_model_aliases_for_auth_check(source.token),
            )
            return await _read(replace(facts, team=team_scope), source)
        case "ReadMember":
            if source.team is None:
                raise NativeAuthRuntimeError("resolved member model read without team")
            member: Final = await read_member_model_facts(
                source.model,
                source.team,
                source.token,
                source.router,
                source.prisma,
                source.cache,
                source.logging,
                key_model_aliases=key_model_aliases_for_auth_check(source.token),
            )
            return await _read(replace(facts, member=member), source)
        case "ReadOwner":
            if source.token.user_id is None:
                raise NativeAuthRuntimeError("resolved personal owner read without identity")
            try:
                owner: Final = await get_user_object(
                    user_id=source.token.user_id,
                    prisma_client=source.prisma,
                    user_api_key_cache=source.cache,
                    user_id_upsert=False,
                    parent_otel_span=source.token.parent_otel_span,
                    proxy_logging_obj=source.logging,
                )
            except NativeAuthFailure:
                raise
            except Exception as error:  # noqa: BLE001  # preserve raw owner lookup failure and native fail-closed model selection
                verbose_proxy_logger.warning(
                    "Personal key team model access: owner lookup failed, granting nothing: %s", error
                )
                return await _read(replace(facts, owner_loaded=True, owner_failed=True), source)
            return await _read(
                replace(facts, owner_loaded=True, owner_present=owner is not None), replace(source, owner=owner)
            )
        case "ReadPersonal":
            personal: Final = await read_personal_model_request_facts(
                source.model, source.owner, source.token, source.router, source.prisma, source.cache, source.logging
            )
            return await _read(replace(facts, personal=personal), source)
        case "ReadProjectObject":
            if source.token.project_id is None:
                raise NativeAuthRuntimeError("resolved project read without identity")
            project: Final = await get_project_object(
                project_id=source.token.project_id,
                prisma_client=source.prisma,
                user_api_key_cache=source.cache,
                proxy_logging_obj=source.logging,
            )
            return await _read(
                replace(
                    facts,
                    project_loaded=True,
                    project_grants=object_model_grants(project) if project is not None else (),
                ),
                replace(source, project=project),
            )
        case "ReadProjectScope":
            if source.project is None:
                raise NativeAuthRuntimeError("resolved project scope read without row")
            project_scope: Final = project_model_facts(
                source.model,
                source.router,
                facts.project_grants,
                team_id=source.project.team_id,
                key_model_aliases=key_model_aliases_for_auth_check(source.token),
                object_type="project",
            )
            return await _read(replace(facts, project=project_scope), source)
        case "ReadCustomerObject":
            if source.token.end_user_id is None or source.prisma is None:
                raise NativeAuthRuntimeError("resolved customer read without identity or database")
            customer: Final = await get_end_user_object(
                end_user_id=source.token.end_user_id,
                prisma_client=source.prisma,
                user_api_key_cache=source.cache,
                proxy_logging_obj=source.logging,
                token_end_user_max_budget=source.token.end_user_max_budget,
                key_end_user_budget_id=get_key_end_user_budget_id(_METADATA.validate_python(source.token.metadata)),  # pyright: ignore[reportUnknownMemberType]  # validate the legacy metadata field at the IO boundary
            )
            return await _read(
                replace(
                    facts,
                    customer_loaded=True,
                    customer_grants=tuple(customer.models or ()) if customer is not None else (),
                ),
                replace(source, customer=customer),
            )
        case "ReadCustomerScope":
            customer_scope: Final = project_model_facts(
                source.model,
                source.router,
                facts.customer_grants,
                team_model_aliases_for_auth_check(source.token),
                source.token.team_id,
                key_model_aliases_for_auth_check(source.token),
                "customer",
            )
            return await _read(replace(facts, customer=customer_scope), source)
