from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Final, Literal

from fastapi import HTTPException
from pydantic import BaseModel

import litellm
from litellm._auth.model_access import (
    ModelAccessDecision,
    ModelAccessFacts,
    ModelAliases,
    ModelObject,
    ModelProvider,
    ModelRecord,
    ModelRegistry,
    grouped_model_access,
    key_model_grants,
    model_access,
    model_candidates,
    user_model_access,
)
from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH
from litellm.proxy._types import (
    LiteLLM_EndUserTable,
    LiteLLM_OrganizationTable,
    LiteLLM_ProjectTable,
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
    ModelAccessDeniedProxyException,
    ProxyErrorTypes,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.model_access_denied import (
    customer_model_access_denied_client_message,
    model_access_denied_client_message,
)
from litellm.repositories.prisma_protocols import DatabaseClient
from litellm.router_api import RegistryReader
from litellm.router_strategy.complexity_router.context_compaction import native_compaction_parent


class ModelLists(BaseModel, frozen=True):
    models: tuple[str, ...] = ()


def object_model_grants(source: object) -> tuple[str, ...]:
    return ModelLists.model_validate(source, from_attributes=True).models


def _provider(model: str) -> ModelProvider | None:
    try:
        stripped, provider, _, _ = litellm.get_llm_provider(model=model)
        return ModelProvider(stripped, provider)
    except Exception:  # noqa: BLE001  # provider inference failure leaves the raw provider fact absent
        return None


def _record(model: str, router: RegistryReader | None, team_id: str | None) -> ModelRecord:
    deployment: Final = router.registry().deployments(model, source="first_in_group") if router is not None else None
    return ModelRecord(
        access_groups=tuple(router.registry().access_groups(model_name=model, team_id=team_id))
        if router is not None
        else (),
        provider=_provider(model),
        deployment_team=deployment.model_info.team_id if deployment is not None else None,
        deployment_public_name=deployment.model_info.team_public_model_name if deployment is not None else None,
    )


def project_model_facts(
    model: str | Sequence[str],
    llm_router: RegistryReader | None,
    models: Sequence[str],
    team_model_aliases: Mapping[str, str] | None = None,
    team_id: str | None = None,
    key_model_aliases: Mapping[str, str] | None = None,
    object_type: ModelObject = "user",
    fallback_depth: int = 0,
) -> ModelAccessFacts:
    requested: Final = (model,) if isinstance(model, str) else tuple(model)
    aliases: Final = ModelAliases(
        team=dict(team_model_aliases or {}), key=dict(key_model_aliases or {}), global_=dict(litellm.model_alias_map)
    )
    names: Final = tuple(dict.fromkeys((*requested, *aliases.team.values())))
    registry: Final = ModelRegistry(
        present=llm_router is not None,
        names=tuple(llm_router.registry().names()) if llm_router is not None else (),
        aliases={
            name: target
            for name in names
            if name in llm_router.registry().aliases()
            and (target := llm_router.registry().resolve(name, kind="alias")) is not None
        }
        if llm_router is not None
        else {},
        records={name: _record(name, llm_router, team_id) for name in names},
        compaction_parents={name: parent for name in names if (parent := native_compaction_parent(name)) is not None},
    )
    initial: Final = ModelAccessFacts(
        requested=requested,
        grants=tuple(models),
        team_id=team_id,
        aliases=aliases,
        registry=registry,
        object_type=object_type,
        fallback_depth=fallback_depth,
        max_depth=DEFAULT_MAX_RECURSE_DEPTH,
        is_list=not isinstance(model, str),
    )
    candidates: Final = model_candidates(replace(initial, requested=names))
    return replace(
        initial,
        registry=replace(
            registry,
            records={
                **registry.records,
                **{name: _record(name, llm_router, team_id) for name in candidates if name not in registry.records},
            },
        ),
    )


def raise_model_denial(
    decision: ModelAccessDecision,
    requested: str | Sequence[str],
    *,
    user_id: str | None = None,
    team_id: str | None = None,
    teams: Sequence[str] | None = None,
    member_models: Sequence[str] | None = None,
) -> Literal[True]:
    model: Final = requested if isinstance(requested, str) else list(requested)
    match decision.kind:
        case "Allow":
            return True
        case "Depth":
            raise Exception(  # noqa: TRY002  # preserve the legacy depth failure type and envelope
                f"Unable to parse model, max fallback depth exceeded - received model: {decision.model if decision.model is not None else model}"
            )
        case "NoAgentModels":
            raise HTTPException(403, "This agent has no model grants")
        case "ReadGroups":
            raise RuntimeError("model grant lookup was not answered")
        case "Deny" | "NoDefaultModels" | "EmptyAgentCeiling" | "PersonalTeam":
            error_model: Final = decision.model if decision.model is not None else model
            object_type: Final = decision.object_type
            message: Final = (
                customer_model_access_denied_client_message(error_model)
                if object_type == "customer"
                else model_access_denied_client_message(error_model)
            )
            internal: Final = (
                f"Team member not allowed to access model. User={user_id}, Team={team_id}, Model={model}. Allowed member models = {list(member_models)}"
                if member_models is not None
                else (
                    f"User not allowed to access model. No default model access, only team models allowed. Tried to access {model}"
                    if decision.kind == "NoDefaultModels"
                    else (
                        f"agent {user_id} access groups grant no models"
                        if decision.kind == "EmptyAgentCeiling"
                        else (
                            f"Personal key model access is derived from the owner's teams. User={user_id}, teams={list(teams) if teams is not None else None}, model={error_model}"
                            if decision.kind == "PersonalTeam"
                            else (
                                f"{object_type} not allowed to access model. This {object_type} can only access models={list(decision.grants)}. Tried to access {error_model}"
                            )
                        )
                    )
                )
            )
            error_type: Final = (
                ProxyErrorTypes.team_model_access_denied
                if member_models is not None
                else (
                    ProxyErrorTypes.key_model_access_denied
                    if decision.kind in ("NoDefaultModels", "PersonalTeam")
                    else (
                        ProxyErrorTypes.agent_model_access_denied
                        if decision.kind == "EmptyAgentCeiling"
                        else ProxyErrorTypes.get_model_access_error_type_for_object(object_type)
                    )
                )
            )
            raise ModelAccessDeniedProxyException(
                message=message, internal_message=internal, type=error_type, param="model", code=403
            )


def can_object_call_model(
    model: str | Sequence[str],
    llm_router: RegistryReader | None,
    models: Sequence[str],
    team_model_aliases: Mapping[str, str] | None = None,
    team_id: str | None = None,
    key_model_aliases: Mapping[str, str] | None = None,
    object_type: ModelObject = "user",
    fallback_depth: int = 0,
) -> Literal[True]:
    facts: Final = project_model_facts(
        model, llm_router, models, team_model_aliases, team_id, key_model_aliases, object_type, fallback_depth
    )
    return raise_model_denial(model_access(facts), model)


async def _grouped_model_access(
    facts: ModelAccessFacts,
    group_ids: Sequence[str],
    prisma_client: DatabaseClient | None,
    *,
    merge: bool,
) -> ModelAccessDecision:
    from litellm.proxy.auth.auth_checks import get_models_from_access_groups

    decision: Final = grouped_model_access(facts, tuple(group_ids), merge=merge)
    if decision.kind != "ReadGroups":
        return decision
    models: Final = await get_models_from_access_groups(list(group_ids), prisma_client=prisma_client)
    return grouped_model_access(facts, tuple(group_ids), tuple(models), merge=merge)


async def can_key_call_model(
    model: str | Sequence[str],
    llm_model_list: Sequence[object] | None,
    valid_token: UserAPIKeyAuth,
    llm_router: RegistryReader | None,
    prisma_client: DatabaseClient | None = None,
) -> Literal[True]:
    from litellm.proxy.auth.auth_checks import key_model_aliases_for_auth_check

    grants: Final = key_model_grants(
        object_model_grants(valid_token), valid_token.team_id, tuple(valid_token.team_models or ())
    )
    facts: Final = project_model_facts(
        model,
        llm_router,
        grants,
        valid_token.team_model_aliases,
        valid_token.team_id,
        key_model_aliases_for_auth_check(valid_token),
        "key",
    )
    decision: Final = await _grouped_model_access(facts, valid_token.access_group_ids or (), prisma_client, merge=False)
    return raise_model_denial(decision, model)


async def can_team_access_model(
    model: str | Sequence[str],
    team_object: LiteLLM_TeamTable | None,
    llm_router: RegistryReader | None,
    team_model_aliases: Mapping[str, str] | None = None,
    key_model_aliases: Mapping[str, str] | None = None,
    prisma_client: DatabaseClient | None = None,
) -> Literal[True]:
    facts: Final = project_model_facts(
        model,
        llm_router,
        team_object.models if team_object is not None else (),
        team_model_aliases,
        team_object.team_id if team_object is not None else None,
        key_model_aliases,
        "team",
    )
    decision: Final = await _grouped_model_access(
        facts, (team_object.access_group_ids or ()) if team_object is not None else (), prisma_client, merge=True
    )
    return raise_model_denial(decision, model)


def can_org_access_model(
    model: str,
    org_object: LiteLLM_OrganizationTable | None,
    llm_router: RegistryReader | None,
    team_model_aliases: Mapping[str, str] | None = None,
) -> Literal[True]:
    return can_object_call_model(
        model, llm_router, org_object.models if org_object is not None else (), team_model_aliases, object_type="org"
    )


def can_project_access_model(
    model: str | Sequence[str],
    project_object: LiteLLM_ProjectTable,
    llm_router: RegistryReader | None,
    key_model_aliases: Mapping[str, str] | None = None,
) -> Literal[True]:
    return can_object_call_model(
        model,
        llm_router,
        project_object.models,
        team_id=project_object.team_id,
        key_model_aliases=key_model_aliases,
        object_type="project",
    )


def can_customer_access_model(
    model: str | Sequence[str],
    end_user_object: LiteLLM_EndUserTable,
    llm_router: RegistryReader | None,
    valid_token: UserAPIKeyAuth | None,
) -> Literal[True]:
    from litellm.proxy.auth.auth_checks import key_model_aliases_for_auth_check, team_model_aliases_for_auth_check

    facts: Final = project_model_facts(
        model,
        llm_router,
        end_user_object.models or (),
        team_model_aliases_for_auth_check(valid_token) if valid_token is not None else None,
        valid_token.team_id if valid_token is not None else None,
        key_model_aliases_for_auth_check(valid_token),
        "customer",
    )
    return raise_model_denial(model_access(facts, kind="Customer"), model)


async def can_user_call_model(
    model: str | Sequence[str],
    llm_router: RegistryReader | None,
    user_object: LiteLLM_UserTable | None,
    key_model_aliases: Mapping[str, str] | None = None,
) -> Literal[True]:
    facts: Final = project_model_facts(
        model,
        llm_router,
        object_model_grants(user_object) if user_object is not None else (),
        key_model_aliases=key_model_aliases,
    )
    return raise_model_denial(user_model_access(facts, present=user_object is not None), model)
