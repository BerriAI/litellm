from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Final, Literal

from litellm._auth.failure import NativeAuthRuntimeError
from litellm._auth.fallback_access import GroupedModelScope
from litellm._auth.team_override import TeamOverrideFacts, team_override
from litellm.proxy._types import LiteLLM_TeamTable, UserAPIKeyAuth
from litellm.proxy.auth.model_access_adapter import object_model_grants, project_model_facts, raise_model_denial
from litellm.proxy.auth.model_group_adapter import get_authorized_resources_from_key_access_groups
from litellm.repositories.prisma_protocols import DatabaseClient
from litellm.router_api import RegistryReader


@dataclass(frozen=True, slots=True)
class _Source:
    model: str | Sequence[str]
    token: UserAPIKeyAuth
    team: LiteLLM_TeamTable | None
    router: RegistryReader | None
    team_aliases: Mapping[str, str] | None
    key_aliases: Mapping[str, str] | None
    prisma: DatabaseClient | None


async def read_team_override_facts(
    model: str | Sequence[str],
    valid_token: UserAPIKeyAuth,
    team_object: LiteLLM_TeamTable | None,
    llm_router: RegistryReader | None,
    team_model_aliases: Mapping[str, str] | None = None,
    key_model_aliases: Mapping[str, str] | None = None,
    prisma_client: DatabaseClient | None = None,
) -> TeamOverrideFacts:
    return await _read(
        TeamOverrideFacts(),
        _Source(model, valid_token, team_object, llm_router, team_model_aliases, key_model_aliases, prisma_client),
    )


async def can_team_call_model_with_override(
    model: str | Sequence[str],
    valid_token: UserAPIKeyAuth,
    team_object: LiteLLM_TeamTable | None,
    llm_router: RegistryReader | None,
    team_model_aliases: Mapping[str, str] | None = None,
    key_model_aliases: Mapping[str, str] | None = None,
    prisma_client: DatabaseClient | None = None,
) -> Literal[True]:
    facts: Final = await read_team_override_facts(
        model, valid_token, team_object, llm_router, team_model_aliases, key_model_aliases, prisma_client
    )
    decision: Final = team_override(facts)
    if decision.kind == "Allow":
        return True
    if decision.kind != "Deny" or decision.denial is None:
        raise NativeAuthRuntimeError("incomplete team override authorization")
    return raise_model_denial(decision.denial, model)


async def _read(facts: TeamOverrideFacts, source: _Source) -> TeamOverrideFacts:
    from litellm.proxy.auth.auth_checks import get_models_from_access_groups

    decision: Final = team_override(facts)
    match decision.kind:
        case "Allow" | "Deny":
            return facts
        case "ReadTeam":
            scope: Final = project_model_facts(
                source.model,
                source.router,
                object_model_grants(source.team) if source.team is not None else (),
                source.team_aliases,
                source.team.team_id if source.team is not None else None,
                source.key_aliases,
                "team",
            )
            groups: Final = tuple(source.team.access_group_ids or ()) if source.team is not None else ()
            return await _read(replace(facts, team=GroupedModelScope(scope, groups)), source)
        case "ReadTeamGroups":
            if facts.team is None:
                raise NativeAuthRuntimeError("missing team override scope")
            models: Final = await get_models_from_access_groups(facts.team.group_ids, prisma_client=source.prisma)
            return await _read(replace(facts, team=replace(facts.team, group_models=tuple(models))), source)
        case "ReadAssignedGroups":
            assigned: Final = await get_authorized_resources_from_key_access_groups(
                source.token, source.team, "access_model_names"
            )
            return await _read(replace(facts, assigned_models=tuple(assigned)), source)
        case "ReadAssignedScope":
            if facts.assigned_models is None:
                raise NativeAuthRuntimeError("missing assigned model override grants")
            assigned_scope: Final = project_model_facts(
                source.model,
                source.router,
                facts.assigned_models,
                source.token.team_model_aliases,
                source.token.team_id,
                source.key_aliases,
                "key",
            )
            return await _read(replace(facts, assigned_scope=assigned_scope), source)
