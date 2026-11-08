from dataclasses import dataclass, replace
from typing import Final

from litellm._auth.fallback_access import FallbackAuthorizationFacts, GroupedModelScope, fallback_authorization
from litellm._auth.model_access import key_model_grants
from litellm.proxy._types import LiteLLM_ProjectTable, LiteLLM_TeamMembership, LiteLLM_TeamTable, UserAPIKeyAuth
from litellm.proxy.auth.model_access_adapter import object_model_grants, project_model_facts
from litellm.proxy.auth.model_group_adapter import get_authorized_resources_from_key_access_groups
from litellm.repositories.prisma_protocols import DatabaseClient
from litellm.router_api import RegistryReader


@dataclass(frozen=True, slots=True)
class _Sources:
    model: str
    token: UserAPIKeyAuth
    router: RegistryReader | None
    prisma: DatabaseClient | None
    team: LiteLLM_TeamTable | None
    project: LiteLLM_ProjectTable | None


async def load_fallback_authorization_facts(
    model: str,
    valid_token: UserAPIKeyAuth,
    llm_router: RegistryReader | None,
    prisma_client: DatabaseClient | None = None,
    team_membership: LiteLLM_TeamMembership | None = None,
    project_object: LiteLLM_ProjectTable | None = None,
    refuse_agent_keys: bool = False,
    team_object: LiteLLM_TeamTable | None = None,
) -> FallbackAuthorizationFacts:
    facts: Final = FallbackAuthorizationFacts(
        refuse_agent_keys=refuse_agent_keys,
        agent_id=valid_token.agent_id,
        team_present=team_object is not None,
        embedded_team_grants=tuple(valid_token.team_models or ()),
        member_grants=tuple(team_membership.litellm_budget_table.allowed_models or ())
        if team_membership is not None and team_membership.litellm_budget_table is not None
        else (),
        project_grants=tuple(project_object.models) if project_object is not None else (),
    )
    return await _read(facts, _Sources(model, valid_token, llm_router, prisma_client, team_object, project_object))


async def _read(facts: FallbackAuthorizationFacts, source: _Sources) -> FallbackAuthorizationFacts:
    from litellm.proxy.auth.auth_checks import get_models_from_access_groups, key_model_aliases_for_auth_check

    decision: Final = fallback_authorization(facts)
    match decision.kind:
        case "Allow" | "Deny":
            return facts
        case "ReadKey":
            grants: Final = key_model_grants(
                object_model_grants(source.token), source.token.team_id, tuple(source.token.team_models or ())
            )
            key: Final = GroupedModelScope(
                project_model_facts(
                    source.model,
                    source.router,
                    grants,
                    source.token.team_model_aliases,
                    source.token.team_id,
                    key_model_aliases_for_auth_check(source.token),
                    "key",
                ),
                tuple(source.token.access_group_ids or ()),
            )
            return await _read(replace(facts, key=key), source)
        case "ReadTeam":
            if source.team is None:
                raise RuntimeError("missing fallback team source")
            team: Final = GroupedModelScope(
                project_model_facts(
                    source.model,
                    source.router,
                    source.team.models,
                    source.token.team_model_aliases,
                    source.team.team_id,
                    key_model_aliases_for_auth_check(source.token),
                    "team",
                ),
                tuple(source.team.access_group_ids or ()),
            )
            return await _read(replace(facts, team=team), source)
        case "ReadEmbeddedTeam":
            embedded: Final = project_model_facts(
                source.model,
                source.router,
                facts.embedded_team_grants,
                source.token.team_model_aliases,
                source.token.team_id,
                key_model_aliases_for_auth_check(source.token),
                "team",
            )
            return await _read(replace(facts, embedded_team=embedded), source)
        case "ReadMember":
            member: Final = project_model_facts(
                source.model,
                source.router,
                facts.member_grants,
                team_id=source.token.team_id,
                key_model_aliases=key_model_aliases_for_auth_check(source.token),
                object_type="team",
            )
            return await _read(replace(facts, member=member), source)
        case "ReadProject":
            if source.project is None:
                raise RuntimeError("missing fallback project source")
            project: Final = project_model_facts(
                source.model,
                source.router,
                facts.project_grants,
                team_id=source.project.team_id,
                key_model_aliases=key_model_aliases_for_auth_check(source.token),
                object_type="project",
            )
            return await _read(replace(facts, project=project), source)
        case "ReadKeyGroups":
            if facts.key is None:
                raise RuntimeError("missing fallback key scope")
            models: Final = await get_models_from_access_groups(facts.key.group_ids, prisma_client=source.prisma)
            return await _read(replace(facts, key=replace(facts.key, group_models=tuple(models))), source)
        case "ReadTeamGroups":
            if facts.team is None:
                raise RuntimeError("missing fallback team scope")
            team_models: Final = await get_models_from_access_groups(facts.team.group_ids, prisma_client=source.prisma)
            return await _read(replace(facts, team=replace(facts.team, group_models=tuple(team_models))), source)
        case "ReadAssignedGroups":
            assigned: Final = await get_authorized_resources_from_key_access_groups(
                source.token, source.team, "access_model_names"
            )
            return await _read(replace(facts, assigned_models=tuple(assigned)), source)
