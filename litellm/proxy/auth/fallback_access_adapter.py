from dataclasses import replace
from typing import Final

from litellm._auth.fallback_access import FallbackAuthorizationFacts, GroupedModelScope, fallback_authorization
from litellm._auth.model_access import key_model_grants
from litellm.proxy._types import LiteLLM_ProjectTable, LiteLLM_TeamMembership, LiteLLM_TeamTable, UserAPIKeyAuth
from litellm.proxy.auth.model_access_adapter import object_model_grants, project_model_facts
from litellm.proxy.auth.model_group_adapter import get_authorized_resources_from_key_access_groups
from litellm.repositories.prisma_protocols import DatabaseClient
from litellm.router_api import RegistryReader


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
    from litellm.proxy.auth.auth_checks import key_model_aliases_for_auth_check

    aliases: Final = key_model_aliases_for_auth_check(valid_token)
    key_grants: Final = key_model_grants(
        object_model_grants(valid_token), valid_token.team_id, tuple(valid_token.team_models or ())
    )
    member_grants: Final = (
        tuple(team_membership.litellm_budget_table.allowed_models or ())
        if team_membership is not None and team_membership.litellm_budget_table is not None
        else ()
    )
    facts: Final = FallbackAuthorizationFacts(
        key=GroupedModelScope(
            project_model_facts(
                model, llm_router, key_grants, valid_token.team_model_aliases, valid_token.team_id, aliases, "key"
            ),
            tuple(valid_token.access_group_ids or ()),
        ),
        refuse_agent_keys=refuse_agent_keys,
        agent_id=valid_token.agent_id,
        team=GroupedModelScope(
            project_model_facts(
                model,
                llm_router,
                team_object.models,
                valid_token.team_model_aliases,
                team_object.team_id,
                aliases,
                "team",
            ),
            tuple(team_object.access_group_ids or ()),
        )
        if team_object is not None
        else None,
        embedded_team=project_model_facts(
            model,
            llm_router,
            valid_token.team_models or (),
            valid_token.team_model_aliases,
            valid_token.team_id,
            aliases,
            "team",
        )
        if team_object is None
        else None,
        member=project_model_facts(
            model, llm_router, member_grants, team_id=valid_token.team_id, key_model_aliases=aliases, object_type="team"
        ),
        project=project_model_facts(
            model,
            llm_router,
            project_object.models,
            team_id=project_object.team_id,
            key_model_aliases=aliases,
            object_type="project",
        )
        if project_object is not None
        else None,
    )
    return await _load_groups(facts, valid_token, team_object, prisma_client)


async def _load_groups(
    facts: FallbackAuthorizationFacts,
    token: UserAPIKeyAuth,
    team: LiteLLM_TeamTable | None,
    prisma: DatabaseClient | None,
) -> FallbackAuthorizationFacts:
    from litellm.proxy.auth.auth_checks import get_models_from_access_groups

    decision: Final = fallback_authorization(facts)
    match decision.kind:
        case "Allow" | "Deny":
            return facts
        case "ReadKeyGroups":
            models: Final = await get_models_from_access_groups(facts.key.group_ids, prisma_client=prisma)
            return await _load_groups(
                replace(facts, key=replace(facts.key, group_models=tuple(models))), token, team, prisma
            )
        case "ReadTeamGroups":
            if facts.team is None:
                raise RuntimeError("fallback requested missing team")
            team_models: Final = await get_models_from_access_groups(facts.team.group_ids, prisma_client=prisma)
            return await _load_groups(
                replace(facts, team=replace(facts.team, group_models=tuple(team_models))), token, team, prisma
            )
        case "ReadAssignedGroups":
            assigned: Final = await get_authorized_resources_from_key_access_groups(token, team, "access_model_names")
            return await _load_groups(replace(facts, assigned_models=tuple(assigned)), token, team, prisma)
