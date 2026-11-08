from dataclasses import replace
from typing import Final

from litellm._auth.fallback_access import FallbackAuthorizationDecision as Decision
from litellm._auth.fallback_access import FallbackAuthorizationFacts as Facts
from litellm._auth.fallback_access import GroupedModelScope
from litellm._auth.model_access import ModelAccessDecision
from tests.unit._auth.support.model_reference import model_access


def grouped(scope: GroupedModelScope, *, merge: bool = False) -> ModelAccessDecision:
    primary: Final = model_access(scope.facts)
    if primary.kind != "Deny" or not scope.group_ids:
        return primary
    if scope.group_models is None:
        return ModelAccessDecision("ReadGroups")
    if not scope.group_models:
        return primary
    grants: Final = tuple(dict.fromkeys((*scope.facts.grants, *scope.group_models))) if merge else scope.group_models
    return model_access(replace(scope.facts, grants=grants))


def authorization(facts: Facts) -> Decision:
    if facts.refuse_agent_keys and facts.agent_id is not None:
        return Decision("Deny")
    if facts.key is None:
        return Decision("ReadKey")
    key: Final = grouped(facts.key)
    if key.kind == "ReadGroups":
        return Decision("ReadKeyGroups")
    if key.kind != "Allow":
        return Decision("Deny", key)
    if facts.team_present and facts.team is None:
        return Decision("ReadTeam")
    if facts.team is not None:
        team: Final = grouped(facts.team, merge=True)
        if team.kind == "ReadGroups":
            return Decision("ReadTeamGroups")
        if team.kind != "Allow":
            if team.kind != "Deny":
                return Decision("Deny", team)
            if facts.assigned_models is None:
                return Decision("ReadAssignedGroups")
            if (
                not facts.assigned_models
                or model_access(replace(facts.team.facts, grants=facts.assigned_models)).kind != "Allow"
            ):
                return Decision("Deny", team)
    elif facts.embedded_team is None and facts.embedded_team_grants:
        return Decision("ReadEmbeddedTeam")
    elif facts.embedded_team is not None:
        embedded: Final = model_access(facts.embedded_team)
        if embedded.kind != "Allow":
            return Decision("Deny", embedded)
    if facts.member is None and facts.member_grants:
        return Decision("ReadMember")
    if facts.member is not None:
        member: Final = model_access(facts.member)
        if member.kind != "Allow":
            return Decision("Deny", member)
    if facts.project is None and facts.project_grants:
        return Decision("ReadProject")
    if facts.project is not None:
        project: Final = model_access(facts.project)
        if project.kind != "Allow":
            return Decision("Deny", project)
    return Decision("Allow")
