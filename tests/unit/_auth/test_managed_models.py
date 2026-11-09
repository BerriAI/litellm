from collections.abc import Callable
from dataclasses import replace
from typing import Final

import pytest

from litellm._auth.fallback_access import GroupedModelScope
from litellm._auth.managed_models import ManagedModelDecision as Decision
from litellm._auth.managed_models import ManagedModelFacts as Facts
from litellm._auth.managed_models import managed_models
from litellm._auth.member_models import MemberModelFacts
from litellm._auth.model_access import ModelAccessDecision, ModelAccessFacts
from tests.unit._auth.support.fallback_reference import grouped
from tests.unit._auth.support.model_reference import model_access
from tests.unit._auth.test_member_models import reference as member_reference


def reference(facts: Facts) -> Decision:
    if facts.purpose == "All" and not facts.managed_loaded:
        return Decision("ReadManaged")
    if not facts.model_present or not facts.token_present:
        return Decision("Allow")
    if facts.purpose == "All" and facts.managed_present:
        if not facts.managed_models_valid or not facts.managed_models_count:
            return Decision("Deny", ModelAccessDecision("NoAgentModels"))
        if facts.managed is None:
            return Decision("ReadManagedScope")
        managed: Final = model_access(facts.managed)
        if managed.kind != "Allow":
            return Decision("Deny", managed)
    if facts.purpose != "Caller":
        if facts.ceilings is None:
            return Decision("ReadCeilings")
        if not facts.aliases_loaded:
            return Decision("ReadAliases")
        for index, ceiling in enumerate(facts.ceilings):
            if not ceiling:
                return Decision("Deny", ModelAccessDecision("EmptyAgentCeiling"))
            if index >= len(facts.scopes):
                return Decision("ReadCeilingScope", index=index)
            if (scope := model_access(facts.scopes[index])).kind != "Allow":
                return Decision("Deny", scope)
        if facts.purpose == "Ceilings":
            return Decision("Allow")
    if not facts.caller_loaded:
        return Decision("ReadCaller")
    if not facts.caller_present:
        return Decision("Allow")
    if not facts.team_loaded:
        return Decision("ReadTeam")
    if facts.team_present:
        if facts.team is None:
            return Decision("ReadTeamScope")
        team: Final = grouped(facts.team, merge=True)
        if team.kind == "ReadGroups":
            return Decision("ReadTeamGroups")
        if team.kind != "Allow":
            return Decision("Deny", team)
        if facts.member is None:
            return Decision("ReadMember")
        member: Final = member_reference(facts.member)
        return Decision("Allow") if member.kind == "Allow" else Decision("Deny", member.denial, member.member_models)
    if not facts.user_loaded:
        return Decision("ReadUser")
    if not facts.user_present or not facts.user_grants:
        return Decision("Allow")
    if "no-default-models" in facts.user_grants:
        return Decision("Deny", ModelAccessDecision("NoDefaultModels"))
    if facts.user is None:
        return Decision("ReadUserScope")
    user: Final = model_access(facts.user)
    return Decision("Allow") if user.kind == "Allow" else Decision("Deny", user)


_ALL: Final = Facts("All", True, True)
_GROUPS: Final = Facts("Ceilings", True, True)
_CALLER: Final = Facts("Caller", True, True, caller_loaded=True, caller_present=True, team_loaded=True)
_ALLOW: Final = ModelAccessFacts(("model",), ("model",), object_type="agent")
_DENY: Final = replace(_ALLOW, grants=("other",))
_TEAM: Final = GroupedModelScope(replace(_ALLOW, object_type="team"))


@pytest.mark.parametrize("policy", (reference, managed_models))
@pytest.mark.parametrize(
    "facts",
    (
        _ALL,
        replace(_ALL, model_present=False),
        replace(_ALL, managed_loaded=True, model_present=False),
        replace(_ALL, managed_loaded=True, managed_present=True),
        replace(_ALL, managed_loaded=True, managed_present=True, managed_models_valid=True, managed_models_count=1),
        replace(
            _ALL,
            managed_loaded=True,
            managed_present=True,
            managed_models_valid=True,
            managed_models_count=1,
            managed=_DENY,
        ),
        _GROUPS,
        replace(_GROUPS, ceilings=()),
        replace(_GROUPS, ceilings=(), aliases_loaded=True),
        replace(_GROUPS, ceilings=((),), aliases_loaded=True),
        replace(_GROUPS, ceilings=(("model",), ("other",)), aliases_loaded=True, scopes=(_ALLOW,)),
        replace(_GROUPS, ceilings=(("model",), ("other",)), aliases_loaded=True, scopes=(_ALLOW, _DENY)),
        _CALLER,
        replace(_CALLER, team_present=True),
        replace(_CALLER, team_present=True, team=replace(_TEAM, group_ids=("g",))),
        replace(_CALLER, team_present=True, team=_TEAM),
        replace(_CALLER, team_present=True, team=_TEAM, member=MemberModelFacts(True, True, True, ())),
        replace(_CALLER, user_loaded=True, user_present=True, user_grants=("no-default-models",)),
        replace(_CALLER, user_loaded=True, user_present=True, user_grants=("model",)),
    ),
)
def test_managed_caller_and_ceiling_read_denial_order(policy: Callable[[Facts], Decision], facts: Facts) -> None:
    assert policy(facts) == reference(facts)
