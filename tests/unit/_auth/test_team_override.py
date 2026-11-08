from collections.abc import Callable
from dataclasses import replace
from typing import Final

import pytest

from litellm._auth.failure import NativeAuthFailure
from litellm._auth.fallback_access import GroupedModelScope
from litellm._auth.model_access import ModelAccessFacts
from litellm._auth.team_override import TeamOverrideDecision as Decision
from litellm._auth.team_override import TeamOverrideFacts as Facts
from litellm._auth.team_override import team_override
from tests.unit._auth.support.fallback_reference import grouped
from tests.unit._auth.support.model_reference import model_access


def reference(facts: Facts) -> Decision:
    if facts.team is None:
        return Decision("ReadTeam")
    denied: Final = grouped(facts.team, merge=True)
    if denied.kind == "Allow":
        return Decision("Allow")
    if denied.kind == "ReadGroups":
        return Decision("ReadTeamGroups")
    if denied.kind != "Deny" or denied.object_type != "team":
        return Decision("Deny", denied)
    if facts.assigned_models is None:
        return Decision("ReadAssignedGroups")
    if not facts.assigned_models:
        return Decision("Deny", denied)
    if facts.assigned_scope is None:
        return Decision("ReadAssignedScope")
    assigned: Final = model_access(facts.assigned_scope)
    if assigned.kind == "Allow":
        return Decision("Allow")
    return Decision("Deny", denied if assigned.kind == "Deny" else assigned)


_MODEL: Final = ModelAccessFacts(("model",), ("other",), object_type="team")
_TEAM: Final = Facts(GroupedModelScope(_MODEL))
_ASSIGNED: Final = replace(_TEAM, assigned_models=("model",))


@pytest.mark.parametrize("policy", (reference, team_override))
@pytest.mark.parametrize(
    "facts",
    (
        Facts(),
        _TEAM,
        Facts(GroupedModelScope(replace(_MODEL, grants=()))),
        Facts(GroupedModelScope(_MODEL, ("team-group",))),
        Facts(GroupedModelScope(_MODEL, ("team-group",), ("model",))),
        Facts(GroupedModelScope(_MODEL, ("team-group",), ("no-access",))),
        Facts(GroupedModelScope(replace(_MODEL, object_type="key"))),
        Facts(GroupedModelScope(replace(_MODEL, fallback_depth=10))),
        replace(_TEAM, assigned_models=()),
        _ASSIGNED,
        replace(_ASSIGNED, assigned_scope=replace(_MODEL, object_type="key", grants=("model",))),
        replace(_ASSIGNED, assigned_scope=replace(_MODEL, object_type="key")),
        replace(_ASSIGNED, assigned_scope=replace(_MODEL, object_type="key", fallback_depth=10)),
    ),
)
def test_team_override_reads_groups_only_after_team_denial(policy: Callable[[Facts], Decision], facts: Facts) -> None:
    assert policy(facts) == reference(facts)


def test_assigned_denial_keeps_team_error_envelope() -> None:
    facts: Final = replace(_ASSIGNED, assigned_scope=replace(_MODEL, object_type="key"))
    assert team_override(facts) == Decision("Deny", model_access(_MODEL))


def test_bad_assigned_scope_is_terminal_not_team_denial() -> None:
    with pytest.raises(NativeAuthFailure):
        team_override(replace(_ASSIGNED, assigned_scope=replace(_MODEL, fallback_depth=-1)))
