from collections.abc import Callable
from dataclasses import replace
from typing import Final

import pytest

from litellm._auth.failure import NativeAuthFailure
from litellm._auth.fallback_access import GroupedModelScope
from litellm._auth.model_access import ModelAccessFacts, personal_model_access_applies
from litellm._auth.personal_models import LoadedTeamModelFacts, loaded_team_models, personal_model_scope
from litellm._auth.personal_models import PersonalModelDecision as Decision
from litellm._auth.personal_models import PersonalModelScope as Scope
from tests.unit._auth.support.fallback_reference import grouped
from tests.unit._auth.support.model_reference import model_access


def reference(facts: Scope) -> Decision:
    if facts.io_failed:
        return Decision("Complete")
    if facts.team is None:
        return Decision("ReadTeam")
    team: Final = grouped(facts.team, merge=True)
    if team.kind == "ReadGroups":
        return Decision("ReadGroups")
    if team.kind != "Allow":
        return Decision("Complete")
    if not facts.user_present or not facts.team_id_present:
        return Decision("Complete", allowed=True)
    if facts.member_grants is None:
        return Decision("ReadMembership")
    if not facts.member_grants:
        return Decision("Complete", allowed=True)
    if facts.member is None:
        return Decision("ReadMember")
    return Decision("Complete", allowed=model_access(facts.member).kind == "Allow")


INITIAL: Final = Scope(True, True)
MODEL: Final = ModelAccessFacts(("model",), ("model",), object_type="team")
TEAM: Final = replace(INITIAL, team=GroupedModelScope(MODEL))
ALLOW: Final = replace(TEAM, member_grants=())


@pytest.mark.parametrize("policy", (reference, personal_model_scope))
@pytest.mark.parametrize(
    ("facts", "expected"),
    (
        (INITIAL, Decision("ReadTeam")),
        (replace(INITIAL, io_failed=True), Decision("Complete")),
        (TEAM, Decision("ReadMembership")),
        (replace(TEAM, team=GroupedModelScope(replace(MODEL, grants=("other",)))), Decision("Complete")),
        (replace(TEAM, team=GroupedModelScope(replace(MODEL, grants=("other",)), ("group",))), Decision("ReadGroups")),
        (
            replace(TEAM, team=GroupedModelScope(replace(MODEL, grants=("other",)), ("group",), ("model",))),
            Decision("ReadMembership"),
        ),
        (replace(TEAM, team_id_present=False), Decision("Complete", allowed=True)),
        (replace(TEAM, user_present=False), Decision("Complete", allowed=True)),
        (ALLOW, Decision("Complete", allowed=True)),
        (replace(TEAM, member_grants=("member",)), Decision("ReadMember")),
        (replace(TEAM, member_grants=("other",), member=replace(MODEL, grants=("other",))), Decision("Complete")),
        (replace(TEAM, member_grants=("model",), member=MODEL), Decision("Complete", allowed=True)),
    ),
)
def test_team_model_scope_orders_grants_before_membership(
    policy: Callable[[Scope], Decision], facts: Scope, expected: Decision
) -> None:
    assert policy(facts) == expected


@pytest.mark.parametrize(
    ("facts", "expected"),
    (
        (LoadedTeamModelFacts(()), Decision("Complete")),
        (LoadedTeamModelFacts((False, True, False, True)), Decision("ReadScopes", indices=(1, 3))),
        (
            LoadedTeamModelFacts((True, True), (ALLOW, replace(INITIAL, io_failed=True))),
            Decision("Complete", allowed=True),
        ),
        (LoadedTeamModelFacts((True,), (replace(INITIAL, io_failed=True),)), Decision("Complete")),
    ),
)
def test_loaded_teams_are_all_read_before_union(facts: LoadedTeamModelFacts, expected: Decision) -> None:
    assert loaded_team_models(facts) == expected


def test_later_native_failure_is_not_hidden_by_first_team_allow() -> None:
    with pytest.raises(NativeAuthFailure):
        loaded_team_models(LoadedTeamModelFacts((True, True), (ALLOW, INITIAL)))


@pytest.mark.parametrize(
    ("api_key_present", "inheritance", "expected"), ((False, False, True), (False, True, False), (True, True, True))
)
def test_inheritance_requires_virtual_key_not_just_owner(
    api_key_present: bool, inheritance: bool, expected: bool
) -> None:
    assert (
        personal_model_access_applies(
            enabled=True,
            token_present=True,
            team_id=None,
            user_id="",
            master=False,
            role=None,
            api_key_present=api_key_present,
            inheritance=inheritance,
        )
        is expected
    )
