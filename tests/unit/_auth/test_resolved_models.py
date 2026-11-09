from collections.abc import Callable
from dataclasses import replace
from typing import Final

import pytest

from litellm._auth.fallback_access import GroupedModelScope
from litellm._auth.member_models import MemberModelFacts
from litellm._auth.model_access import ModelAccessDecision, ModelAccessFacts
from litellm._auth.resolved_models import ResolvedModelDecision as Decision
from litellm._auth.resolved_models import ResolvedModelFacts as Facts
from litellm._auth.resolved_models import ResolvedTeamRead, resolved_models
from litellm._auth.team_override import TeamOverrideFacts
from tests.unit._auth.support.fallback_reference import grouped
from tests.unit._auth.support.model_reference import model_access
from tests.unit._auth.test_member_models import reference as member_reference
from tests.unit._auth.test_personal_model_request import reference as personal_reference
from tests.unit._auth.test_team_override import reference as team_reference


def reference(facts: Facts) -> Decision:
    if not facts.config_nonempty and "all-team-models" not in facts.key_models:
        if facts.key is None:
            return Decision("ReadKey")
        key: Final = grouped(facts.key, merge=False)
        if key.kind == "ReadGroups":
            return Decision("ReadKeyGroups")
        if key.kind != "Allow":
            return Decision("Deny", key)
    if facts.team_id is not None:
        if facts.team_read is None:
            return Decision("ReadTeamObject")
        if facts.team_read.failed and not facts.embedded_ready:
            return Decision("ReadEmbeddedTeam")
        if facts.team_read.found or facts.team_read.failed:
            if facts.team is None:
                return Decision("ReadTeamScope")
            team: Final = team_reference(facts.team)
            if team.kind == "Deny":
                return Decision("Deny", team.denial)
            if facts.team_read.found and facts.user_id is not None:
                if facts.member is None:
                    return Decision("ReadMember")
                member: Final = member_reference(facts.member)
                if member.kind == "Deny":
                    return Decision("Deny", member.denial, member.member_models)
    if (
        facts.personal_enabled
        and facts.team_id is None
        and facts.user_id is not None
        and not facts.master
        and facts.role != "proxy_admin"
    ):
        if not facts.owner_loaded:
            return Decision("ReadOwner")
        if facts.owner_failed or not facts.owner_present:
            return Decision("Deny", ModelAccessDecision("PersonalTeam", model=facts.model))
        if facts.personal is None:
            return Decision("ReadPersonal")
        personal: Final = personal_reference(facts.personal)
        if personal.kind == "Deny":
            return Decision("Deny", ModelAccessDecision("PersonalTeam", model=personal.model))
    if facts.project_present:
        if not facts.project_loaded:
            return Decision("ReadProjectObject")
        if facts.project_grants:
            if facts.project is None:
                return Decision("ReadProjectScope")
            project: Final = model_access(facts.project)
            if project.kind != "Allow":
                return Decision("Deny", project)
    if facts.customer_present and facts.database_present:
        if not facts.customer_loaded:
            return Decision("ReadCustomerObject")
        if facts.customer_grants:
            if facts.customer is None:
                return Decision("ReadCustomerScope")
            customer: Final = model_access(facts.customer)
            if customer.kind != "Allow":
                return Decision("Deny", customer)
    return Decision("Allow")


_MODEL: Final = ModelAccessFacts(("model",), ("model",))
_PASS: Final = Facts(model="model", config_nonempty=True)
_TEAM: Final = replace(
    _PASS,
    team_id="team",
    team_read=ResolvedTeamRead(found=True),
    team=TeamOverrideFacts(GroupedModelScope(replace(_MODEL, object_type="team"))),
)
_PERSONAL: Final = replace(_PASS, user_id="owner", personal_enabled=True)
_PROJECT: Final = replace(_PASS, project_present=True, project_loaded=True, project_grants=("model",))
_CUSTOMER: Final = replace(
    _PASS, customer_present=True, database_present=True, customer_loaded=True, customer_grants=("model",)
)


@pytest.mark.parametrize("policy", (reference, resolved_models))
@pytest.mark.parametrize(
    ("facts", "expected"),
    (
        (Facts(), Decision("ReadKey")),
        (_PASS, Decision("Allow")),
        (Facts(key_models=("all-team-models",)), Decision("Allow")),
        (Facts(key=GroupedModelScope(_MODEL)), Decision("Allow")),
        (
            Facts(key=GroupedModelScope(replace(_MODEL, grants=("other",)), ("group",)), project_present=True),
            Decision("ReadKeyGroups"),
        ),
        (replace(_TEAM, team_read=None), Decision("ReadTeamObject")),
        (replace(_TEAM, team_read=ResolvedTeamRead(failed=True)), Decision("ReadEmbeddedTeam")),
        (replace(_TEAM, team=None), Decision("ReadTeamScope")),
        (_TEAM, Decision("Allow")),
        (replace(_TEAM, user_id="member", project_present=True), Decision("ReadMember")),
        (
            replace(_TEAM, user_id="member", member=MemberModelFacts(True, False), project_present=True),
            Decision("ReadProjectObject"),
        ),
        (
            replace(
                _TEAM,
                user_id="member",
                team_read=ResolvedTeamRead(failed=True),
                embedded_ready=True,
                project_present=True,
            ),
            Decision("ReadProjectObject"),
        ),
        (_PERSONAL, Decision("ReadOwner")),
        (replace(_PERSONAL, master=True), Decision("Allow")),
        (replace(_PERSONAL, role="proxy_admin"), Decision("Allow")),
        (replace(_PERSONAL, owner_loaded=True), Decision("Deny", ModelAccessDecision("PersonalTeam", model="model"))),
        (
            replace(_PERSONAL, owner_loaded=True, owner_present=True, owner_failed=True),
            Decision("Deny", ModelAccessDecision("PersonalTeam", model="model")),
        ),
        (replace(_PERSONAL, owner_loaded=True, owner_present=True), Decision("ReadPersonal")),
        (replace(_PROJECT, project_loaded=False), Decision("ReadProjectObject")),
        (_PROJECT, Decision("ReadProjectScope")),
        (replace(_PROJECT, project_grants=()), Decision("Allow")),
        (
            replace(
                _PROJECT, project=replace(_MODEL, object_type="project"), customer_present=True, database_present=True
            ),
            Decision("ReadCustomerObject"),
        ),
        (replace(_CUSTOMER, database_present=False), Decision("Allow")),
        (replace(_CUSTOMER, customer_loaded=False), Decision("ReadCustomerObject")),
        (_CUSTOMER, Decision("ReadCustomerScope")),
        (replace(_CUSTOMER, customer_grants=()), Decision("Allow")),
        (replace(_CUSTOMER, customer=replace(_MODEL, object_type="customer")), Decision("Allow")),
    ),
)
def test_resolved_scope_and_lookup_order(policy: Callable[[Facts], Decision], facts: Facts, expected: Decision) -> None:
    assert policy(facts) == expected
