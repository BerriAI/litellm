from dataclasses import replace
from typing import Final

import pytest

from litellm._auth.fallback_access import FallbackAuthorizationFacts, GroupedModelScope, fallback_authorization
from litellm._auth.model_access import ModelAccessFacts

_ALLOW: Final = ModelAccessFacts(("model",), ("model",), object_type="key")
_DENY: Final = replace(_ALLOW, grants=("other",))
_BASE: Final = FallbackAuthorizationFacts(GroupedModelScope(_ALLOW))


@pytest.mark.parametrize(
    ("facts", "kind"),
    (
        pytest.param(_BASE, "Allow", id="no-extra-ceilings"),
        pytest.param(replace(_BASE, refuse_agent_keys=True, agent_id=""), "Deny", id="agent-presence-not-truthiness"),
        pytest.param(
            replace(_BASE, key=GroupedModelScope(_DENY, ("group",))), "ReadKeyGroups", id="key-groups-before-team"
        ),
        pytest.param(
            replace(_BASE, key=GroupedModelScope(_DENY, ("group",), ())), "Deny", id="empty-key-group-retains-denial"
        ),
        pytest.param(
            replace(_BASE, key=GroupedModelScope(_DENY, ("group",), ("model",))), "Allow", id="key-group-fallback"
        ),
        pytest.param(
            replace(_BASE, team=GroupedModelScope(_DENY, ("group",))), "ReadTeamGroups", id="team-group-fallback"
        ),
        pytest.param(
            replace(_BASE, team=GroupedModelScope(_DENY)), "ReadAssignedGroups", id="team-denial-asks-owned-key-grants"
        ),
        pytest.param(
            replace(_BASE, team=GroupedModelScope(_DENY), assigned_models=()),
            "Deny",
            id="empty-assigned-does-not-widen",
        ),
        pytest.param(
            replace(_BASE, team=GroupedModelScope(_DENY), assigned_models=("model",)),
            "Allow",
            id="assigned-overrides-team",
        ),
        pytest.param(
            replace(_BASE, embedded_team=_DENY, assigned_models=("model",)),
            "Deny",
            id="embedded-team-no-assigned-override",
        ),
        pytest.param(replace(_BASE, member=_DENY), "Deny", id="member-ceiling"),
        pytest.param(replace(_BASE, project=_DENY), "Deny", id="project-ceiling"),
    ),
)
def test_fallback_authorization_stage_order(facts: FallbackAuthorizationFacts, kind: str) -> None:
    assert fallback_authorization(facts).kind == kind


def test_key_denial_wins_before_later_scopes() -> None:
    decision: Final = fallback_authorization(
        replace(_BASE, key=GroupedModelScope(_DENY), project=replace(_DENY, object_type="project"))
    )
    assert decision.denial is not None
    assert decision.denial.object_type == "key"


@pytest.mark.parametrize(
    ("facts", "kind"),
    (
        (FallbackAuthorizationFacts(), "ReadKey"),
        (FallbackAuthorizationFacts(refuse_agent_keys=True, agent_id="agent"), "Deny"),
        (replace(_BASE, team_present=True), "ReadTeam"),
        (replace(_BASE, embedded_team_grants=("model",)), "ReadEmbeddedTeam"),
        (replace(_BASE, member_grants=("model",)), "ReadMember"),
        (replace(_BASE, project_grants=("model",)), "ReadProject"),
        (replace(_BASE, key=GroupedModelScope(_DENY), team_present=True, member_grants=("model",)), "Deny"),
        (replace(_BASE, member=_DENY, project_grants=("model",)), "Deny"),
    ),
)
def test_later_scope_projection_is_requested_only_after_prior_allow(
    facts: FallbackAuthorizationFacts, kind: str
) -> None:
    assert fallback_authorization(facts).kind == kind
