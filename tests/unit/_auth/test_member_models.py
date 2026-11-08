from collections.abc import Callable
from dataclasses import replace
from typing import Final

import pytest

from litellm._auth.member_models import MemberModelDecision as Decision
from litellm._auth.member_models import MemberModelFacts as Facts
from litellm._auth.member_models import member_models
from litellm._auth.model_access import ModelAccessDecision, ModelAccessFacts
from tests.unit._auth.support.model_reference import model_access


def reference(facts: Facts) -> Decision:
    if not facts.user_present or not facts.team_present:
        return Decision("Allow")
    if not facts.membership_loaded:
        return Decision("ReadMembership")
    if facts.grants is None:
        return Decision("ReadGrants")
    if not facts.grants:
        return Decision("Allow")
    if facts.scope is None:
        return Decision("ReadScope")
    denied: Final = model_access(facts.scope)
    if denied.kind == "Allow":
        return Decision("Allow")
    return Decision("Deny", denied, facts.grants if denied.kind == "Deny" else None)


_BASE: Final = Facts(True, True)
_GRANTS: Final = replace(_BASE, membership_loaded=True, grants=("allowed",))
_SCOPE: Final = ModelAccessFacts(("requested",), ("allowed",), object_type="team")


@pytest.mark.parametrize("policy", (reference, member_models))
@pytest.mark.parametrize(
    ("facts", "expected"),
    (
        (_BASE, Decision("ReadMembership")),
        (replace(_BASE, user_present=False), Decision("Allow")),
        (replace(_BASE, team_present=False), Decision("Allow")),
        (replace(_BASE, membership_loaded=True), Decision("ReadGrants")),
        (replace(_GRANTS, grants=()), Decision("Allow")),
        (_GRANTS, Decision("ReadScope")),
        (replace(_GRANTS, scope=replace(_SCOPE, grants=())), Decision("Allow")),
        (
            replace(_GRANTS, scope=_SCOPE),
            Decision("Deny", ModelAccessDecision("Deny", "requested", ("allowed",), "team"), ("allowed",)),
        ),
    ),
)
def test_member_scope_load_order_and_denial_envelope(
    policy: Callable[[Facts], Decision], facts: Facts, expected: Decision
) -> None:
    assert policy(facts) == expected
