from collections.abc import Callable
from dataclasses import replace
from typing import Final

import pytest

from litellm._auth.model_attribution import ModelGroupAttributionDecision as Decision
from litellm._auth.model_attribution import ModelGroupAttributionFacts as Facts
from litellm._auth.model_attribution import model_group_attribution


def reference(facts: Facts) -> Decision:
    if facts.io_failed or not all(
        (facts.model_present, facts.token_present, facts.router_present, facts.database_present)
    ):
        return Decision("Complete")
    if not facts.registry_loaded:
        return Decision("ReadRegistry")
    if facts.registry is not None and not facts.registry:
        return Decision("Complete")
    if facts.covering is None:
        return Decision("ReadCovering")
    budgeted: Final = set(facts.covering) if facts.registry is None else set(facts.covering) & set(facts.registry)
    if not budgeted:
        return Decision("Complete")
    match len(facts.grants):
        case 0:
            return Decision("ReadKey")
        case 1:
            return Decision("ReadTeam")
        case 2:
            return Decision(
                "ReadMember",
                include=facts.team_present and facts.user_present,
                lookup=facts.team_present and facts.user_present and not facts.membership_loaded,
            )
        case 3:
            return Decision("ReadProject")
        case 4:
            return Decision("ReadOrg", id=facts.token_org or facts.team_org)
        case _:
            scopes: Final = facts.grants[:4] if facts.org_failed else facts.grants
            granted: Final = set[str]().union(*scopes)
            groups: Final = tuple(sorted(budgeted & granted))
            return Decision("Complete", groups, stamp=bool(groups))


_BASE: Final = Facts(model_present=True, token_present=True, router_present=True, database_present=True)
_COVERED: Final = replace(_BASE, registry_loaded=True, registry=("a", "b"), covering=("b", "a", "c", "b"))
_FINAL: Final = replace(_COVERED, grants=(("a",), (), ("b",), (), ("c",)))


@pytest.mark.parametrize("policy", (reference, model_group_attribution))
@pytest.mark.parametrize(
    ("facts", "expected"),
    (
        (_BASE, Decision("ReadRegistry")),
        (replace(_BASE, model_present=False), Decision("Complete")),
        (replace(_BASE, token_present=False), Decision("Complete")),
        (replace(_BASE, router_present=False), Decision("Complete")),
        (replace(_BASE, database_present=False), Decision("Complete")),
        (replace(_BASE, io_failed=True), Decision("Complete")),
        (replace(_BASE, registry_loaded=True, registry=()), Decision("Complete")),
        (replace(_BASE, registry_loaded=True), Decision("ReadCovering")),
        (replace(_COVERED, covering=()), Decision("Complete")),
        (_COVERED, Decision("ReadKey")),
        (replace(_COVERED, grants=((),)), Decision("ReadTeam")),
        (replace(_COVERED, grants=((), ())), Decision("ReadMember")),
        (
            replace(_COVERED, grants=((), ()), team_present=True, user_present=True),
            Decision("ReadMember", include=True, lookup=True),
        ),
        (
            replace(_COVERED, grants=((), ()), team_present=True, user_present=True, membership_loaded=True),
            Decision("ReadMember", include=True),
        ),
        (replace(_COVERED, grants=((), (), ())), Decision("ReadProject")),
        (replace(_COVERED, grants=((), (), (), ()), token_org="key", team_org="team"), Decision("ReadOrg", id="key")),
        (replace(_COVERED, grants=((), (), (), ()), token_org="", team_org="team"), Decision("ReadOrg", id="team")),
        (_FINAL, Decision("Complete", ("a", "b"), stamp=True)),
        (replace(_FINAL, registry=None), Decision("Complete", ("a", "b", "c"), stamp=True)),
        (replace(_FINAL, registry=None, org_failed=True), Decision("Complete", ("a", "b"), stamp=True)),
        (replace(_FINAL, grants=(("*",), (), (), (), ())), Decision("Complete")),
    ),
)
def test_attribution_case_table(policy: Callable[[Facts], Decision], facts: Facts, expected: Decision) -> None:
    assert policy(facts) == expected
