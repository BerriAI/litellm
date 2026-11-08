from collections.abc import Callable
from dataclasses import replace
from typing import Final

import pytest

from litellm._auth.owner_teams import OwnerTeamFacts, OwnerTeamLoad
from litellm._auth.personal_model_request import PersonalModelRequestDecision as Decision
from litellm._auth.personal_model_request import PersonalModelRequestFacts as Facts
from litellm._auth.personal_model_request import personal_model_request
from litellm._auth.personal_models import LoadedTeamModelFacts
from tests.unit._auth.test_owner_teams import reference as teams_reference
from tests.unit._auth.test_personal_models import ALLOW, INITIAL, reference as scope_reference


def reference(facts: Facts) -> Decision:
    if facts.owner_present and facts.teams is None:
        return Decision("ReadTeams")
    teams: Final = teams_reference(facts.teams).selected if facts.teams is not None else ()
    for index, model in enumerate(facts.requested):
        if not facts.owner_present:
            return Decision("Deny", model=model)
        if index >= len(facts.models):
            return Decision("ReadModel", index=index, teams=teams)
        if not any(scope_reference(scope).allowed for scope in facts.models[index].scopes):
            return Decision("Deny", model=model)
    return Decision("Allow")


_BASE: Final = Facts(("first", "second"), True)
_TEAMS: Final = replace(
    _BASE, teams=OwnerTeamFacts("Models", ("team",), "ui", loaded=True, loads=(OwnerTeamLoad(True),))
)
_ONE: Final = replace(_TEAMS, models=(LoadedTeamModelFacts((True,), (ALLOW,)),))


@pytest.mark.parametrize("policy", (reference, personal_model_request))
@pytest.mark.parametrize(
    ("facts", "expected"),
    (
        (_BASE, Decision("ReadTeams")),
        (replace(_BASE, requested=()), Decision("ReadTeams")),
        (replace(_BASE, owner_present=False), Decision("Deny", model="first")),
        (replace(_BASE, owner_present=False, requested=()), Decision("Allow")),
        (_TEAMS, Decision("ReadModel", teams=(0,))),
        (_ONE, Decision("ReadModel", index=1, teams=(0,))),
        (replace(_ONE, models=(*_ONE.models, LoadedTeamModelFacts((True,), (ALLOW,)))), Decision("Allow")),
        (
            replace(_TEAMS, models=(LoadedTeamModelFacts((True,), (replace(INITIAL, io_failed=True),)),)),
            Decision("Deny", model="first"),
        ),
    ),
)
def test_personal_requests_read_all_teams_but_stop_at_first_model_denial(
    policy: Callable[[Facts], Decision], facts: Facts, expected: Decision
) -> None:
    assert policy(facts) == expected
