from collections.abc import Callable
from dataclasses import replace
from typing import Final

import pytest

from litellm._auth.owner_teams import OwnerTeamFacts, OwnerTeamLoad
from litellm._auth.personal_owner import PersonalOwnerDecision as Decision
from litellm._auth.personal_owner import PersonalOwnerFacts as Facts
from litellm._auth.personal_owner import personal_owner
from tests.unit._auth.test_owner_teams import reference as select_teams


def reference(facts: Facts) -> Decision:
    if not all((facts.user_present, facts.database_present, facts.cache_present, facts.logging_present)):
        return Decision("Complete")
    if not facts.owner_loaded:
        return Decision("ReadOwner")
    if facts.owner_failed or not facts.owner_present:
        return Decision("Complete")
    if facts.teams is None:
        return Decision("ReadTeams")
    teams: Final = select_teams(facts.teams)
    return Decision("Complete", teams.selected, teams.withheld, teams.unreadable)


_BASE: Final = Facts(True, True, True, True)
_OWNER: Final = replace(_BASE, owner_loaded=True, owner_present=True)


@pytest.mark.parametrize("policy", (reference, personal_owner))
@pytest.mark.parametrize(
    ("facts", "expected"),
    (
        (_BASE, Decision("ReadOwner")),
        (replace(_BASE, user_present=False), Decision("Complete")),
        (replace(_BASE, database_present=False), Decision("Complete")),
        (replace(_BASE, cache_present=False), Decision("Complete")),
        (replace(_BASE, logging_present=False), Decision("Complete")),
        (replace(_BASE, owner_loaded=True), Decision("Complete")),
        (replace(_OWNER, owner_failed=True), Decision("Complete")),
        (_OWNER, Decision("ReadTeams")),
        (
            replace(
                _OWNER,
                teams=OwnerTeamFacts(
                    "Grants",
                    (),
                    "ui",
                    loaded=True,
                    loads=(OwnerTeamLoad(True, blocked=True), OwnerTeamLoad(False, failed=True)),
                ),
            ),
            Decision("Complete", withheld=(0,), unreadable=True),
        ),
    ),
)
def test_owner_lookup_and_team_status_order(
    policy: Callable[[Facts], Decision], facts: Facts, expected: Decision
) -> None:
    assert policy(facts) == expected
