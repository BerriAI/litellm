from collections.abc import Callable
from dataclasses import replace
from typing import Final

import pytest

from litellm._auth.failure import NativeAuthFailure
from litellm._auth.owner_teams import OwnerTeamBudget, OwnerTeamLoad, owner_teams
from litellm._auth.owner_teams import OwnerTeamDecision as Decision
from litellm._auth.owner_teams import OwnerTeamFacts as Facts
from tests.unit._auth.test_personal_budget import ALLOW, INITIAL
from tests.unit._auth.test_personal_budget import reference as budget


def reference(facts: Facts) -> Decision:
    if facts.purpose == "Grants" and not (facts.user_present and facts.database_present):
        return Decision("Complete")
    if not facts.loaded:
        ids: Final = tuple(dict.fromkeys(facts.ids))
        return Decision(
            "ReadTeams",
            ids=tuple(team for team in ids if facts.purpose == "Models" or team and team != facts.ui_team_id),
        )
    eligible: Final = tuple(
        index
        for index, row in enumerate(facts.loads)
        if row.present
        and not row.blocked
        and (facts.purpose == "Models" or facts.org is None or row.org in (None, facts.org))
    )
    unreadable: Final = any(row.failed for row in facts.loads)
    if facts.purpose == "Models":
        return Decision("Complete", selected=eligible, unreadable=unreadable)
    missing: Final = tuple(index for index in eligible if not any(read.index == index for read in facts.budgets))
    if missing:
        return Decision("ReadBudgets", indices=missing)
    outcomes: Final = tuple((read.index, budget(read.facts).allowed) for read in facts.budgets)
    blocked: Final = tuple(
        index
        for index, row in enumerate(facts.loads)
        if row.present and row.blocked and (facts.org is None or row.org in (None, facts.org))
    )
    return Decision(
        "Complete",
        selected=tuple(index for index in eligible if (index, True) in outcomes),
        withheld=(*blocked, *(index for index in eligible if (index, False) in outcomes)),
        unreadable=unreadable,
    )


_BASE: Final = Facts("Grants", ("", "ui", "a", "a", "b"), "ui", org="org")
_ROWS: Final = replace(
    _BASE,
    loaded=True,
    loads=(
        OwnerTeamLoad(True, blocked=True, org="org"),
        OwnerTeamLoad(True, org="org"),
        OwnerTeamLoad(True, org="other"),
        OwnerTeamLoad(False, failed=True),
        OwnerTeamLoad(True),
    ),
)
_COMPLETE: Final = replace(
    _ROWS, budgets=(OwnerTeamBudget(1, ALLOW), OwnerTeamBudget(4, replace(ALLOW, io_failed=True)))
)


@pytest.mark.parametrize("policy", (reference, owner_teams))
@pytest.mark.parametrize(
    ("facts", "expected"),
    (
        (_BASE, Decision("ReadTeams", ids=("a", "b"))),
        (replace(_BASE, user_present=False), Decision("Complete")),
        (replace(_BASE, database_present=False), Decision("Complete")),
        (replace(_BASE, purpose="Models"), Decision("ReadTeams", ids=("", "ui", "a", "b"))),
        (_ROWS, Decision("ReadBudgets", indices=(1, 4))),
        (replace(_ROWS, purpose="Models"), Decision("Complete", selected=(1, 2, 4), unreadable=True)),
        (_COMPLETE, Decision("Complete", selected=(1,), withheld=(0, 4), unreadable=True)),
        (replace(_ROWS, org=None), Decision("ReadBudgets", indices=(1, 2, 4))),
        (replace(_ROWS, org=""), Decision("ReadBudgets", indices=(4,))),
    ),
)
def test_owner_team_selection_keeps_order_scope_and_withheld_rows(
    policy: Callable[[Facts], Decision], facts: Facts, expected: Decision
) -> None:
    assert policy(facts) == expected


def test_later_incomplete_budget_is_not_hidden_by_an_allowed_team() -> None:
    with pytest.raises(NativeAuthFailure):
        owner_teams(replace(_ROWS, budgets=(OwnerTeamBudget(1, ALLOW), OwnerTeamBudget(4, INITIAL))))
