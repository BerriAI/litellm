from itertools import product
from typing import Final

import pytest

from litellm._auth import auth_context

_FETCH: Final = tuple(
    auth_context.Fetch(team, "ui", user, project, end_user)
    for team, user, project, end_user in product(
        (None, "ui", "", "team"), (None, "", "user"), (None, "", "project"), (None, "", "customer")
    )
)
_TEAM: Final = tuple(
    auth_context.Team(team, "ui", failed, missing, models, allow)
    for team, failed, missing, models, allow in product(
        (None, "ui", "team"), (False, True), (False, True), ((), ("model",)), (None, False, True)
    )
)
_ADMIN: Final = tuple(
    auth_context.Admin(role, user, "admin", row)
    for role, user, row in product((None, "proxy_admin", "internal_user"), (None, "admin", "user"), (False, True))
)


@pytest.mark.parametrize("facts", _FETCH)
def test_context_fetches_preserve_missing_and_empty_ids(facts: auth_context.Fetch) -> None:
    assert auth_context.fetch(facts) == auth_context.FetchPlan(
        facts.team_id is not None and facts.team_id != facts.ui_team_id,
        facts.user_id is not None,
        facts.project_id is not None,
        bool(facts.end_user_id),
    )


@pytest.mark.parametrize("facts", _TEAM)
def test_team_context_preserves_fallback_rules(facts: auth_context.Team) -> None:
    expected: Final = (
        ("Token" if facts.team_id == facts.ui_team_id else "Loaded")
        if not facts.failed
        else (
            "None"
            if facts.team_id is None
            else "Token"
            if facts.team_id == facts.ui_team_id
            else "Failure"
            if facts.missing
            else "Token"
            if facts.recorded_models
            else "ReadUnavailablePolicy"
            if facts.allow_unavailable is None
            else "Token"
            if facts.allow_unavailable
            else "Failure"
        )
    )
    assert auth_context.team(facts).source == expected


@pytest.mark.parametrize("facts", _ADMIN)
def test_admin_projection_preserves_personal_budgets(facts: auth_context.Admin) -> None:
    assert auth_context.admin(facts) == auth_context.AdminPlan(
        facts.role == "proxy_admin", facts.user_row and facts.user_id != facts.admin_id
    )
