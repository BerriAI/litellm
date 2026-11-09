from dataclasses import replace
from itertools import product
from math import inf, nan
from typing import Final

import pytest

from litellm._auth import reservation

_SCOPES: Final[tuple[reservation.Scope, ...]] = (
    "Key",
    "Team",
    "User",
    "EndUser",
    "Tag",
    "AccessGroup",
    "Member",
    "Organization",
    "Project",
)
_SOURCES: Final = tuple(
    reservation.Sources(key, team, team_id, user, user_id, apply)
    for key, team, team_id, user, user_id, apply in product(
        (False, True), (False, True), (None, "", "team"), (False, True), (None, "", "user"), (False, True)
    )
)


@pytest.mark.parametrize("scope", _SCOPES)
@pytest.mark.parametrize("limit", (None, -inf, -1.0, 0.0, 1.0, inf, nan))
def test_reservation_counter_reference(scope: reservation.Scope, limit: float | None) -> None:
    expected: Final = limit is not None and (
        limit > 0
        if scope in ("Key", "Team", "User")
        else limit > 0 and limit < inf
        if scope == "Project"
        else not limit <= 0
    )
    assert reservation.counter(scope, limit) == expected


@pytest.mark.parametrize("facts", _SOURCES)
def test_source_selection_preserves_team_key_user_opt_in(facts: reservation.Sources) -> None:
    team: Final = facts.team_present and facts.team_id is not None
    assert reservation.sources(facts) == reservation.SourcePlan(
        facts.key_present,
        team,
        (not team or facts.apply_user_budget_to_team_keys) and facts.user_present and facts.user_id is not None,
    )


@pytest.mark.parametrize("cost", (None, -inf, -1.0, 0.0, 1.0, inf, nan))
def test_estimate_and_reservation_staging(cost: float | None) -> None:
    initial: Final = reservation.Gate(True)
    route: Final = replace(initial, llm_route=True)
    billable: Final = replace(route, unbilled=False)
    model: Final = replace(billable, model_present=True)
    counters: Final = replace(model, counter_count=1)
    assert tuple(reservation.action(facts).kind for facts in (initial, route, billable, model, counters)) == (
        "ReadRoute",
        "ReadUnbilled",
        "ReadModel",
        "ReadCounters",
        "ReadEstimate",
    )
    assert reservation.action(replace(counters, cost_read=True, cost=cost)).kind == (
        "Skip" if cost is None or cost <= 0 else "Reserve"
    )


@pytest.mark.parametrize("duration", (False, True))
@pytest.mark.parametrize("limit", (None, -1.0, 0.0, 1.0, inf, nan))
@pytest.mark.parametrize("parsed", (None, False, True))
def test_window_parse_and_warning_timing(duration: bool, limit: float | None, parsed: bool | None) -> None:
    expected: Final = (
        "Skip"
        if not duration or limit is None or limit <= 0
        else "ReadStart"
        if parsed is None
        else "AppendWindow"
        if parsed
        else "WarnWindow"
    )
    assert reservation.action(reservation.Window(duration, limit, parsed)).kind == expected


@pytest.mark.parametrize("scope", ("EndUser", "Organization", "Project"))
@pytest.mark.parametrize("primary", (None, "", "primary"))
@pytest.mark.parametrize("fallback", (None, "", "fallback"))
def test_reservation_identity_keeps_empty_scope_semantics(
    scope: str, primary: str | None, fallback: str | None
) -> None:
    from pydantic import TypeAdapter

    facts: Final = TypeAdapter(reservation.Identity).validate_python(
        {"scope": scope, "primary": primary, "fallback": fallback}
    )
    expected: Final = (
        primary or fallback
        if scope == "EndUser"
        else primary
        if scope == "Project" or primary is not None
        else fallback
    )
    assert reservation.identity(facts) == expected


@pytest.mark.parametrize("team", (False, True))
@pytest.mark.parametrize("user", (False, True))
@pytest.mark.parametrize("team_id", (None, "", "team"))
@pytest.mark.parametrize("user_id", (None, "", "user"))
def test_reservation_membership_requires_both_subjects(
    team: bool, user: bool, team_id: str | None, user_id: str | None
) -> None:
    expected: Final = "ReadMember" if team and team_id is not None and user and user_id is not None else "Skip"
    assert reservation.action(reservation.MemberGate(team, team_id, user, user_id)).kind == expected


@pytest.mark.parametrize("cap", (None, -1.0, 0.0, 1.0, inf, nan))
@pytest.mark.parametrize("read", (False, True))
def test_member_default_read_only_when_own_cap_absent(cap: float | None, read: bool) -> None:
    expected: Final = "ReadDefault" if cap is None and not read else "ReadCap"
    assert reservation.action(reservation.MemberCap(cap, read)).kind == expected
