import math
from collections.abc import Callable
from dataclasses import replace
from typing import Final

import pytest

from litellm._auth.common_budgets import BudgetFacts, MemberLimitFacts
from litellm._auth.failure import NativeAuthFailure
from litellm._auth.personal_budget import PersonalTeamBudgetDecision as Decision
from litellm._auth.personal_budget import PersonalTeamBudgetFacts as Facts
from litellm._auth.personal_budget import personal_team_budget


def denied(budget: BudgetFacts) -> bool:
    if budget.limit is None or budget.spend is None or not math.isfinite(budget.limit):
        return False
    return budget.spend > budget.limit if budget.scope == "Team" else budget.spend >= budget.limit


def reference(facts: Facts) -> Decision:
    org: Final = facts.token_org or facts.team_org
    if not facts.source_ready:
        return Decision("Source", org_id=org)
    if facts.io_failed:
        return Decision("Complete")
    if facts.team is None:
        return Decision("ReadTeam")
    if denied(facts.team):
        return Decision("Complete") if facts.team_alerted else Decision("AlertTeam")
    if facts.window_count is None:
        return Decision("ReadWindows")
    for index in range(facts.window_count):
        if index >= len(facts.windows):
            return Decision("ReadWindow", index=index)
        if denied(facts.windows[index]):
            return Decision("Complete")
    if facts.check_db_only and not facts.user_present:
        return Decision("Complete")
    if facts.user_present and (facts.team_id_present or facts.check_db_only) and not facts.membership_loaded:
        return Decision("ReadMembership")
    if facts.user_present and facts.team_id_present:
        limit: Final = facts.member_limit
        if limit is None:
            return Decision("ReadMemberLimit")
        if limit.member_limit is None and facts.default_id is not None and not facts.default_loaded:
            return Decision("ReadDefault", id=facts.default_id)
        base: Final = (
            limit.member_limit
            if limit.member_limit is not None
            else limit.default_limit
            if limit.default_limit is not None and not limit.default_limit <= 0
            else None
        )
        if (
            base is not None
            and limit.temporary_increase is not None
            and limit.expires_at is not None
            and limit.now_naive
        ):
            return Decision("Complete", log_failure=True)
        increase: Final = (
            limit.temporary_increase
            if limit.temporary_increase is not None and limit.expires_at is not None and limit.expires_at > limit.now
            else 0.0
        )
        if facts.member is None:
            return Decision("ReadMember", limit=base + increase if base is not None else None)
        if facts.member.limit is not None and math.isfinite(facts.member.limit) and not facts.member_alerted:
            return Decision("AlertMember")
        if denied(facts.member):
            return Decision("Complete")
    if org is not None:
        if not facts.org_loaded:
            return Decision("ReadOrg", id=org)
        if not facts.org_failed and facts.org is not None and denied(facts.org):
            return Decision("Complete") if facts.org_alerted else Decision("AlertOrg", id=org)
    return Decision("Complete", allowed=True)


INITIAL: Final = Facts(user_present=True, token_org="org")
TEAM: Final = replace(INITIAL, source_ready=True, team=BudgetFacts("Team", None, token_present=True))
WINDOWS: Final = replace(TEAM, window_count=0)
MEMBERSHIP: Final = replace(WINDOWS, membership_loaded=True)
LIMIT: Final = replace(MEMBERSHIP, member_limit=MemberLimitFacts(None, None, None, None, 100))
MEMBER: Final = replace(LIMIT, member=BudgetFacts("TeamMember", None))
ALLOW: Final = replace(MEMBER, org_loaded=True)


@pytest.mark.parametrize("policy", (reference, personal_team_budget))
@pytest.mark.parametrize(
    ("facts", "expected"),
    (
        (INITIAL, Decision("Source", org_id="org")),
        (replace(INITIAL, token_org="", team_org="team-org"), Decision("Source", org_id="team-org")),
        (replace(INITIAL, token_org=None, team_org=""), Decision("Source", org_id="")),
        (replace(INITIAL, source_ready=True), Decision("ReadTeam")),
        (replace(TEAM, io_failed=True), Decision("Complete")),
        (replace(TEAM, team=BudgetFacts("Team", 1, 2, token_present=True)), Decision("AlertTeam")),
        (replace(TEAM, team=BudgetFacts("Team", 1, 2, token_present=True), team_alerted=True), Decision("Complete")),
        (replace(TEAM, team=BudgetFacts("Team", 1, 1, token_present=True)), Decision("ReadWindows")),
        (TEAM, Decision("ReadWindows")),
        (replace(TEAM, window_count=2), Decision("ReadWindow")),
        (replace(TEAM, window_count=2, windows=(BudgetFacts("TeamWindow", 1, 0),)), Decision("ReadWindow", index=1)),
        (replace(TEAM, window_count=2, windows=(BudgetFacts("TeamWindow", 1, 1),)), Decision("Complete")),
        (replace(WINDOWS, user_present=False, check_db_only=True), Decision("Complete")),
        (replace(WINDOWS, user_present=False), Decision("ReadOrg", id="org")),
        (WINDOWS, Decision("ReadMembership")),
        (replace(WINDOWS, team_id_present=False), Decision("ReadOrg", id="org")),
        (replace(WINDOWS, team_id_present=False, check_db_only=True), Decision("ReadMembership")),
        (replace(MEMBERSHIP, team_id_present=False, check_db_only=True), Decision("ReadOrg", id="org")),
        (MEMBERSHIP, Decision("ReadMemberLimit")),
        (replace(LIMIT, default_id=""), Decision("ReadDefault")),
        (
            replace(LIMIT, default_id="default", member_limit=MemberLimitFacts(0, None, None, None, 100)),
            Decision("ReadMember", limit=0),
        ),
        (
            replace(
                LIMIT, default_id="default", default_loaded=True, member_limit=MemberLimitFacts(None, 10, 2, 101, 100)
            ),
            Decision("ReadMember", limit=12),
        ),
        (replace(LIMIT, member_limit=MemberLimitFacts(10, None, 2, 100, 100)), Decision("ReadMember", limit=10)),
        (
            replace(LIMIT, member_limit=MemberLimitFacts(10, None, 2, 101, 100, True)),
            Decision("Complete", log_failure=True),
        ),
        (LIMIT, Decision("ReadMember")),
        (replace(MEMBER, member=BudgetFacts("TeamMember", 1, 1)), Decision("AlertMember")),
        (replace(MEMBER, member=BudgetFacts("TeamMember", 1, 1), member_alerted=True), Decision("Complete")),
        (replace(MEMBER, member=BudgetFacts("TeamMember", float("inf"), 100)), Decision("ReadOrg", id="org")),
        (MEMBER, Decision("ReadOrg", id="org")),
        (replace(MEMBER, token_org=None), Decision("Complete", allowed=True)),
        (ALLOW, Decision("Complete", allowed=True)),
        (replace(ALLOW, org_failed=True), Decision("Complete", allowed=True)),
        (replace(ALLOW, org=BudgetFacts("Organization", 1, 1)), Decision("AlertOrg", id="org")),
        (replace(ALLOW, org=BudgetFacts("Organization", 1, 1), org_alerted=True), Decision("Complete")),
    ),
)
def test_personal_budget_read_and_denial_order(
    policy: Callable[[Facts], Decision], facts: Facts, expected: Decision
) -> None:
    assert policy(facts) == expected


def test_incomplete_budget_read_is_terminal_protocol_failure() -> None:
    with pytest.raises(NativeAuthFailure):
        personal_team_budget(replace(TEAM, team=BudgetFacts("Team", 1)))
