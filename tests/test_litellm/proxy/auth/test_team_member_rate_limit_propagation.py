"""
Verifies that the combined-view query result names align with what
LiteLLM_VerificationTokenView and UserAPIKeyAuth read for team member
rate limits.

The actual SQL query in litellm/proxy/utils.py was verified manually
against a real Postgres (see PR description). This module guards the
Python-side contract: if the SQL aliases drift away from these field
names, the rate limit will silently stop being enforced again — the
exact failure mode of https://github.com/BerriAI/litellm/issues/26378.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.proxy._types import (
    LiteLLM_VerificationTokenView,
    UserAPIKeyAuth,
)


def test_view_carries_team_member_rate_limits_from_combined_view_row():
    """
    Construct the view from the same kwargs the SQL row delivers.
    A row with team_member_tpm_limit / team_member_rpm_limit columns
    must populate the matching fields on the view.
    """
    row = {
        "token": "hashed-test-token",
        "user_id": "user-1",
        "team_id": "team-1",
        "team_member_spend": 0.5,
        "team_member_tpm_limit": 100,
        "team_member_rpm_limit": 3,
    }

    view = LiteLLM_VerificationTokenView(**row)

    assert view.team_member_spend == 0.5
    assert view.team_member_tpm_limit == 100
    assert view.team_member_rpm_limit == 3


def test_user_api_key_auth_inherits_team_member_rate_limits_from_view():
    """
    The auth flow converts the view to UserAPIKeyAuth via
    `UserAPIKeyAuth(**view.model_dump(exclude_none=True))`. The two
    team_member fields must round-trip so the rate limiter
    (parallel_request_limiter_v3) sees them.
    """
    view = LiteLLM_VerificationTokenView(
        token="hashed-test-token",
        user_id="user-1",
        team_id="team-1",
        team_member_tpm_limit=100,
        team_member_rpm_limit=3,
    )

    auth = UserAPIKeyAuth(**view.model_dump(exclude_none=True))

    assert auth.team_member_tpm_limit == 100
    assert auth.team_member_rpm_limit == 3


def test_view_leaves_team_member_rate_limits_none_when_missing():
    """
    LEFT JOIN against a membership without a budget row produces NULLs
    for these columns. The view must accept that and leave the fields
    None — otherwise the rate limiter would mis-fire on members with
    no per-member limits configured.
    """
    view = LiteLLM_VerificationTokenView(
        token="hashed-test-token",
        user_id="user-1",
        team_id="team-1",
        team_member_spend=0.0,
    )

    assert view.team_member_tpm_limit is None
    assert view.team_member_rpm_limit is None
