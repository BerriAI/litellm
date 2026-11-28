import pytest
from litellm_enterprise.proxy.management_endpoints.key_management_endpoints import add_team_member_key_duration

from litellm.proxy._types import GenerateKeyRequest, LiteLLM_TeamTable


def _member_key(duration: str | None, team_duration: object) -> str | None:
    team = LiteLLM_TeamTable(team_id="t", metadata={"team_member_key_duration": team_duration})
    return add_team_member_key_duration(team, GenerateKeyRequest(user_id="member", duration=duration)).duration


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        (None, "30d"),
        ("1d", "1d"),
        ("30d", "30d"),
        ("4w", "4w"),
        ("31d", "30d"),
        ("2mo", "30d"),
        ("-1", "30d"),
        ("not-a-duration", "30d"),
    ],
    ids=[
        "unset-gets-default",
        "shorter-kept",
        "equal-kept",
        "shorter-other-unit-kept",
        "longer-capped",
        "months-capped",
        "never-expires-capped",
        "malformed-capped",
    ],
)
def test_team_key_duration_is_the_default_and_the_cap(requested: str | None, expected: str) -> None:
    assert _member_key(requested, "30d") == expected


@pytest.mark.parametrize("team_duration", [None, "", "   ", 30])
def test_no_usable_team_key_duration_leaves_the_member_request_alone(team_duration: object) -> None:
    assert _member_key("365d", team_duration) == "365d"
    assert _member_key(None, team_duration) is None


def test_service_account_keys_ignore_the_team_key_duration() -> None:
    team = LiteLLM_TeamTable(team_id="t", metadata={"team_member_key_duration": "1d"})
    key = add_team_member_key_duration(team, GenerateKeyRequest(user_id=None, duration="365d"))
    assert key.duration == "365d"
