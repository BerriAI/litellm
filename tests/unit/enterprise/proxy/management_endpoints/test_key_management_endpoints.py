import pytest

from litellm.enterprise.litellm_enterprise.proxy.management_endpoints.key_management_endpoints import (
    add_team_member_key_duration,
)
from litellm.proxy._types import GenerateKeyRequest, LiteLLM_TeamTable


def _team(duration):
    metadata = None if duration == "no-metadata" else {"team_member_key_duration": duration}
    return LiteLLM_TeamTable(team_id="team-1", metadata=metadata)


@pytest.mark.parametrize("cleared", ["", "   ", None, "no-metadata"])
def test_cleared_team_duration_keeps_member_request(cleared):
    data = GenerateKeyRequest(user_id="member-1", duration="7d")
    result = add_team_member_key_duration(_team(cleared), data)
    assert result.duration == "7d"


@pytest.mark.parametrize("cleared", ["", "   "])
def test_cleared_team_duration_gives_no_expiry_when_member_asks_for_none(cleared):
    data = GenerateKeyRequest(user_id="member-1")
    result = add_team_member_key_duration(_team(cleared), data)
    assert result.duration is None


def test_team_duration_is_default_when_member_gives_none():
    data = GenerateKeyRequest(user_id="member-1")
    result = add_team_member_key_duration(_team("30d"), data)
    assert result.duration == "30d"


def test_shorter_member_duration_is_kept():
    data = GenerateKeyRequest(user_id="member-1", duration="7d")
    result = add_team_member_key_duration(_team("30d"), data)
    assert result.duration == "7d"


def test_longer_member_duration_is_capped_to_team_duration():
    data = GenerateKeyRequest(user_id="member-1", duration="90d")
    result = add_team_member_key_duration(_team("30d"), data)
    assert result.duration == "30d"


def test_service_account_key_is_not_changed():
    data = GenerateKeyRequest(duration="90d")
    result = add_team_member_key_duration(_team("30d"), data)
    assert result.duration == "90d"
