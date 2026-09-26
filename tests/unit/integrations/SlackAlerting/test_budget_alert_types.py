from litellm.integrations.SlackAlerting.budget_alert_types import (
    SoftBudgetAlert,
    TokenBudgetAlert,
)
from litellm.proxy._types import CallInfo, Litellm_EntityType


class TestSoftBudgetAlert:
    def test_get_id_with_token(self):
        """Test that get_id returns user_info.token when token is provided"""
        token_value = "test_token_123"
        alert = SoftBudgetAlert()
        user_info = CallInfo(
            spend=120.0,
            token=token_value,
            event_group=Litellm_EntityType.KEY,
        )

        result = alert.get_id(user_info)
        assert result == token_value

    def test_get_id_without_token(self):
        """Test that get_id returns 'default_id' when token is None"""
        alert = SoftBudgetAlert()
        user_info = CallInfo(
            spend=100.0,
            token=None,
            event_group=Litellm_EntityType.KEY,
        )

        result = alert.get_id(user_info)
        assert result == "default_id"

    def test_get_id_returns_team_id_for_team_event_group(self):
        """Team soft budget alerts dedupe by team, not by the calling key's token"""
        alert = SoftBudgetAlert()
        user_info = CallInfo(
            spend=120.0,
            token="test_token_123",
            team_id="team_456",
            event_group=Litellm_EntityType.TEAM,
        )

        result = alert.get_id(user_info)
        assert result == "team_456"

    def test_get_id_returns_default_id_for_team_event_group_without_team_id(self):
        alert = SoftBudgetAlert()
        user_info = CallInfo(
            spend=120.0,
            token="test_token_123",
            team_id=None,
            event_group=Litellm_EntityType.TEAM,
        )

        result = alert.get_id(user_info)
        assert result == "default_id"

    def test_get_id_with_empty_token(self):
        """Test that get_id returns 'default_id' when token is empty string"""
        alert = SoftBudgetAlert()
        user_info = CallInfo(
            spend=100.0,
            token="",
            event_group=Litellm_EntityType.KEY,
        )

        result = alert.get_id(user_info)
        assert result == "default_id"


class TestTokenBudgetAlert:
    def test_get_id_dedupes_team_member_alerts_per_member_and_team(self):
        alert = TokenBudgetAlert()
        team_a = CallInfo(
            spend=8.0, max_budget=10.0, user_id="member_1", team_id="team_a", event_group=Litellm_EntityType.TEAM_MEMBER
        )
        team_b = CallInfo(
            spend=8.0, max_budget=10.0, user_id="member_1", team_id="team_b", event_group=Litellm_EntityType.TEAM_MEMBER
        )

        assert alert.get_id(team_a) == "team_member:member_1:team_a"
        assert alert.get_id(team_b) == "team_member:member_1:team_b"

    def test_get_id_uses_token_for_key_alerts(self):
        alert = TokenBudgetAlert()
        user_info = CallInfo(
            spend=8.0,
            max_budget=10.0,
            token="hashed_key",
            user_id="member_1",
            team_id="team_a",
            event_group=Litellm_EntityType.KEY,
        )

        assert alert.get_id(user_info) == "hashed_key"
        assert alert.get_event_message() == "Key Budget: "
