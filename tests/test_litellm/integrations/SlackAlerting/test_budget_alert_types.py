from litellm.integrations.SlackAlerting.budget_alert_types import SoftBudgetAlert
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

    def test_get_id_team_event_group_returns_team_id(self):
        """Team soft-budget alerts must dedupe by team_id, not token (issue #27398)."""
        alert = SoftBudgetAlert()
        user_info = CallInfo(
            spend=120.0,
            token="some_active_key_token",
            team_id="team_abc",
            event_group=Litellm_EntityType.TEAM,
        )

        result = alert.get_id(user_info)
        assert result == "team_abc"

    def test_get_id_team_event_group_without_team_id_falls_back_to_token(self):
        """If event_group is TEAM but team_id is missing, fall back to token."""
        alert = SoftBudgetAlert()
        user_info = CallInfo(
            spend=120.0,
            token="key_token",
            team_id=None,
            event_group=Litellm_EntityType.TEAM,
        )

        result = alert.get_id(user_info)
        assert result == "key_token"
