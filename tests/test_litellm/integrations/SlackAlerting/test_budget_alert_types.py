from litellm.integrations.SlackAlerting.budget_alert_types import (
    EndUserBudgetAlert,
    SoftBudgetAlert,
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


class TestEndUserBudgetAlert:
    def test_get_event_message(self):
        """Test that get_event_message returns the correct customer budget message"""
        alert = EndUserBudgetAlert()
        assert alert.get_event_message() == "Customer Budget: "

    def test_get_id_with_customer_id(self):
        """Test that get_id returns user_info.customer_id when customer_id is provided"""
        alert = EndUserBudgetAlert()
        user_info = CallInfo(
            spend=50.0,
            customer_id="customer_123",
            event_group=Litellm_EntityType.END_USER,
        )
        result = alert.get_id(user_info)
        assert result == "customer_123"

    def test_get_id_without_customer_id(self):
        """Test that get_id returns 'default_id' when customer_id is None"""
        alert = EndUserBudgetAlert()
        user_info = CallInfo(
            spend=50.0,
            customer_id=None,
            event_group=Litellm_EntityType.END_USER,
        )
        result = alert.get_id(user_info)
        assert result == "default_id"

    def test_get_id_with_empty_customer_id(self):
        """Test that get_id returns 'default_id' when customer_id is empty string"""
        alert = EndUserBudgetAlert()
        user_info = CallInfo(
            spend=50.0,
            customer_id="",
            event_group=Litellm_EntityType.END_USER,
        )
        result = alert.get_id(user_info)
        assert result == "default_id"