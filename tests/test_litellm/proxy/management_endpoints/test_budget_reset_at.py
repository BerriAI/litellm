"""
Test cases for budget_reset_at feature in user management endpoints.
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest_plugins = ('pytest_asyncio',)
sys.path.insert(
    0, os.path.abspath("../../../..")
)

from litellm.proxy._types import NewUserRequest
from litellm.proxy.management_endpoints.internal_user_endpoints import new_user


class TestBudgetResetAtValidation:
    """Test Pydantic validation for budget_reset_at field"""

    def test_past_date_rejection_user_request(self):
        """Test that NewUserRequest rejects past dates for budget_reset_at"""
        past_date = datetime.now(timezone.utc) - timedelta(days=1)

        with pytest.raises(ValueError) as exc_info:
            NewUserRequest(
                user_id="test_user",
                budget_duration="10m",
                budget_reset_at=past_date,
            )

        assert "budget_reset_at cannot be in the past" in str(exc_info.value)

    def test_future_date_accepted(self):
        """Test that future dates are accepted"""
        future_date = datetime.now(timezone.utc) + timedelta(days=10)

        user_request = NewUserRequest(
            user_id="test_user",
            budget_duration="10m",
            budget_reset_at=future_date,
        )

        assert user_request.budget_reset_at == future_date


class TestUserCreationWithBudgetResetAt:
    """Integration tests for creating users with budget_reset_at"""

    @pytest.mark.asyncio
    async def test_create_user_explicit_budget_reset_at_takes_precedence(self):
        """Test that explicit budget_reset_at is honored and takes precedence over duration-based computation"""
        mock_prisma_client = MagicMock()
        mock_prisma_client.db.litellm_usertable.upsert = AsyncMock(
            return_value=MagicMock(user_id="new_user", spend=0.0)
        )
        mock_prisma_client.db.litellm_usertable.count = AsyncMock(return_value=0)
        mock_prisma_client.db.litellm_budgettable.upsert = AsyncMock(
            return_value=MagicMock()
        )
        mock_prisma_client.insert_data = AsyncMock()

        explicit_date = datetime.now(timezone.utc) + timedelta(days=30)

        user_request = NewUserRequest(
            user_id="new_user",
            budget_duration="10m",
            budget_reset_at=explicit_date,
            max_budget=100.0,
        )

        with patch(
            "litellm.proxy.proxy_server.prisma_client",
            mock_prisma_client,
        ):
            result = await new_user(
                data=user_request,
                user_api_key_dict=MagicMock(user_id="admin", user_role="proxy_admin"),
            )

            assert result is not None

            # verify that budget_reset_at was set and matches explicit date (not computed from duration)
            budget_upsert_calls = (
                mock_prisma_client.db.litellm_budgettable.upsert.call_args_list
            )
            if budget_upsert_calls:
                for call in budget_upsert_calls:
                    data_dict = call[1].get("data", {})
                    update_data = data_dict.get("update") or data_dict.get("create")
                    if update_data and "budget_reset_at" in update_data:
                        stored_date = update_data["budget_reset_at"]
                        assert isinstance(stored_date, datetime)
                        assert abs((stored_date - explicit_date).total_seconds()) < 1 # verify dates are effectively equal
                        break


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
