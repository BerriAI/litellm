"""
Test cases for initial_budget_reset_at feature in user management endpoints.
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

from litellm.proxy._types import NewUserRequest, UpdateUserRequest, UpdateUserRequestNoUserIDorEmail
from litellm.proxy.management_endpoints.internal_user_endpoints import new_user, user_update


class TestInitialBudgetResetAtValidation:
    """Test Pydantic validation for initial_budget_reset_at field"""

    def test_past_date_rejection_user_request(self):
        """Test that NewUserRequest rejects past dates for initial_budget_reset_at"""
        past_date = datetime.now(timezone.utc) - timedelta(days=1)

        with pytest.raises(ValueError) as exc_info:
            NewUserRequest(
                user_id="test_user",
                budget_duration="10m",
                initial_budget_reset_at=past_date,
            )

        assert "initial_budget_reset_at cannot be in the past" in str(exc_info.value)

    def test_future_date_accepted(self):
        """Test that future dates are accepted"""
        future_date = datetime.now(timezone.utc) + timedelta(days=10)

        user_request = NewUserRequest(
            user_id="test_user",
            budget_duration="10m",
            initial_budget_reset_at=future_date,
        )

        assert user_request.initial_budget_reset_at == future_date

    def test_past_date_rejection_update_user_request(self):
        """Test that UpdateUserRequest rejects past dates for initial_budget_reset_at"""
        past_date = datetime.now(timezone.utc) - timedelta(days=1)

        with pytest.raises(ValueError) as exc_info:
            UpdateUserRequest(
                user_id="test_user",
                budget_duration="10m",
                initial_budget_reset_at=past_date,
            )

        assert "initial_budget_reset_at cannot be in the past" in str(exc_info.value)

    def test_past_date_rejection_update_user_request_no_id_or_email(self):
        """Test that UpdateUserRequestNoUserIDorEmail rejects past dates for initial_budget_reset_at"""
        past_date = datetime.now(timezone.utc) - timedelta(days=1)

        with pytest.raises(ValueError) as exc_info:
            UpdateUserRequestNoUserIDorEmail(
                budget_duration="10m",
                initial_budget_reset_at=past_date,
            )

        assert "initial_budget_reset_at cannot be in the past" in str(exc_info.value)

    def test_future_date_accepted_update_user_request(self):
        """Test that future dates are accepted on UpdateUserRequest"""
        future_date = datetime.now(timezone.utc) + timedelta(days=10)

        update_request = UpdateUserRequest(
            user_id="test_user",
            budget_duration="10m",
            initial_budget_reset_at=future_date,
        )

        assert update_request.initial_budget_reset_at == future_date


class TestUserCreationWithInitialBudgetResetAt:
    """Integration test for creating users with initial_budget_reset_at"""

    @pytest.mark.asyncio
    async def test_create_user_explicit_initial_budget_reset_at_takes_precedence(self):
        """Test that explicit initial_budget_reset_at is honored and takes precedence over duration-based computation"""
        mock_prisma_client = MagicMock()
        mock_prisma_client.insert_data = AsyncMock(
            return_value=MagicMock(user_id="new_user", spend=0.0, models=[])
        )
        mock_prisma_client.db.litellm_usertable.count = AsyncMock(return_value=0)

        explicit_date = datetime.now(timezone.utc) + timedelta(days=30)

        user_request = NewUserRequest(
            user_id="new_user",
            budget_duration="10m",
            initial_budget_reset_at=explicit_date,
            max_budget=100.0,
        )

        mock_license_check = MagicMock()
        mock_license_check.is_over_limit.return_value = False

        with patch(
            "litellm.proxy.proxy_server.prisma_client",
            mock_prisma_client,
        ), patch(
            "litellm.proxy.proxy_server._license_check",
            mock_license_check,
        ):
            result = await new_user(
                data=user_request,
                user_api_key_dict=MagicMock(user_id="admin", user_role="proxy_admin"),
            )

            assert result is not None
            assert mock_prisma_client.insert_data.called

            # find the user insert call and verify budget_reset_at
            for call in mock_prisma_client.insert_data.call_args_list:
                if call.kwargs.get("table_name") == "user":
                    user_data = call.kwargs["data"]
                    assert "budget_reset_at" in user_data
                    stored_date = user_data["budget_reset_at"]
                    assert isinstance(stored_date, datetime)
                    assert abs((stored_date - explicit_date).total_seconds()) < 1 # verify dates are effectively equal
                    break
            else:
                raise AssertionError("User insert call not found")


class TestUserUpdateWithInitialBudgetResetAt:
    """Integration test for updating users with initial_budget_reset_at"""

    @pytest.mark.asyncio
    async def test_update_user_explicit_initial_budget_reset_at_translated(self):
        """Test that initial_budget_reset_at is translated to budget_reset_at on the update path"""
        mock_prisma_client = MagicMock()
        mock_prisma_client.db.litellm_usertable.find_first = AsyncMock(return_value=None)
        mock_prisma_client.update_data = AsyncMock(
            return_value={"user_id": "existing_user"}
        )

        explicit_date = datetime.now(timezone.utc) + timedelta(days=30)

        update_request = UpdateUserRequest(
            user_id="existing_user",
            budget_duration="10m",
            initial_budget_reset_at=explicit_date,
            max_budget=100.0,
        )

        with patch(
            "litellm.proxy.proxy_server.prisma_client",
            mock_prisma_client,
        ), patch(
            "litellm.proxy.proxy_server.litellm_proxy_admin_name",
            "admin",
        ):
            result = await user_update(
                data=update_request,
                user_api_key_dict=MagicMock(
                    user_id="existing_user",
                    user_role="proxy_admin",
                ),
            )

            assert result is not None
            assert mock_prisma_client.update_data.called

            call_kwargs = mock_prisma_client.update_data.call_args.kwargs
            user_data = call_kwargs["data"]
            assert "budget_reset_at" in user_data, "budget_reset_at should be set from initial_budget_reset_at"
            assert "initial_budget_reset_at" not in user_data, "initial_budget_reset_at should be removed"
            stored_date = user_data["budget_reset_at"]
            assert isinstance(stored_date, datetime)
            assert abs((stored_date - explicit_date).total_seconds()) < 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
