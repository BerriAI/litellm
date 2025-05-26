import json
import os
import sys

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path


from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter


@pytest.mark.asyncio
async def test_daily_spend_tracking_with_disabled_spend_logs():
    """
    Test that add_spend_log_transaction_to_daily_user_transaction is still called
    even when disable_spend_logs is True
    """
    # Setup
    db_writer = DBSpendUpdateWriter()

    # Mock the methods we want to track
    db_writer._insert_spend_log_to_db = AsyncMock()
    db_writer.add_spend_log_transaction_to_daily_user_transaction = AsyncMock()

    # Mock the imported modules/variables
    with patch("litellm.proxy.proxy_server.disable_spend_logs", True), patch(
        "litellm.proxy.proxy_server.prisma_client", MagicMock()
    ), patch("litellm.proxy.proxy_server.user_api_key_cache", MagicMock()), patch(
        "litellm.proxy.proxy_server.litellm_proxy_budget_name", "test-budget"
    ):
        # Test data
        test_data = {
            "token": "test-token",
            "user_id": "test-user",
            "end_user_id": "test-end-user",
            "start_time": datetime.now(),
            "end_time": datetime.now(),
            "team_id": "test-team",
            "org_id": "test-org",
            "completion_response": MagicMock(),
            "response_cost": 0.1,
            "kwargs": {"model": "gpt-4", "custom_llm_provider": "openai"},
        }

        # Call the method
        await db_writer.update_database(**test_data)

        # Verify that _insert_spend_log_to_db was NOT called (since disable_spend_logs is True)
        db_writer._insert_spend_log_to_db.assert_not_called()

        # Verify that add_spend_log_transaction_to_daily_user_transaction WAS called
        assert db_writer.add_spend_log_transaction_to_daily_user_transaction.called

        # Verify the payload passed to add_spend_log_transaction_to_daily_user_transaction
        call_args = (
            db_writer.add_spend_log_transaction_to_daily_user_transaction.call_args[1]
        )
        assert "payload" in call_args
        assert call_args["payload"]["spend"] == 0.1
        assert call_args["payload"]["model"] == "gpt-4"
        assert call_args["payload"]["custom_llm_provider"] == "openai"
