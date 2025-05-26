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


@pytest.mark.asyncio
async def test_update_daily_spend_with_null_entity_id():
    """
    Test that table.upsert is called even when entity_id is null

    Ensures 'global view' has all daily spend transactions
    """
    # Setup
    mock_prisma_client = MagicMock()
    mock_batcher = MagicMock()
    mock_table = MagicMock()
    mock_prisma_client.db.batch_.return_value.__aenter__.return_value = mock_batcher
    mock_batcher.litellm_dailyuserspend = mock_table

    # Create a transaction with null entity_id
    daily_spend_transactions = {
        "test_key": {
            "user_id": None,  # null entity_id
            "date": "2024-01-01",
            "api_key": "test-api-key",
            "model": "gpt-4",
            "custom_llm_provider": "openai",
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "spend": 0.1,
            "api_requests": 1,
            "successful_requests": 1,
            "failed_requests": 0,
        }
    }

    # Call the method
    await DBSpendUpdateWriter._update_daily_spend(
        n_retry_times=1,
        prisma_client=mock_prisma_client,
        proxy_logging_obj=MagicMock(),
        daily_spend_transactions=daily_spend_transactions,
        entity_type="user",
        entity_id_field="user_id",
        table_name="litellm_dailyuserspend",
        unique_constraint_name="user_id_date_api_key_model_custom_llm_provider",
    )

    # Verify that table.upsert was called
    mock_table.upsert.assert_called_once()

    # Verify the where clause contains null entity_id
    call_args = mock_table.upsert.call_args[1]
    where_clause = call_args["where"]["user_id_date_api_key_model_custom_llm_provider"]
    assert where_clause["user_id"] is None
    assert where_clause["date"] == "2024-01-01"
    assert where_clause["api_key"] == "test-api-key"
    assert where_clause["model"] == "gpt-4"
    assert where_clause["custom_llm_provider"] == "openai"

    # Verify the create data contains null entity_id
    create_data = call_args["data"]["create"]
    assert create_data["user_id"] is None
    assert create_data["date"] == "2024-01-01"
    assert create_data["api_key"] == "test-api-key"
    assert create_data["model"] == "gpt-4"
    assert create_data["custom_llm_provider"] == "openai"
    assert create_data["prompt_tokens"] == 10
    assert create_data["completion_tokens"] == 20
    assert create_data["spend"] == 0.1
    assert create_data["api_requests"] == 1
    assert create_data["successful_requests"] == 1
    assert create_data["failed_requests"] == 0
