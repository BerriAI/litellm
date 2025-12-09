import json
import os
import sys

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path


from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch, call

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


@pytest.mark.asyncio
async def test_update_daily_spend_sorting():
    """
    Test that table.upsert is called with events sorted

    Ensures that writes are sorted between transactions to minimize deadlocks
    """
    # Setup
    mock_prisma_client = MagicMock()
    mock_batcher = MagicMock()
    mock_table = MagicMock()
    mock_prisma_client.db.batch_.return_value.__aenter__.return_value = mock_batcher
    mock_batcher.litellm_dailyuserspend = mock_table

    # Create a 50 transactions with out-of-order entity_ids
    # In reality we sort using multiple fields, but entity_id is sufficient to test sorting
    daily_spend_transactions = {}
    upsert_calls = []
    for i in range(50):
        daily_spend_transactions[f"test_key_{i}"] = {
            "user_id": f"user{60-i}", # user60 ... user11, reverse order
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
        upsert_calls.append(call(
            where={
                "user_id_date_api_key_model_custom_llm_provider": {
                    "user_id": f"user{i+11}", # user11 ... user60, sorted order
                    "date": "2024-01-01",
                    "api_key": "test-api-key",
                    "model": "gpt-4",
                    "custom_llm_provider": "openai",
                    "mcp_namespaced_tool_name": "",
                }
            },
            data={
                "create": {
                    "user_id": f"user{i+11}",
                    "date": "2024-01-01",
                    "api_key": "test-api-key",
                    "model": "gpt-4",
                    "model_group": None,
                    "mcp_namespaced_tool_name": "",
                    "custom_llm_provider": "openai",
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "spend": 0.1,
                    "api_requests": 1,
                    "successful_requests": 1,
                    "failed_requests": 0,
                },
                "update": {
                    "prompt_tokens": {"increment": 10},
                    "completion_tokens": {"increment": 20},
                    "spend": {"increment": 0.1},
                    "api_requests": {"increment": 1},
                    "successful_requests": {"increment": 1},
                    "failed_requests": {"increment": 0},
                },
            },
        ))

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
    mock_table.upsert.assert_has_calls(upsert_calls)
# Tag Spend Tracking Tests


@pytest.mark.asyncio
async def test_update_tag_db_with_valid_tags():
    """
    Test that _update_tag_db correctly processes valid tags and adds them to the spend update queue.
    """
    from litellm.proxy._types import Litellm_EntityType, SpendUpdateQueueItem

    writer = DBSpendUpdateWriter()
    mock_prisma = MagicMock()
    response_cost = 0.05
    request_tags = '["prod-tag", "test-tag"]'

    writer.spend_update_queue.add_update = AsyncMock()

    await writer._update_tag_db(
        response_cost=response_cost,
        request_tags=request_tags,
        prisma_client=mock_prisma,
    )

    assert writer.spend_update_queue.add_update.call_count == 2

    first_call_args = writer.spend_update_queue.add_update.call_args_list[0][1]
    assert first_call_args["update"]["entity_type"] == Litellm_EntityType.TAG
    assert first_call_args["update"]["entity_id"] == "prod-tag"
    assert first_call_args["update"]["response_cost"] == response_cost

    second_call_args = writer.spend_update_queue.add_update.call_args_list[1][1]
    assert second_call_args["update"]["entity_type"] == Litellm_EntityType.TAG
    assert second_call_args["update"]["entity_id"] == "test-tag"
    assert second_call_args["update"]["response_cost"] == response_cost


@pytest.mark.asyncio
async def test_update_tag_db_with_list_input():
    """
    Test that _update_tag_db correctly handles tags passed as a list instead of JSON string.
    """
    writer = DBSpendUpdateWriter()
    mock_prisma = MagicMock()
    response_cost = 0.1
    request_tags = ["tag1", "tag2", "tag3"]

    writer.spend_update_queue.add_update = AsyncMock()

    await writer._update_tag_db(
        response_cost=response_cost,
        request_tags=request_tags,
        prisma_client=mock_prisma,
    )

    assert writer.spend_update_queue.add_update.call_count == 3


@pytest.mark.asyncio
async def test_update_tag_db_with_no_tags():
    """
    Test that _update_tag_db handles None and empty tags gracefully.
    """
    writer = DBSpendUpdateWriter()
    mock_prisma = MagicMock()
    response_cost = 0.05

    writer.spend_update_queue.add_update = AsyncMock()

    await writer._update_tag_db(
        response_cost=response_cost,
        request_tags=None,
        prisma_client=mock_prisma,
    )
    assert writer.spend_update_queue.add_update.call_count == 0

    await writer._update_tag_db(
        response_cost=response_cost,
        request_tags=[],
        prisma_client=mock_prisma,
    )
    assert writer.spend_update_queue.add_update.call_count == 0


@pytest.mark.asyncio
async def test_update_tag_db_with_invalid_json():
    """
    Test that _update_tag_db handles invalid JSON gracefully.
    """
    writer = DBSpendUpdateWriter()
    mock_prisma = MagicMock()
    response_cost = 0.05
    request_tags = '{"invalid": json}'

    writer.spend_update_queue.add_update = AsyncMock()

    await writer._update_tag_db(
        response_cost=response_cost,
        request_tags=request_tags,
        prisma_client=mock_prisma,
    )

    assert writer.spend_update_queue.add_update.call_count == 0


@pytest.mark.asyncio
async def test_update_tag_db_without_prisma_client():
    """
    Test that _update_tag_db returns early when prisma_client is None.
    """
    writer = DBSpendUpdateWriter()
    response_cost = 0.05
    request_tags = '["tag1"]'

    writer.spend_update_queue.add_update = AsyncMock()

    await writer._update_tag_db(
        response_cost=response_cost,
        request_tags=request_tags,
        prisma_client=None,
    )

    assert writer.spend_update_queue.add_update.call_count == 0

@pytest.mark.asyncio
async def test_add_spend_log_transaction_to_daily_tag_transaction_with_request_id():
    """
    Test that add_spend_log_transaction_to_daily_tag_transaction correctly processes request_id.
    This tests that request_id is included in the DailyTagSpendTransaction for the LiteLLM_DailyTagSpend table.
    """
    writer = DBSpendUpdateWriter()
    mock_prisma = MagicMock()
    mock_prisma.get_request_status = MagicMock(return_value="success")
    
    request_id = "test-request-id-123"
    payload = {
        "request_id": request_id,
        "request_tags": '["prod-tag", "test-tag"]',
        "user": "test-user",
        "startTime": "2024-01-01T00:00:00",
        "api_key": "test-key",
        "model": "gpt-4",
        "custom_llm_provider": "openai",
        "model_group": "gpt-4-group",
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "spend": 0.05,
        "metadata": '{"usage_object": {}}',
    }

    # Mock the add_update method to capture what's being added
    original_add_update = writer.daily_tag_spend_update_queue.add_update
    writer.daily_tag_spend_update_queue.add_update = AsyncMock()

    await writer.add_spend_log_transaction_to_daily_tag_transaction(
        payload=payload,
        prisma_client=mock_prisma,
    )

    # Should be called twice (once for each tag)
    assert writer.daily_tag_spend_update_queue.add_update.call_count == 2
    
    # Check that request_id is included in both transactions
    for call in writer.daily_tag_spend_update_queue.add_update.call_args_list:
        transaction_dict = call[1]["update"]
        # Each transaction should have one key with the format tag_date_api_key_model_provider
        for key, transaction in transaction_dict.items():
            assert transaction["request_id"] == request_id, f"request_id should be {request_id} but got {transaction.get('request_id')}"