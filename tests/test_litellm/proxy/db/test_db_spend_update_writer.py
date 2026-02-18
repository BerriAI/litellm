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
        unique_constraint_name="user_id_date_api_key_model_custom_llm_provider_mcp_namespaced_tool_name_endpoint",
    )

    # Verify that table.upsert was called
    mock_table.upsert.assert_called_once()

    # Verify the where clause contains null entity_id
    call_args = mock_table.upsert.call_args[1]
    where_clause = call_args["where"]["user_id_date_api_key_model_custom_llm_provider_mcp_namespaced_tool_name_endpoint"]
    assert where_clause["user_id"] is None
    assert where_clause["date"] == "2024-01-01"
    assert where_clause["api_key"] == "test-api-key"
    assert where_clause["model"] == "gpt-4"
    assert where_clause["custom_llm_provider"] == "openai"
    assert where_clause["mcp_namespaced_tool_name"] == ""
    assert where_clause["endpoint"] == ""

    # Verify the create data contains null entity_id
    create_data = call_args["data"]["create"]
    assert create_data["user_id"] is None
    assert create_data["date"] == "2024-01-01"
    assert create_data["api_key"] == "test-api-key"
    assert create_data["model"] == "gpt-4"
    assert create_data["custom_llm_provider"] == "openai"
    assert create_data["mcp_namespaced_tool_name"] == ""
    assert create_data["endpoint"] == ""
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
                "user_id_date_api_key_model_custom_llm_provider_mcp_namespaced_tool_name_endpoint": {
                    "user_id": f"user{i+11}", # user11 ... user60, sorted order
                    "date": "2024-01-01",
                    "api_key": "test-api-key",
                    "model": "gpt-4",
                    "custom_llm_provider": "openai",
                    "mcp_namespaced_tool_name": "",
                    "endpoint": "",
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
                    "endpoint": "",
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
                    "endpoint": "",
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
        unique_constraint_name="user_id_date_api_key_model_custom_llm_provider_mcp_namespaced_tool_name_endpoint",
    )

    # Verify that table.upsert was called
    mock_table.upsert.assert_has_calls(upsert_calls)


@pytest.mark.asyncio
async def test_update_daily_spend_tag_with_request_id():
    """
    Test that request_id is included in update_data when updating tag transactions.
    """
    # Setup
    mock_prisma_client = MagicMock()
    mock_batcher = MagicMock()
    mock_table = MagicMock()
    mock_prisma_client.db.batch_.return_value.__aenter__.return_value = mock_batcher
    mock_batcher.litellm_dailytagspend = mock_table

    # Create a transaction with request_id
    daily_spend_transactions = {
        "test_key": {
            "tag": "prod-tag",
            "date": "2024-01-01",
            "api_key": "test-api-key",
            "model": "gpt-4",
            "custom_llm_provider": "openai",
            "mcp_namespaced_tool_name": "",
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "spend": 0.1,
            "api_requests": 1,
            "successful_requests": 1,
            "failed_requests": 0,
            "request_id": "test-request-id-123",
        }
    }

    # Call the method
    await DBSpendUpdateWriter._update_daily_spend(
        n_retry_times=1,
        prisma_client=mock_prisma_client,
        proxy_logging_obj=MagicMock(),
        daily_spend_transactions=daily_spend_transactions,
        entity_type="tag",
        entity_id_field="tag",
        table_name="litellm_dailytagspend",
        unique_constraint_name="tag_date_api_key_model_custom_llm_provider_mcp_namespaced_tool_name",
    )

    # Verify that table.upsert was called
    mock_table.upsert.assert_called_once()
    
    # Verify request_id is in update_data
    call_args = mock_table.upsert.call_args[1]
    update_data = call_args["data"]["update"]
    assert "request_id" in update_data
    assert update_data["request_id"] == "test-request-id-123"




@pytest.mark.asyncio
async def test_update_daily_spend_with_none_values_in_sorting_fields():
    """
    Test that _update_daily_spend handles None values in sorting fields correctly.
    
    This test ensures that when fields like date, api_key, model, or custom_llm_provider
    are None, the sorting doesn't crash with TypeError: '<' not supported between 
    instances of 'NoneType' and 'str'.
    """
    # Setup
    mock_prisma_client = MagicMock()
    mock_batcher = MagicMock()
    mock_table = MagicMock()
    mock_prisma_client.db.batch_.return_value.__aenter__.return_value = mock_batcher
    mock_batcher.litellm_dailyuserspend = mock_table

    # Create transactions with None values in various sorting fields
    daily_spend_transactions = {
        "key1": {
            "user_id": "user1",
            "date": None,  # None date
            "api_key": "test-api-key",
            "model": "gpt-4",
            "custom_llm_provider": "openai",
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "spend": 0.1,
            "api_requests": 1,
            "successful_requests": 1,
            "failed_requests": 0,
        },
        "key2": {
            "user_id": "user2",
            "date": "2024-01-01",
            "api_key": None,  # None api_key
            "model": "gpt-4",
            "custom_llm_provider": "openai",
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "spend": 0.1,
            "api_requests": 1,
            "successful_requests": 1,
            "failed_requests": 0,
        },
        "key3": {
            "user_id": "user3",
            "date": "2024-01-01",
            "api_key": "test-api-key",
            "model": None,  # None model
            "custom_llm_provider": "openai",
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "spend": 0.1,
            "api_requests": 1,
            "successful_requests": 1,
            "failed_requests": 0,
        },
        "key4": {
            "user_id": "user4",
            "date": "2024-01-01",
            "api_key": "test-api-key",
            "model": "gpt-4",
            "custom_llm_provider": None,  # None custom_llm_provider
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "spend": 0.1,
            "api_requests": 1,
            "successful_requests": 1,
            "failed_requests": 0,
        },
        "key5": {
            "user_id": None,  # None entity_id
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
        },
    }

    # Call the method - this should not raise TypeError
    await DBSpendUpdateWriter._update_daily_spend(
        n_retry_times=1,
        prisma_client=mock_prisma_client,
        proxy_logging_obj=MagicMock(),
        daily_spend_transactions=daily_spend_transactions,
        entity_type="user",
        entity_id_field="user_id",
        table_name="litellm_dailyuserspend",
        unique_constraint_name="user_id_date_api_key_model_custom_llm_provider_mcp_namespaced_tool_name_endpoint",
    )

    # Verify that table.upsert was called (should be called 5 times, once for each transaction)
    assert mock_table.upsert.call_count == 5


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


@pytest.mark.asyncio
async def test_add_spend_log_transaction_to_daily_org_transaction_injects_org_id_and_queues_update():
    """
    Verify org_id is injected into payload for daily aggregation and the update is queued.
    """
    writer = DBSpendUpdateWriter()
    mock_prisma = MagicMock()
    mock_prisma.get_request_status = MagicMock(return_value="success")

    org_id = "org-xyz"
    payload = {
        "request_id": "req-1",
        "user": "test-user",
        "startTime": "2024-01-01T12:00:00",
        "api_key": "test-key",
        "model": "gpt-4",
        "custom_llm_provider": "openai",
        "model_group": "gpt-4-group",
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "spend": 0.2,
        "metadata": '{"usage_object": {}}',
    }

    writer.daily_org_spend_update_queue.add_update = AsyncMock()

    await writer.add_spend_log_transaction_to_daily_org_transaction(
        payload=payload,
        prisma_client=mock_prisma,
        org_id=org_id,
    )

    # Should enqueue one org spend update
    writer.daily_org_spend_update_queue.add_update.assert_called_once()

    # Validate key and injected fields
    call_args = writer.daily_org_spend_update_queue.add_update.call_args[1]
    update_dict = call_args["update"]
    assert len(update_dict) == 1
    for key, transaction in update_dict.items():
        assert key == f"{org_id}_2024-01-01_test-key_gpt-4_openai_"
        assert transaction["organization_id"] == org_id
        assert transaction["date"] == "2024-01-01"
        assert transaction["api_key"] == "test-key"
        assert transaction["model"] == "gpt-4"
        assert transaction["custom_llm_provider"] == "openai"


@pytest.mark.asyncio
async def test_add_spend_log_transaction_to_daily_org_transaction_skips_when_org_id_missing():
    """
    Ensure no update is queued when org_id is None.
    """
    writer = DBSpendUpdateWriter()
    mock_prisma = MagicMock()
    mock_prisma.get_request_status = MagicMock(return_value="success")

    payload = {
        "request_id": "req-2",
        "user": "test-user",
        "startTime": "2024-01-01T12:00:00",
        "api_key": "test-key",
        "model": "gpt-4",
        "custom_llm_provider": "openai",
        "model_group": "gpt-4-group",
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "spend": 0.2,
        "metadata": '{"usage_object": {}}',
    }

    writer.daily_org_spend_update_queue.add_update = AsyncMock()

    await writer.add_spend_log_transaction_to_daily_org_transaction(
        payload=payload,
        prisma_client=mock_prisma,
        org_id=None,
    )

    writer.daily_org_spend_update_queue.add_update.assert_not_called()


@pytest.mark.asyncio
async def test_add_spend_log_transaction_to_daily_end_user_transaction_injects_end_user_id_and_queues_update():
    writer = DBSpendUpdateWriter()
    mock_prisma = MagicMock()
    mock_prisma.get_request_status = MagicMock(return_value="success")

    end_user_id = "end-user-xyz"
    payload = {
        "request_id": "req-1",
        "user": "test-user",
        "end_user": end_user_id,
        "startTime": "2024-01-01T12:00:00",
        "api_key": "test-key",
        "model": "gpt-4",
        "custom_llm_provider": "openai",
        "model_group": "gpt-4-group",
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "spend": 0.2,
        "metadata": '{"usage_object": {}}',
    }

    writer.daily_end_user_spend_update_queue.add_update = AsyncMock()

    await writer.add_spend_log_transaction_to_daily_end_user_transaction(
        payload=payload,
        prisma_client=mock_prisma,
    )

    writer.daily_end_user_spend_update_queue.add_update.assert_called_once()

    call_args = writer.daily_end_user_spend_update_queue.add_update.call_args[1]
    update_dict = call_args["update"]
    assert len(update_dict) == 1
    for key, transaction in update_dict.items():
        assert key == f"{end_user_id}_2024-01-01_test-key_gpt-4_openai_"
        assert transaction["end_user_id"] == end_user_id
        assert transaction["date"] == "2024-01-01"
        assert transaction["api_key"] == "test-key"
        assert transaction["model"] == "gpt-4"
        assert transaction["custom_llm_provider"] == "openai"


@pytest.mark.asyncio
async def test_add_spend_log_transaction_to_daily_end_user_transaction_skips_when_end_user_id_missing():
    writer = DBSpendUpdateWriter()
    mock_prisma = MagicMock()
    mock_prisma.get_request_status = MagicMock(return_value="success")

    payload = {
        "request_id": "req-2",
        "user": "test-user",
        "startTime": "2024-01-01T12:00:00",
        "api_key": "test-key",
        "model": "gpt-4",
        "custom_llm_provider": "openai",
        "model_group": "gpt-4-group",
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "spend": 0.2,
        "metadata": '{"usage_object": {}}',
    }

    writer.daily_end_user_spend_update_queue.add_update = AsyncMock()

    await writer.add_spend_log_transaction_to_daily_end_user_transaction(
        payload=payload,
        prisma_client=mock_prisma,
    )

    writer.daily_end_user_spend_update_queue.add_update.assert_not_called()


@pytest.mark.asyncio
async def test_add_spend_log_transaction_to_daily_agent_transaction_injects_agent_id_and_queues_update():
    """
    Ensure agent_id is injected and queued for daily aggregation.
    """
    writer = DBSpendUpdateWriter()
    mock_prisma = MagicMock()
    mock_prisma.get_request_status = MagicMock(return_value="success")

    agent_id = "agent-123"
    payload = {
        "request_id": "req-123",
        "agent_id": agent_id,
        "user": "test-user",
        "startTime": "2024-01-01T12:00:00",
        "api_key": "test-key",
        "model": "gpt-4",
        "custom_llm_provider": "openai",
        "model_group": "gpt-4-group",
        "prompt_tokens": 20,
        "completion_tokens": 10,
        "spend": 0.3,
        "metadata": '{"usage_object": {}}',
    }

    writer.daily_agent_spend_update_queue.add_update = AsyncMock()

    await writer.add_spend_log_transaction_to_daily_agent_transaction(
        payload=payload,
        prisma_client=mock_prisma,
    )

    writer.daily_agent_spend_update_queue.add_update.assert_called_once()

    call_args = writer.daily_agent_spend_update_queue.add_update.call_args[1]
    update_dict = call_args["update"]
    assert len(update_dict) == 1
    for key, transaction in update_dict.items():
        assert key == f"{agent_id}_2024-01-01_test-key_gpt-4_openai_"
        assert transaction["agent_id"] == agent_id
        assert transaction["date"] == "2024-01-01"
        assert transaction["api_key"] == "test-key"
        assert transaction["model"] == "gpt-4"
        assert transaction["custom_llm_provider"] == "openai"


@pytest.mark.asyncio
async def test_add_spend_log_transaction_to_daily_agent_transaction_calls_common_helper_once():
    writer = DBSpendUpdateWriter()
    mock_prisma = MagicMock()
    mock_prisma.get_request_status = MagicMock(return_value="success")

    payload = {
        "request_id": "req-common-helper",
        "agent_id": "agent-abc",
        "user": "test-user",
        "startTime": "2024-01-01T12:00:00",
        "api_key": "test-key",
        "model": "gpt-4",
        "custom_llm_provider": "openai",
        "model_group": "gpt-4-group",
        "prompt_tokens": 12,
        "completion_tokens": 6,
        "spend": 0.25,
        "metadata": '{"usage_object": {}}',
    }

    writer.daily_agent_spend_update_queue.add_update = AsyncMock()
    original_common_helper = (
        writer._common_add_spend_log_transaction_to_daily_transaction
    )
    writer._common_add_spend_log_transaction_to_daily_transaction = AsyncMock(
        wraps=original_common_helper
    )

    await writer.add_spend_log_transaction_to_daily_agent_transaction(
        payload=payload,
        prisma_client=mock_prisma,
    )

    assert (
        writer._common_add_spend_log_transaction_to_daily_transaction.await_count == 1
    )


@pytest.mark.asyncio
async def test_add_spend_log_transaction_to_daily_agent_transaction_skips_when_agent_id_missing():
    """
    Do not queue agent spend updates when agent_id is None.
    """
    writer = DBSpendUpdateWriter()
    mock_prisma = MagicMock()
    mock_prisma.get_request_status = MagicMock(return_value="success")

    payload = {
        "request_id": "req-456",
        "agent_id": None,
        "user": "test-user",
        "startTime": "2024-01-01T12:00:00",
        "api_key": "test-key",
        "model": "gpt-4",
        "custom_llm_provider": "openai",
        "model_group": "gpt-4-group",
        "prompt_tokens": 15,
        "completion_tokens": 5,
        "spend": 0.1,
        "metadata": '{"usage_object": {}}',
    }

    writer.daily_agent_spend_update_queue.add_update = AsyncMock()

    await writer.add_spend_log_transaction_to_daily_agent_transaction(
        payload=payload,
        prisma_client=mock_prisma,
    )

    writer.daily_agent_spend_update_queue.add_update.assert_not_called()


@pytest.mark.asyncio
async def test_endpoint_field_is_correctly_mapped_from_call_type():
    """
    Test that the endpoint field is correctly mapped from call_type using ROUTE_ENDPOINT_MAPPING.
    Verifies that when call_type is provided, the endpoint is set in the transaction and included in the key.
    """
    writer = DBSpendUpdateWriter()
    mock_prisma = MagicMock()
    mock_prisma.get_request_status = MagicMock(return_value="success")

    payload = {
        "request_id": "req-endpoint-test",
        "user": "test-user",
        "call_type": "acompletion",  # Maps to "/chat/completions"
        "startTime": "2024-01-01T12:00:00",
        "api_key": "test-key",
        "model": "gpt-4",
        "custom_llm_provider": "openai",
        "model_group": "gpt-4-group",
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "spend": 0.15,
        "metadata": '{"usage_object": {}}',
    }

    writer.daily_spend_update_queue.add_update = AsyncMock()

    await writer.add_spend_log_transaction_to_daily_user_transaction(
        payload=payload,
        prisma_client=mock_prisma,
    )

    writer.daily_spend_update_queue.add_update.assert_called_once()

    call_args = writer.daily_spend_update_queue.add_update.call_args[1]
    update_dict = call_args["update"]
    assert len(update_dict) == 1
    
    for key, transaction in update_dict.items():
        # Verify endpoint is included in the key
        assert key == f"test-user_2024-01-01_test-key_gpt-4_openai_/chat/completions"
        
        # Verify endpoint is set in the transaction
        assert transaction["endpoint"] == "/chat/completions"
        assert transaction["user_id"] == "test-user"
        assert transaction["date"] == "2024-01-01"
        assert transaction["api_key"] == "test-key"
        assert transaction["model"] == "gpt-4"
        assert transaction["custom_llm_provider"] == "openai"


@pytest.mark.asyncio
async def test_update_daily_spend_logs_detailed_error_on_batch_upsert_failure():
    """
    Test that when batch upsert fails, detailed error information is logged.
    This ensures proper debugging information is available for issues like unique constraint violations.
    """
    from litellm._logging import verbose_proxy_logger
    
    # Setup
    mock_prisma_client = MagicMock()
    mock_batcher = MagicMock()
    mock_table = MagicMock()
    mock_batch_context = MagicMock()
    mock_batch_context.__aenter__ = AsyncMock(return_value=mock_batcher)
    mock_batcher.litellm_dailyuserspend = mock_table
    
    # Make the batch context manager's exit raise an exception
    # This simulates a batch commit failure (e.g., unique constraint violation)
    test_exception = Exception("Unique constraint violation")
    mock_batch_context.__aexit__ = AsyncMock(side_effect=test_exception)
    mock_prisma_client.db.batch_.return_value = mock_batch_context
    
    # Create a transaction
    daily_spend_transactions = {
        "test_key": {
            "user_id": "test-user",
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
    
    # Create a mock proxy_logging_obj with failure_handler as AsyncMock
    mock_proxy_logging = MagicMock()
    mock_proxy_logging.failure_handler = AsyncMock()
    
    # Mock the logger to capture exception calls
    with patch.object(verbose_proxy_logger, 'exception') as mock_exception_logger:
        # Call the method and expect it to raise the exception
        with pytest.raises(Exception, match="Unique constraint violation"):
            await DBSpendUpdateWriter._update_daily_spend(
                n_retry_times=0,  # No retries to make test faster
                prisma_client=mock_prisma_client,
                proxy_logging_obj=mock_proxy_logging,
                daily_spend_transactions=daily_spend_transactions,
                entity_type="user",
                entity_id_field="user_id",
                table_name="litellm_dailyuserspend",
                unique_constraint_name="user_id_date_api_key_model_custom_llm_provider_mcp_namespaced_tool_name_endpoint",
            )
        
        # Verify that exception was logged with detailed information
        assert mock_exception_logger.called
        call_args = mock_exception_logger.call_args[0][0]
        assert "Daily user spend batch upsert failed" in call_args
        assert "Table: litellm_dailyuserspend" in call_args
        assert "Constraint: user_id_date_api_key_model_custom_llm_provider_mcp_namespaced_tool_name_endpoint" in call_args
        assert "Batch size: 1" in call_args
        assert "Unique constraint violation" in call_args


@pytest.mark.asyncio
async def test_update_daily_spend_re_raises_exception_after_logging():
    """
    Test that when batch upsert fails, the exception is properly re-raised after logging.
    This ensures that error handling continues to work correctly upstream.
    """
    # Setup
    mock_prisma_client = MagicMock()
    mock_batcher = MagicMock()
    mock_table = MagicMock()
    mock_batch_context = MagicMock()
    mock_batch_context.__aenter__ = AsyncMock(return_value=mock_batcher)
    mock_batcher.litellm_dailyuserspend = mock_table
    
    # Create a transaction
    daily_spend_transactions = {
        "test_key": {
            "user_id": "test-user",
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
    
    # Create a custom exception to verify it's re-raised
    custom_exception = ValueError("Database connection lost")
    mock_batch_context.__aexit__ = AsyncMock(side_effect=custom_exception)
    mock_prisma_client.db.batch_.return_value = mock_batch_context
    
    # Create a mock proxy_logging_obj with failure_handler as AsyncMock
    mock_proxy_logging = MagicMock()
    mock_proxy_logging.failure_handler = AsyncMock()
    
    # Verify the exception is re-raised
    with pytest.raises(ValueError, match="Database connection lost"):
        await DBSpendUpdateWriter._update_daily_spend(
            n_retry_times=0,  # No retries to make test faster
            prisma_client=mock_prisma_client,
            proxy_logging_obj=mock_proxy_logging,
            daily_spend_transactions=daily_spend_transactions,
            entity_type="user",
            entity_id_field="user_id",
            table_name="litellm_dailyuserspend",
            unique_constraint_name="user_id_date_api_key_model_custom_llm_provider_mcp_namespaced_tool_name_endpoint",
        )
