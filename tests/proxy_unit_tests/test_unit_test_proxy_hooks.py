import asyncio
import os
import sys
from unittest.mock import Mock, patch, AsyncMock
import pytest
from fastapi import Request
from litellm.proxy.utils import _get_redoc_url, _get_docs_url
from datetime import datetime

sys.path.insert(0, os.path.abspath("../.."))
import litellm


@pytest.mark.asyncio
async def test_disable_spend_logs():
    """
    Test that the spend logs are not written to the database when disable_spend_logs is True
    """
    # Mock the necessary components
    import asyncio

    mock_prisma_client = Mock()
    mock_prisma_client.spend_log_transactions = []
    # Add lock for spend_log_transactions (matches real PrismaClient)
    mock_prisma_client._spend_log_transactions_lock = asyncio.Lock()

    with (
        patch("litellm.proxy.proxy_server.disable_spend_logs", True),
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client),
    ):
        from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter

        db_spend_update_writer = DBSpendUpdateWriter()

        # Call update_database with disable_spend_logs=True
        await db_spend_update_writer.update_database(
            token="fake-token",
            response_cost=0.1,
            user_id="user123",
            completion_response=None,
            start_time=datetime.now(),
            end_time=datetime.now(),
            end_user_id="end_user_id",
            team_id="team_id",
            org_id="org_id",
            kwargs={},
        )
        # Verify no spend logs were added
        assert len(mock_prisma_client.spend_log_transactions) == 0


@pytest.mark.asyncio
async def test_disable_entity_and_daily_skips_batch_task():
    """
    When both entity and daily spend updates are fully disabled, the batch
    task is not scheduled (SpendLogs still written).
    """
    mock_prisma_client = Mock()
    mock_prisma_client.spend_log_transactions = []
    mock_prisma_client._spend_log_transactions_lock = asyncio.Lock()

    with (
        patch("litellm.proxy.proxy_server.disable_spend_logs", False),
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client),
        patch(
            "litellm.proxy.proxy_server.general_settings",
            {
                "disable_entity_spend_updates": True,
                "disable_daily_spend_updates": True,
            },
        ),
        patch(
            "litellm.proxy.db.db_spend_update_writer.DBSpendUpdateWriter._batch_database_updates",
            new_callable=AsyncMock,
        ) as mock_batch,
    ):
        from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter

        db_spend_update_writer = DBSpendUpdateWriter()

        await db_spend_update_writer.update_database(
            token="sk-test",
            response_cost=0.1,
            user_id="user123",
            completion_response=None,
            start_time=datetime.now(),
            end_time=datetime.now(),
            end_user_id="end_user_id",
            team_id="team_id",
            org_id="org_id",
            kwargs={},
        )
        await asyncio.sleep(0.1)
        assert not mock_batch.called
        assert len(mock_prisma_client.spend_log_transactions) == 1


@pytest.mark.asyncio
async def test_disable_entity_keeps_daily_batch_scheduled():
    """
    disable_entity_spend_updates alone still schedules the batch so daily
    tables (e.g. DailyTagSpend for warehouse CDC) keep updating.
    """
    mock_prisma_client = Mock()
    mock_prisma_client.spend_log_transactions = []
    mock_prisma_client._spend_log_transactions_lock = asyncio.Lock()

    with (
        patch("litellm.proxy.proxy_server.disable_spend_logs", False),
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client),
        patch(
            "litellm.proxy.proxy_server.general_settings",
            {"disable_entity_spend_updates": True},
        ),
        patch(
            "litellm.proxy.db.db_spend_update_writer.DBSpendUpdateWriter._batch_database_updates",
            new_callable=AsyncMock,
        ) as mock_batch,
    ):
        from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter

        db_spend_update_writer = DBSpendUpdateWriter()

        await db_spend_update_writer.update_database(
            token="sk-test",
            response_cost=0.1,
            user_id="user123",
            completion_response=None,
            start_time=datetime.now(),
            end_time=datetime.now(),
            end_user_id="end_user_id",
            team_id="team_id",
            org_id="org_id",
            kwargs={},
        )
        await asyncio.sleep(0.1)
        assert mock_batch.called
        assert len(mock_prisma_client.spend_log_transactions) == 1


@pytest.mark.asyncio
async def test_batch_skips_entity_helpers_keeps_daily_tag():
    """
    Inside the batch: entity helpers skipped, daily tag helper still runs.
    """
    mock_prisma_client = Mock()

    with patch(
        "litellm.proxy.proxy_server.general_settings",
        {"disable_entity_spend_updates": True},
    ):
        from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter

        writer = DBSpendUpdateWriter()
        writer._update_user_db = AsyncMock()
        writer._update_key_db = AsyncMock()
        writer._update_team_db = AsyncMock()
        writer._update_org_db = AsyncMock()
        writer._update_tag_db = AsyncMock()
        writer._update_agent_db = AsyncMock()
        writer.add_spend_log_transaction_to_daily_user_transaction = AsyncMock()
        writer.add_spend_log_transaction_to_daily_end_user_transaction = AsyncMock()
        writer.add_spend_log_transaction_to_daily_agent_transaction = AsyncMock()
        writer.add_spend_log_transaction_to_daily_team_transaction = AsyncMock()
        writer.add_spend_log_transaction_to_daily_org_transaction = AsyncMock()
        writer.add_spend_log_transaction_to_daily_tag_transaction = AsyncMock()

        await writer._batch_database_updates(
            response_cost=0.1,
            user_id="user123",
            hashed_token="hashed",
            team_id="team",
            org_id="org",
            end_user_id="end",
            prisma_client=mock_prisma_client,
            litellm_proxy_budget_name=None,
            payload={"request_tags": [], "agent_id": None},
        )

        writer._update_user_db.assert_not_called()
        writer._update_key_db.assert_not_called()
        writer._update_team_db.assert_not_called()
        writer._update_org_db.assert_not_called()
        writer._update_tag_db.assert_not_called()
        writer._update_agent_db.assert_not_called()

        writer.add_spend_log_transaction_to_daily_tag_transaction.assert_awaited_once()
        writer.add_spend_log_transaction_to_daily_user_transaction.assert_awaited_once()


@pytest.mark.asyncio
async def test_list_config_disables_only_named_entity_and_daily_types():
    """List form: only named types are skipped."""
    mock_prisma_client = Mock()

    with patch(
        "litellm.proxy.proxy_server.general_settings",
        {
            "disable_entity_spend_updates": ["user", "key"],
            "disable_daily_spend_updates": ["user", "team"],
        },
    ):
        from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter

        writer = DBSpendUpdateWriter()
        writer._update_user_db = AsyncMock()
        writer._update_key_db = AsyncMock()
        writer._update_team_db = AsyncMock()
        writer._update_org_db = AsyncMock()
        writer._update_tag_db = AsyncMock()
        writer._update_agent_db = AsyncMock()
        writer.add_spend_log_transaction_to_daily_user_transaction = AsyncMock()
        writer.add_spend_log_transaction_to_daily_end_user_transaction = AsyncMock()
        writer.add_spend_log_transaction_to_daily_agent_transaction = AsyncMock()
        writer.add_spend_log_transaction_to_daily_team_transaction = AsyncMock()
        writer.add_spend_log_transaction_to_daily_org_transaction = AsyncMock()
        writer.add_spend_log_transaction_to_daily_tag_transaction = AsyncMock()

        await writer._batch_database_updates(
            response_cost=0.1,
            user_id="user123",
            hashed_token="hashed",
            team_id="team",
            org_id="org",
            end_user_id="end",
            prisma_client=mock_prisma_client,
            litellm_proxy_budget_name=None,
            payload={"request_tags": [], "agent_id": None},
        )

        writer._update_user_db.assert_not_called()
        writer._update_key_db.assert_not_called()
        writer._update_team_db.assert_awaited_once()
        writer.add_spend_log_transaction_to_daily_user_transaction.assert_not_called()
        writer.add_spend_log_transaction_to_daily_team_transaction.assert_not_called()
        writer.add_spend_log_transaction_to_daily_tag_transaction.assert_awaited_once()


@pytest.mark.asyncio
async def test_disable_entity_spend_updates_false_still_runs_batch():
    """
    Test that _batch_database_updates IS scheduled when flags are absent.
    """
    mock_prisma_client = Mock()
    mock_prisma_client.spend_log_transactions = []
    mock_prisma_client._spend_log_transactions_lock = asyncio.Lock()

    with (
        patch("litellm.proxy.proxy_server.disable_spend_logs", False),
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client),
        patch(
            "litellm.proxy.proxy_server.general_settings",
            {},  # flag absent — default behavior
        ),
        patch(
            "litellm.proxy.db.db_spend_update_writer.DBSpendUpdateWriter._batch_database_updates",
            new_callable=AsyncMock,
        ) as mock_batch,
    ):
        from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter

        db_spend_update_writer = DBSpendUpdateWriter()

        await db_spend_update_writer.update_database(
            token="sk-test",
            response_cost=0.1,
            user_id="user123",
            completion_response=None,
            start_time=datetime.now(),
            end_time=datetime.now(),
            end_user_id="end_user_id",
            team_id="team_id",
            org_id="org_id",
            kwargs={},
        )
        await asyncio.sleep(0.1)
        assert mock_batch.called, (
            "_batch_database_updates should have been called by default"
        )
