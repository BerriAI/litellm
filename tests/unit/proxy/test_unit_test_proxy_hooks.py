import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pytest

from litellm.proxy.db.db_spend_update_writer import DBSpendUpdateWriter


@pytest.mark.asyncio
async def test_disabled_spend_logs_retain_lean_accounting_charge():
    mock_prisma_client = Mock()
    mock_prisma_client.spend_log_transactions = []
    mock_prisma_client._spend_log_transactions_lock = asyncio.Lock()
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    with (
        patch("litellm.proxy.proxy_server.disable_spend_logs", True),
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma_client),
    ):
        await DBSpendUpdateWriter().update_database(
            token="fake-token",
            response_cost=0.1,
            user_id="user123",
            completion_response=None,
            start_time=now,
            end_time=now,
            end_user_id="end_user_id",
            team_id="team_id",
            org_id="org_id",
            kwargs={},
        )

    assert len(mock_prisma_client.spend_log_transactions) == 1
    row = mock_prisma_client.spend_log_transactions[0]
    assert row["spend"] == 0.1
    assert row["accounting_path"] == "ledger"
    assert (row["messages"], row["response"], row["proxy_server_request"]) == (None, None, None)
    metadata = json.loads(row["metadata"])
    assert metadata["accounting_lean"] is True
    assert metadata["accounting_entities"]["user"] == "user123"
