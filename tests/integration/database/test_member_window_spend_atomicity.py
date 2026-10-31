"""A retried flush must not charge a member again after a sibling write fails."""

import json
import os
import uuid
from datetime import datetime, timezone

import psycopg
import pytest
from psycopg import sql
from prisma.errors import RawQueryError

from integration._support.prisma_clients import connected_proxy_prisma
from litellm.proxy.db.budget_window_spend_writer import commit_window_spend_updates
from litellm.proxy.db.db_transaction_queue.window_spend_update_queue import build_window_spend_transaction


@pytest.mark.asyncio
async def test_failed_key_window_rolls_back_member_spend_before_retry() -> None:
    marker = uuid.uuid4().hex
    member_id = json.dumps(["member-" + marker, "team-" + marker], separators=(",", ":"))
    key_id = "key-" + marker
    constraint = "window_fault_" + marker
    start = datetime(2026, 10, 1, tzinfo=timezone.utc)
    transactions = [
        build_window_spend_transaction("team_member", member_id, "30d", start, 0.06, start),
        build_window_spend_transaction("key", key_id, "30d", start, 0.06, start),
    ]
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as control:
        control.execute(
            sql.SQL('ALTER TABLE "LiteLLM_BudgetWindowSpend" ADD CONSTRAINT {} CHECK (entity_id <> {})').format(
                sql.Identifier(constraint),
                sql.Literal(key_id),
            )
        )
        try:
            async with connected_proxy_prisma() as client:
                with pytest.raises(RawQueryError, match=constraint):
                    await commit_window_spend_updates(client, transactions)
                assert (
                    await client.db.litellm_budgetwindowspend.find_many(
                        where={"entity_id": {"in": [member_id, key_id]}}
                    )
                    == []
                )
        finally:
            control.execute(
                sql.SQL('ALTER TABLE "LiteLLM_BudgetWindowSpend" DROP CONSTRAINT {}').format(sql.Identifier(constraint))
            )
        try:
            async with connected_proxy_prisma() as client:
                await commit_window_spend_updates(client, transactions)
                rows = await client.db.litellm_budgetwindowspend.find_many(
                    where={"entity_id": {"in": [member_id, key_id]}}
                )
                assert len(rows) == 2
                assert all(row.spend == pytest.approx(0.06) for row in rows)
        finally:
            control.execute('DELETE FROM "LiteLLM_BudgetWindowSpend" WHERE entity_id = ANY(%s)', ([member_id, key_id],))
