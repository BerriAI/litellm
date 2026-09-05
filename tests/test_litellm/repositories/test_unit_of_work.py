from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Tuple

import pytest

from litellm.repositories.unit_of_work import (
    budget_cascade_unit_of_work,
    spend_reset_unit_of_work,
)


class FakeBatchTable:
    def __init__(self, table_name: str, calls: List[Tuple[str, Dict[str, Any], Dict[str, Any]]]):
        self._table_name = table_name
        self._calls = calls

    def update(self, where: Mapping[str, object], data: Mapping[str, object]) -> None:
        self._calls.append((self._table_name, dict(where), dict(data)))

    def update_many(self, where: Mapping[str, object], data: Mapping[str, object]) -> None:
        self._calls.append((f"{self._table_name}.update_many", dict(where), dict(data)))


class FakeBatch:
    def __init__(self):
        self.calls: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
        self.commit_count = 0
        self.litellm_verificationtoken = FakeBatchTable("litellm_verificationtoken", self.calls)
        self.litellm_usertable = FakeBatchTable("litellm_usertable", self.calls)
        self.litellm_teamtable = FakeBatchTable("litellm_teamtable", self.calls)
        self.litellm_budgettable = FakeBatchTable("litellm_budgettable", self.calls)
        self.litellm_teammembership = FakeBatchTable("litellm_teammembership", self.calls)
        self.litellm_organizationtable = FakeBatchTable("litellm_organizationtable", self.calls)
        self.litellm_tagtable = FakeBatchTable("litellm_tagtable", self.calls)
        self.litellm_endusertable = FakeBatchTable("litellm_endusertable", self.calls)

    async def commit(self) -> None:
        self.commit_count += 1


async def test_updates_across_tables_share_one_batch_and_commit_once():
    batch = FakeBatch()
    reset_at = datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc)

    async with spend_reset_unit_of_work(lambda: batch) as uow:
        uow.keys.queue_spend_reset(token="tok-1", budget_reset_at=reset_at)
        uow.users.queue_spend_reset(user_id="user-1", budget_reset_at=reset_at)
        uow.teams.queue_spend_reset(team_id="team-1", budget_reset_at=None)
        assert batch.commit_count == 0

    assert batch.commit_count == 1
    assert batch.calls == [
        ("litellm_verificationtoken", {"token": "tok-1"}, {"spend": 0, "budget_reset_at": reset_at}),
        ("litellm_usertable", {"user_id": "user-1"}, {"spend": 0, "budget_reset_at": reset_at}),
        ("litellm_teamtable", {"team_id": "team-1"}, {"spend": 0, "budget_reset_at": None}),
    ]


async def test_raising_inside_block_skips_commit():
    batch = FakeBatch()

    async def _blow_up_mid_transaction():
        async with spend_reset_unit_of_work(lambda: batch) as uow:
            uow.keys.queue_spend_reset(token="tok-1", budget_reset_at=None)
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await _blow_up_mid_transaction()

    assert batch.commit_count == 0


async def test_empty_block_still_commits_the_batch():
    batch = FakeBatch()

    async with spend_reset_unit_of_work(lambda: batch):
        pass

    assert batch.commit_count == 1
    assert batch.calls == []


async def test_budget_cascade_dependents_and_window_advance_share_one_batch():
    batch = FakeBatch()
    reset_at = datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc)
    linked = {"budget_id": {"in": ["budget-1"]}}

    async with budget_cascade_unit_of_work(lambda: batch) as uow:
        uow.team_memberships.queue_spend_zero(where=linked)
        uow.keys.queue_spend_zero(where=linked)
        uow.organizations.queue_spend_zero(where=linked)
        uow.tags.queue_spend_zero(where=linked)
        uow.endusers.queue_spend_zero(where={"user_id": {"in": ["enduser-1"]}})
        uow.budgets.queue_window_advance(budget_id="budget-1", budget_reset_at=reset_at)
        assert batch.commit_count == 0

    assert batch.commit_count == 1
    assert batch.calls == [
        ("litellm_teammembership.update_many", linked, {"spend": 0}),
        ("litellm_verificationtoken.update_many", linked, {"spend": 0}),
        ("litellm_organizationtable.update_many", linked, {"spend": 0}),
        ("litellm_tagtable.update_many", linked, {"spend": 0}),
        ("litellm_endusertable.update_many", {"user_id": {"in": ["enduser-1"]}}, {"spend": 0}),
        ("litellm_budgettable.update_many", {"budget_id": "budget-1"}, {"budget_reset_at": reset_at}),
    ]


async def test_budget_window_advance_tolerates_a_tier_deleted_mid_chunk():
    """A tier deleted between the read and the commit must not abort the batch:
    ``update`` raises P2025 on a missing row and takes every other write in the
    chunk down with it, while ``update_many`` just matches nothing."""
    batch = FakeBatch()

    async with budget_cascade_unit_of_work(lambda: batch) as uow:
        uow.budgets.queue_window_advance(budget_id="budget-1", budget_reset_at=datetime.now(timezone.utc))

    assert [call[0] for call in batch.calls] == ["litellm_budgettable.update_many"]


async def test_budget_cascade_raising_inside_block_skips_commit():
    """A failure part-way through must leave budget_reset_at where it was, so
    the tier is still due on the next tick."""
    batch = FakeBatch()

    async def _blow_up_mid_transaction():
        async with budget_cascade_unit_of_work(lambda: batch) as uow:
            uow.team_memberships.queue_spend_zero(where={"budget_id": {"in": ["budget-1"]}})
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await _blow_up_mid_transaction()

    assert batch.commit_count == 0
