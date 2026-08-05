from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Tuple

import pytest

from litellm.repositories.unit_of_work import spend_reset_unit_of_work


class FakeBatchTable:
    def __init__(self, table_name: str, calls: List[Tuple[str, Dict[str, Any], Dict[str, Any]]]):
        self._table_name = table_name
        self._calls = calls

    def update(self, where: Mapping[str, object], data: Mapping[str, object]) -> None:
        self._calls.append((self._table_name, dict(where), dict(data)))


class FakeBatch:
    def __init__(self):
        self.calls: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
        self.commit_count = 0
        self.litellm_verificationtoken = FakeBatchTable("litellm_verificationtoken", self.calls)
        self.litellm_usertable = FakeBatchTable("litellm_usertable", self.calls)
        self.litellm_teamtable = FakeBatchTable("litellm_teamtable", self.calls)

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

    with pytest.raises(RuntimeError, match="boom"):
        async with spend_reset_unit_of_work(lambda: batch) as uow:
            uow.keys.queue_spend_reset(token="tok-1", budget_reset_at=None)
            raise RuntimeError("boom")

    assert batch.commit_count == 0


async def test_empty_block_still_commits_the_batch():
    batch = FakeBatch()

    async with spend_reset_unit_of_work(lambda: batch):
        pass

    assert batch.commit_count == 1
    assert batch.calls == []
