"""Smoke tests for managed_agents reconciler.

Mocks Prisma + boto3. Verifies orphan stop_task_sync called for each scenario:
  - task w/ no DB row              → stopped
  - task w/ row.status = 'dead'    → stopped
  - task w/ row.status = 'stopped' → stopped
  - task w/ row.status = 'failed'  → stopped
  - task w/ row.status = 'creating' young   → skipped
  - task w/ row.status = 'creating' stale   → stopped + row marked failed
  - task w/ row.status = 'ready'   → skipped
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from litellm.proxy.managed_agents_endpoints.fargate.tasks import TAG_SESSION_ID
from litellm.proxy.managed_agents_endpoints.lifecycle import reconcile_orphans


def _make_session_row(session_id: str, status: str, created_at: datetime):
    return SimpleNamespace(
        session_id=session_id,
        status=status,
        created_at=created_at,
        task_arn=f"arn:aws:ecs:us-west-2:123:task/{session_id}",
    )


def _fake_prisma(rows):
    by_id = {r.session_id: r for r in rows}

    async def find_many(where):
        ids = where["session_id"]["in"]
        return [by_id[i] for i in ids if i in by_id]

    update = AsyncMock()
    table = MagicMock()
    table.find_many = AsyncMock(side_effect=find_many)
    table.update = update
    db = MagicMock()
    db.litellm_managedagentsessiontable = table
    client = MagicMock()
    client.db = db
    return client, update


def _fake_tasks(*session_ids):
    return [
        {
            "taskArn": f"arn:aws:ecs:us-west-2:123:task/{sid}",
            "tags": {TAG_SESSION_ID: sid},
            "lastStatus": "RUNNING",
        }
        for sid in session_ids
    ]


@pytest.mark.asyncio
async def test_orphan_no_db_row_stopped():
    tasks = _fake_tasks("s_orphan")
    arns = [t["taskArn"] for t in tasks]

    prisma, update = _fake_prisma([])

    with (
        patch(
            "litellm.proxy.managed_agents_endpoints.lifecycle.list_tagged_task_arns",
            return_value=arns,
        ),
        patch(
            "litellm.proxy.managed_agents_endpoints.lifecycle.describe_tasks_with_tags",
            return_value=tasks,
        ),
        patch(
            "litellm.proxy.managed_agents_endpoints.lifecycle.stop_task_sync"
        ) as stop_mock,
    ):
        stats = await reconcile_orphans(
            prisma_client=prisma, region="us-west-2", cluster="test"
        )

    assert stats == {"scanned": 1, "orphaned_stopped": 1, "stale_creating_stopped": 0}
    stop_mock.assert_called_once()
    assert "missing_db_row" in stop_mock.call_args.args[3]
    update.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["dead", "failed", "stopped"])
async def test_orphan_dead_row_stopped(status):
    rows = [_make_session_row("s1", status, datetime.now(timezone.utc))]
    tasks = _fake_tasks("s1")
    arns = [t["taskArn"] for t in tasks]

    prisma, update = _fake_prisma(rows)

    with (
        patch(
            "litellm.proxy.managed_agents_endpoints.lifecycle.list_tagged_task_arns",
            return_value=arns,
        ),
        patch(
            "litellm.proxy.managed_agents_endpoints.lifecycle.describe_tasks_with_tags",
            return_value=tasks,
        ),
        patch(
            "litellm.proxy.managed_agents_endpoints.lifecycle.stop_task_sync"
        ) as stop_mock,
    ):
        stats = await reconcile_orphans(
            prisma_client=prisma, region="us-west-2", cluster="test"
        )

    assert stats["orphaned_stopped"] == 1
    assert stats["stale_creating_stopped"] == 0
    stop_mock.assert_called_once()
    update.assert_not_called()


@pytest.mark.asyncio
async def test_creating_young_skipped():
    rows = [
        _make_session_row(
            "s_young", "creating", datetime.now(timezone.utc) - timedelta(seconds=30)
        )
    ]
    tasks = _fake_tasks("s_young")
    arns = [t["taskArn"] for t in tasks]

    prisma, update = _fake_prisma(rows)

    with (
        patch(
            "litellm.proxy.managed_agents_endpoints.lifecycle.list_tagged_task_arns",
            return_value=arns,
        ),
        patch(
            "litellm.proxy.managed_agents_endpoints.lifecycle.describe_tasks_with_tags",
            return_value=tasks,
        ),
        patch(
            "litellm.proxy.managed_agents_endpoints.lifecycle.stop_task_sync"
        ) as stop_mock,
    ):
        stats = await reconcile_orphans(
            prisma_client=prisma, region="us-west-2", cluster="test"
        )

    assert stats == {"scanned": 1, "orphaned_stopped": 0, "stale_creating_stopped": 0}
    stop_mock.assert_not_called()
    update.assert_not_called()


@pytest.mark.asyncio
async def test_creating_stale_stopped_and_marked_failed():
    rows = [
        _make_session_row(
            "s_stale", "creating", datetime.now(timezone.utc) - timedelta(hours=1)
        )
    ]
    tasks = _fake_tasks("s_stale")
    arns = [t["taskArn"] for t in tasks]

    prisma, update = _fake_prisma(rows)

    with (
        patch(
            "litellm.proxy.managed_agents_endpoints.lifecycle.list_tagged_task_arns",
            return_value=arns,
        ),
        patch(
            "litellm.proxy.managed_agents_endpoints.lifecycle.describe_tasks_with_tags",
            return_value=tasks,
        ),
        patch(
            "litellm.proxy.managed_agents_endpoints.lifecycle.stop_task_sync"
        ) as stop_mock,
    ):
        stats = await reconcile_orphans(
            prisma_client=prisma, region="us-west-2", cluster="test"
        )

    assert stats["stale_creating_stopped"] == 1
    assert stats["orphaned_stopped"] == 0
    stop_mock.assert_called_once()
    update.assert_called_once()
    update_kwargs = update.call_args.kwargs
    assert update_kwargs["where"] == {"session_id": "s_stale"}
    assert update_kwargs["data"]["status"] == "failed"
    assert "spawn timeout" in update_kwargs["data"]["failure_reason"]


@pytest.mark.asyncio
async def test_ready_row_skipped():
    rows = [_make_session_row("s_live", "ready", datetime.now(timezone.utc))]
    tasks = _fake_tasks("s_live")
    arns = [t["taskArn"] for t in tasks]

    prisma, update = _fake_prisma(rows)

    with (
        patch(
            "litellm.proxy.managed_agents_endpoints.lifecycle.list_tagged_task_arns",
            return_value=arns,
        ),
        patch(
            "litellm.proxy.managed_agents_endpoints.lifecycle.describe_tasks_with_tags",
            return_value=tasks,
        ),
        patch(
            "litellm.proxy.managed_agents_endpoints.lifecycle.stop_task_sync"
        ) as stop_mock,
    ):
        stats = await reconcile_orphans(
            prisma_client=prisma, region="us-west-2", cluster="test"
        )

    assert stats == {"scanned": 1, "orphaned_stopped": 0, "stale_creating_stopped": 0}
    stop_mock.assert_not_called()
    update.assert_not_called()


@pytest.mark.asyncio
async def test_no_managed_tasks_returns_zero():
    untagged = [
        {
            "taskArn": "arn:aws:ecs:us-west-2:123:task/other",
            "tags": {},
            "lastStatus": "RUNNING",
        }
    ]
    arns = [t["taskArn"] for t in untagged]

    prisma, _ = _fake_prisma([])

    with (
        patch(
            "litellm.proxy.managed_agents_endpoints.lifecycle.list_tagged_task_arns",
            return_value=arns,
        ),
        patch(
            "litellm.proxy.managed_agents_endpoints.lifecycle.describe_tasks_with_tags",
            return_value=untagged,
        ),
        patch(
            "litellm.proxy.managed_agents_endpoints.lifecycle.stop_task_sync"
        ) as stop_mock,
    ):
        stats = await reconcile_orphans(
            prisma_client=prisma, region="us-west-2", cluster="test"
        )

    assert stats == {"scanned": 0, "orphaned_stopped": 0, "stale_creating_stopped": 0}
    stop_mock.assert_not_called()
