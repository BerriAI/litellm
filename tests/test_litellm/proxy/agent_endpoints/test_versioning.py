"""Tests for agent version history + rollback (S3-03 + S3-04)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy.agent_endpoints.versioning import (
    list_agent_versions,
    rollback_agent_to_version,
    snapshot_existing_agent,
)


def _mock_prisma():
    prisma = MagicMock()
    prisma.db.litellm_agentversiontable.find_many = AsyncMock(return_value=[])
    prisma.db.litellm_agentversiontable.create = AsyncMock()
    prisma.db.litellm_agentversiontable.find_unique = AsyncMock()
    prisma.db.litellm_agentstable.find_unique = AsyncMock()
    prisma.db.litellm_agentstable.update = AsyncMock()
    return prisma


@pytest.mark.asyncio
async def test_snapshot_existing_agent_writes_first_version():
    prisma = _mock_prisma()
    # No prior versions -> version_number becomes 1.
    prisma.db.litellm_agentversiontable.find_many.return_value = []
    version_num = await snapshot_existing_agent(
        prisma_client=prisma,
        existing_row={
            "agent_id": "a-1",
            "agent_card_params": {"name": "agent"},
            "litellm_params": {"model": "gpt-4"},
            "static_headers": {"x": "y"},
        },
        created_by="u-1",
    )
    assert version_num == 1
    create_data = prisma.db.litellm_agentversiontable.create.call_args.kwargs["data"]
    assert create_data["agent_id"] == "a-1"
    assert create_data["version_number"] == 1
    assert create_data["is_rollback"] is False
    assert create_data["created_by"] == "u-1"


@pytest.mark.asyncio
async def test_snapshot_increments_version_number():
    prisma = _mock_prisma()
    prisma.db.litellm_agentversiontable.find_many.return_value = [
        MagicMock(version_number=5)
    ]
    version_num = await snapshot_existing_agent(
        prisma_client=prisma,
        existing_row={
            "agent_id": "a-1",
            "agent_card_params": {},
            "litellm_params": None,
            "static_headers": None,
        },
        created_by="u-1",
    )
    assert version_num == 6


@pytest.mark.asyncio
async def test_list_versions_orders_newest_first():
    prisma = _mock_prisma()
    rows = [MagicMock(version_number=n) for n in (3, 2, 1)]
    for r, n in zip(rows, (3, 2, 1)):
        r.model_dump = MagicMock(return_value={"version_number": n})
    prisma.db.litellm_agentversiontable.find_many.return_value = rows
    result = await list_agent_versions(prisma_client=prisma, agent_id="a-1", limit=10)
    args = prisma.db.litellm_agentversiontable.find_many.call_args.kwargs
    assert args["order"] == {"version_number": "desc"}
    assert args["take"] == 10
    assert [r["version_number"] for r in result] == [3, 2, 1]


@pytest.mark.asyncio
async def test_rollback_snapshots_current_then_restores_target_then_marks_rollback():
    prisma = _mock_prisma()
    target_version = MagicMock(
        agent_card_params={"name": "old"},
        litellm_params={"model": "old"},
        static_headers={"a": "b"},
    )
    prisma.db.litellm_agentversiontable.find_unique.return_value = target_version
    prisma.db.litellm_agentstable.find_unique.return_value = MagicMock(
        agent_id="a-1",
        agent_card_params={"name": "current"},
        litellm_params={"model": "current"},
        static_headers={},
    )
    prisma.db.litellm_agentversiontable.find_many.return_value = [
        MagicMock(version_number=7)  # latest before rollback
    ]
    prisma.db.litellm_agentstable.update.return_value = MagicMock(agent_id="a-1")

    await rollback_agent_to_version(
        prisma_client=prisma,
        agent_id="a-1",
        target_version_number=3,
        created_by="u-1",
    )

    # find_unique for target version
    prisma.db.litellm_agentversiontable.find_unique.assert_awaited_once_with(
        where={"agent_id_version_number": {"agent_id": "a-1", "version_number": 3}}
    )
    # Two creates: pre-snapshot of current + rollback marker.
    create_calls = prisma.db.litellm_agentversiontable.create.await_args_list
    assert len(create_calls) == 2
    marker = create_calls[1].kwargs["data"]
    assert marker["is_rollback"] is True
    assert marker["rolled_back_from"] == 3
    # The live row is updated with the target version's content.
    update = prisma.db.litellm_agentstable.update.await_args.kwargs["data"]
    assert update["agent_card_params"] == {"name": "old"}
    assert update["litellm_params"] == {"model": "old"}


@pytest.mark.asyncio
async def test_rollback_404s_when_target_version_missing():
    from fastapi import HTTPException

    prisma = _mock_prisma()
    prisma.db.litellm_agentversiontable.find_unique.return_value = None
    with pytest.raises(HTTPException) as exc:
        await rollback_agent_to_version(
            prisma_client=prisma,
            agent_id="a-1",
            target_version_number=999,
            created_by="u-1",
        )
    assert exc.value.status_code == 404
