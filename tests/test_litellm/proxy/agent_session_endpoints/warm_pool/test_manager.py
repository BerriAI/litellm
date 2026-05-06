"""Unit tests for `litellm/proxy/agent_session_endpoints/warm_pool/manager.py`.

Cover the maintenance loop logic without hitting AWS or the database:
* refill spawns the deficit when fewer warm VMs exist than configured
* idle reaper terminates `state='warm'` rows older than max_idle_minutes
* shrink terminates the oldest excess when the pool size shrinks
* tick is a no-op when no team has the feature enabled

Real-cloud counterparts live in `test_warm_pool_real.py` (slow-marked).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, patch

import pytest

from litellm.proxy.agent_session_endpoints.warm_pool.manager import WarmPoolManager


# ---------------------------------------------------------------------------
# Tiny stand-in for the prisma_client used by manager.py
# ---------------------------------------------------------------------------


class _Row:
    def __init__(self, **fields: Any) -> None:
        for k, v in fields.items():
            setattr(self, k, v)


class _AgentVMTable:
    def __init__(self) -> None:
        self.rows: List[_Row] = []

    async def find_many(
        self,
        where: Optional[Dict[str, Any]] = None,
        order: Any = None,
        take: Optional[int] = None,
    ) -> List[_Row]:
        out: List[_Row] = []
        for r in self.rows:
            ok = True
            for k, expected in (where or {}).items():
                actual = getattr(r, k, None)
                if isinstance(expected, dict) and "in" in expected:
                    if actual not in expected["in"]:
                        ok = False
                        break
                elif isinstance(expected, dict) and "lt" in expected:
                    if actual is None or not actual < expected["lt"]:
                        ok = False
                        break
                else:
                    if actual != expected:
                        ok = False
                        break
            if ok:
                out.append(r)
        return out

    async def create(self, data: Dict[str, Any]) -> _Row:
        row = _Row(**data)
        self.rows.append(row)
        return row

    async def update(self, where: Dict[str, Any], data: Dict[str, Any]) -> _Row:
        for r in self.rows:
            if getattr(r, "id", None) == where.get("id"):
                for k, v in data.items():
                    setattr(r, k, v)
                return r
        raise RuntimeError(f"no row for {where}")

    async def update_many(self, where: Dict[str, Any], data: Dict[str, Any]):
        count = 0
        for r in self.rows:
            ok = all(getattr(r, k, None) == v for k, v in where.items())
            if ok:
                for k, v in data.items():
                    setattr(r, k, v)
                count += 1
        return _Row(count=count)

    async def delete(self, where: Dict[str, Any]) -> _Row:
        for i, r in enumerate(self.rows):
            if getattr(r, "id", None) == where.get("id"):
                return self.rows.pop(i)
        raise RuntimeError(f"no row for {where}")


class _ConfigTable:
    def __init__(self, rows: List[_Row]) -> None:
        self.rows = rows

    async def find_many(self, where: Optional[Dict[str, Any]] = None) -> List[_Row]:
        if where is None:
            return list(self.rows)
        out = []
        for r in self.rows:
            ok = True
            for k, expected in where.items():
                actual = getattr(r, k, None)
                if isinstance(expected, dict) and "gt" in expected:
                    if actual is None or not actual > expected["gt"]:
                        ok = False
                        break
                else:
                    if actual != expected:
                        ok = False
                        break
            if ok:
                out.append(r)
        return out

    async def find_unique(self, where: Dict[str, Any]) -> Optional[_Row]:
        for r in self.rows:
            if getattr(r, "team_id", None) == where.get("team_id"):
                return r
        return None


class _FakeDB:
    def __init__(self, configs: List[_Row]) -> None:
        self.litellm_agentvm = _AgentVMTable()
        self.litellm_agentvmconfig = _ConfigTable(configs)


class _FakePrisma:
    def __init__(self, configs: List[_Row]) -> None:
        self.db = _FakeDB(configs)


def _config_row(team_id: str, size: int, idle_min: int = 30) -> _Row:
    return _Row(
        team_id=team_id,
        warm_pool_enabled=True,
        warm_pool_size=size,
        max_idle_minutes=idle_min,
        aws_region="us-west-2",
    )


@pytest.fixture
def fake_prisma() -> _FakePrisma:
    return _FakePrisma(configs=[])


@pytest.fixture
def manager(fake_prisma):
    return WarmPoolManager(prisma_getter=lambda: fake_prisma)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tick_noop_when_no_teams_enabled(fake_prisma, manager):
    """No `LiteLLM_AgentVMConfig` rows -> tick does nothing."""
    await manager.tick()
    assert fake_prisma.db.litellm_agentvm.rows == []


@pytest.mark.asyncio
async def test_refill_spawns_deficit(fake_prisma, manager):
    """Pool size 3, 0 warm -> spawn 3 placeholders."""
    fake_prisma.db.litellm_agentvmconfig.rows = [_config_row("team-1", size=3)]

    # Mock _spawn_one so we don't try to call boto3. Verify it's called 3 times.
    spawn_calls: List[str] = []

    async def fake_spawn(prisma_client, vm_config):
        spawn_calls.append(vm_config.team_id)

    with patch.object(manager, "_spawn_one", new=fake_spawn):
        await manager.tick()

    assert len(spawn_calls) == 3
    assert all(t == "team-1" for t in spawn_calls)


@pytest.mark.asyncio
async def test_refill_caps_at_max_concurrent(fake_prisma, manager):
    """Pool size 100, 0 warm -> spawn at most MAX_CONCURRENT_SPAWNS_PER_TEAM."""
    from litellm.proxy.agent_session_endpoints.warm_pool.manager import (
        MAX_CONCURRENT_SPAWNS_PER_TEAM,
    )

    fake_prisma.db.litellm_agentvmconfig.rows = [_config_row("team-1", size=100)]
    spawn_calls: List[str] = []

    async def fake_spawn(prisma_client, vm_config):
        spawn_calls.append(vm_config.team_id)

    with patch.object(manager, "_spawn_one", new=fake_spawn):
        await manager.tick()

    assert len(spawn_calls) == MAX_CONCURRENT_SPAWNS_PER_TEAM


@pytest.mark.asyncio
async def test_refill_no_spawn_when_pool_full(fake_prisma, manager):
    """Pool size 2, already 2 warm -> 0 spawns."""
    fake_prisma.db.litellm_agentvmconfig.rows = [_config_row("team-1", size=2)]
    now = datetime.now(timezone.utc)
    fake_prisma.db.litellm_agentvm.rows = [
        _Row(id="i-1", team_id="team-1", state="warm", warmed_at=now, provider="ec2"),
        _Row(id="i-2", team_id="team-1", state="warm", warmed_at=now, provider="ec2"),
    ]
    spawn_calls: List[str] = []

    async def fake_spawn(prisma_client, vm_config):
        spawn_calls.append(vm_config.team_id)

    with patch.object(manager, "_spawn_one", new=fake_spawn):
        await manager.tick()

    assert spawn_calls == []


@pytest.mark.asyncio
async def test_reap_stale_warm(fake_prisma, manager):
    """Warm VM older than max_idle_minutes is moved to terminated."""
    fake_prisma.db.litellm_agentvmconfig.rows = [
        _config_row("team-1", size=1, idle_min=30)
    ]
    stale_time = datetime.now(timezone.utc) - timedelta(minutes=60)
    fake_prisma.db.litellm_agentvm.rows = [
        _Row(
            id="i-stale",
            team_id="team-1",
            state="warm",
            warmed_at=stale_time,
            provider="ec2",
            region="us-west-2",
            metadata={},
        )
    ]
    # Stub provider termination — we don't want a boto3 call.
    with (
        patch.object(manager, "_terminate_vm", new=AsyncMock()) as mock_term,
        patch.object(manager, "_spawn_one", new=AsyncMock()),
    ):
        await manager.tick()

    assert mock_term.await_count >= 1
    terminated = mock_term.await_args_list[0].args[1]
    assert terminated.id == "i-stale"


@pytest.mark.asyncio
async def test_shrink_to_desired_terminates_excess(fake_prisma, manager):
    """Pool shrunk from 5 to 2 -> 3 oldest warm VMs are terminated."""
    fake_prisma.db.litellm_agentvmconfig.rows = [_config_row("team-1", size=2)]
    now = datetime.now(timezone.utc)
    # Insert 4 warm rows with staggered warmed_at timestamps. Oldest are "i-1", "i-2".
    fake_prisma.db.litellm_agentvm.rows = [
        _Row(
            id=f"i-{n}",
            team_id="team-1",
            state="warm",
            warmed_at=now - timedelta(minutes=10 - n),  # i-1 oldest, i-4 newest
            provider="ec2",
            region="us-west-2",
            metadata={},
        )
        for n in range(1, 5)
    ]

    terminated_ids: List[str] = []

    async def fake_terminate(prisma_client, row):
        terminated_ids.append(row.id)

    with (
        patch.object(manager, "_terminate_vm", new=fake_terminate),
        patch.object(manager, "_spawn_one", new=AsyncMock()),
    ):
        await manager.tick()

    # Two oldest get terminated.
    assert "i-1" in terminated_ids
    assert "i-2" in terminated_ids
    assert "i-4" not in terminated_ids


@pytest.mark.asyncio
async def test_disabled_team_skipped(fake_prisma, manager):
    """`warm_pool_enabled=false` rows are not returned by `_enabled_teams`."""
    fake_prisma.db.litellm_agentvmconfig.rows = [
        _Row(
            team_id="team-1",
            warm_pool_enabled=False,
            warm_pool_size=5,
            max_idle_minutes=30,
            aws_region="us-west-2",
        ),
    ]
    spawn_calls: List[str] = []

    async def fake_spawn(prisma_client, vm_config):
        spawn_calls.append(vm_config.team_id)

    with patch.object(manager, "_spawn_one", new=fake_spawn):
        await manager.tick()

    assert spawn_calls == []


@pytest.mark.asyncio
async def test_size_zero_team_skipped(fake_prisma, manager):
    """`warm_pool_size=0` rows are filtered out of `_enabled_teams`."""
    fake_prisma.db.litellm_agentvmconfig.rows = [
        _Row(
            team_id="team-1",
            warm_pool_enabled=True,
            warm_pool_size=0,
            max_idle_minutes=30,
            aws_region="us-west-2",
        ),
    ]
    spawn_calls: List[str] = []

    async def fake_spawn(prisma_client, vm_config):
        spawn_calls.append(vm_config.team_id)

    with patch.object(manager, "_spawn_one", new=fake_spawn):
        await manager.tick()

    assert spawn_calls == []
