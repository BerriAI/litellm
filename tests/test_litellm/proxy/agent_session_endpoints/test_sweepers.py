"""
Tests for the agent-session sweepers.

Covers:
- Validation #7: max_session_minutes — sessions older than ceiling get terminated
- Validation #9: bootstrap_timeout — sessions stuck in `provisioning` get terminated
- Validation #10: heartbeat_loss — `ready` sessions whose daemon went quiet get terminated
- The optimistic re-fetch lock: sweeper does not double-terminate a session
  whose status changed underneath it.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.agent_session_endpoints.sweepers import (
    SweeperConfig,
    bootstrap_timeout_sweeper,
    heartbeat_timeout_sweeper,
    max_session_minutes_sweeper,
    STATUS_FAILED,
    STATUS_PROVISIONING,
    STATUS_READY,
    STATUS_TERMINATED,
)
from litellm.managed_agents.vms.base import (
    AgentVMProvider,
    ProvisionContext,
    VMHandle,
    VMState,
    VMStatus,
)


class _FakeProvider(AgentVMProvider):
    name = "fake"

    def __init__(self) -> None:
        self.terminate_calls: List[str] = []

    async def provision(self, ctx: ProvisionContext) -> VMHandle:  # pragma: no cover
        raise AssertionError("provision should not be called by sweepers")

    async def terminate(self, vm: VMHandle, **_kwargs) -> None:
        self.terminate_calls.append(vm.vm_id)

    async def status(self, vm: VMHandle, **_kwargs) -> VMStatus:  # pragma: no cover
        return VMStatus(state=VMState.UNKNOWN)


class _Row:
    """Mutable struct for `LiteLLM_AgentSession` row attribute access."""

    def __init__(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


def _build_handle(row: Any) -> VMHandle:
    return VMHandle(vm_id=row.vm_id, provider="fake", region="us-west-2")


def _build_prisma(rows: Dict[str, Any], find_many_results: List[Any]) -> Any:
    """Wire up a fake Prisma client whose store is a {session_id: row} dict."""
    prisma = MagicMock()
    prisma.db = MagicMock()
    table = MagicMock()

    async def find_many(where: Dict[str, Any], **_kwargs):
        return list(find_many_results)

    async def find_unique(where: Dict[str, Any]):
        return rows.get(where["session_id"])

    async def update(where: Dict[str, Any], data: Dict[str, Any]):
        row = rows.get(where["session_id"])
        if row is None:
            raise Exception("row not found")
        for k, v in data.items():
            setattr(row, k, v)
        return row

    table.find_many = AsyncMock(side_effect=find_many)
    table.find_unique = AsyncMock(side_effect=find_unique)
    table.update = AsyncMock(side_effect=update)
    prisma.db.litellm_agentsession = table
    return prisma


@pytest.mark.asyncio
async def test_bootstrap_timeout_sweeper_terminates_stuck_provisioning():
    now = datetime.now(timezone.utc)
    stuck_row = _Row(
        session_id="sess-stuck",
        status=STATUS_PROVISIONING,
        vm_id="i-stuck",
        created_at=now - timedelta(seconds=600),
        last_heartbeat_at=None,
    )
    fresh_row = _Row(
        session_id="sess-fresh",
        status=STATUS_PROVISIONING,
        vm_id="i-fresh",
        created_at=now,
        last_heartbeat_at=None,
    )
    rows = {stuck_row.session_id: stuck_row, fresh_row.session_id: fresh_row}
    # find_many returns only the stuck one (the WHERE filter is the prisma side).
    prisma = _build_prisma(rows, find_many_results=[stuck_row])

    provider = _FakeProvider()
    config = SweeperConfig(bootstrap_timeout_seconds=180)

    swept = await bootstrap_timeout_sweeper(
        provider=provider,
        prisma_client=prisma,
        config=config,
        handle_builder=_build_handle,
    )
    assert swept == 1
    assert provider.terminate_calls == ["i-stuck"]
    assert stuck_row.status == STATUS_FAILED
    assert stuck_row.failure_reason == "bootstrap_timeout"
    # Untouched session stays in provisioning.
    assert fresh_row.status == STATUS_PROVISIONING


@pytest.mark.asyncio
async def test_heartbeat_timeout_sweeper_terminates_quiet_ready_sessions():
    now = datetime.now(timezone.utc)
    quiet_row = _Row(
        session_id="sess-quiet",
        status=STATUS_READY,
        vm_id="i-quiet",
        created_at=now - timedelta(minutes=10),
        last_heartbeat_at=now - timedelta(seconds=600),
    )
    rows = {quiet_row.session_id: quiet_row}
    prisma = _build_prisma(rows, find_many_results=[quiet_row])

    provider = _FakeProvider()
    config = SweeperConfig(heartbeat_timeout_seconds=120)

    swept = await heartbeat_timeout_sweeper(
        provider=provider,
        prisma_client=prisma,
        config=config,
        handle_builder=_build_handle,
    )
    assert swept == 1
    assert provider.terminate_calls == ["i-quiet"]
    assert quiet_row.status == STATUS_TERMINATED
    assert quiet_row.failure_reason == "heartbeat_timeout"


@pytest.mark.asyncio
async def test_max_session_minutes_sweeper_terminates_long_running_sessions():
    now = datetime.now(timezone.utc)
    old_ready = _Row(
        session_id="sess-old",
        status=STATUS_READY,
        vm_id="i-old",
        created_at=now - timedelta(minutes=200),
        last_heartbeat_at=now,
    )
    rows = {old_ready.session_id: old_ready}
    prisma = _build_prisma(rows, find_many_results=[old_ready])

    provider = _FakeProvider()
    config = SweeperConfig(max_session_minutes=120)

    swept = await max_session_minutes_sweeper(
        provider=provider,
        prisma_client=prisma,
        config=config,
        handle_builder=_build_handle,
    )
    assert swept == 1
    assert provider.terminate_calls == ["i-old"]
    assert old_ready.status == STATUS_TERMINATED
    assert old_ready.failure_reason == "max_session_minutes"


@pytest.mark.asyncio
async def test_sweeper_skips_session_already_terminated():
    """Optimistic re-fetch lock: if a session moved to TERMINATED between
    find_many and our work, the sweeper must NOT re-terminate."""
    now = datetime.now(timezone.utc)
    candidate = _Row(
        session_id="sess-raced",
        status=STATUS_TERMINATED,  # already moved underneath us
        vm_id="i-already-done",
        created_at=now - timedelta(minutes=200),
        last_heartbeat_at=None,
    )
    rows = {candidate.session_id: candidate}
    prisma = _build_prisma(rows, find_many_results=[candidate])

    provider = _FakeProvider()
    config = SweeperConfig(max_session_minutes=120)

    await max_session_minutes_sweeper(
        provider=provider,
        prisma_client=prisma,
        config=config,
        handle_builder=_build_handle,
    )
    # Provider must not be called.
    assert provider.terminate_calls == []


@pytest.mark.asyncio
async def test_sweeper_with_no_vm_id_just_marks_row():
    """If a session never got a VM (e.g. RunInstances failed mid-provision),
    the sweeper still moves the row to FAILED."""

    def _maybe_handle(row: Any) -> Optional[VMHandle]:
        # Build a handle with empty vm_id — sweeper should skip terminate.
        return VMHandle(vm_id="", provider="fake", region="us-west-2")

    now = datetime.now(timezone.utc)
    row = _Row(
        session_id="sess-novm",
        status=STATUS_PROVISIONING,
        vm_id="",
        created_at=now - timedelta(seconds=600),
        last_heartbeat_at=None,
    )
    rows = {row.session_id: row}
    prisma = _build_prisma(rows, find_many_results=[row])

    provider = _FakeProvider()
    config = SweeperConfig(bootstrap_timeout_seconds=180)
    await bootstrap_timeout_sweeper(
        provider=provider,
        prisma_client=prisma,
        config=config,
        handle_builder=_maybe_handle,
    )
    assert provider.terminate_calls == []
    assert row.status == STATUS_FAILED


def test_sweeper_config_from_agent_settings():
    cfg = SweeperConfig.from_agent_settings(
        {
            "sweep_interval_seconds": 17,
            "ec2": {
                "bootstrap_timeout_seconds": 200,
                "heartbeat_timeout_seconds": 99,
                "max_session_minutes": 60,
            },
        }
    )
    assert cfg.sweep_interval_seconds == 17
    assert cfg.bootstrap_timeout_seconds == 200
    assert cfg.heartbeat_timeout_seconds == 99
    assert cfg.max_session_minutes == 60


def test_sweeper_config_defaults_when_missing():
    cfg = SweeperConfig.from_agent_settings(None)
    assert cfg.sweep_interval_seconds == 30
    assert cfg.bootstrap_timeout_seconds == 180
    assert cfg.heartbeat_timeout_seconds == 120
    assert cfg.max_session_minutes == 120
