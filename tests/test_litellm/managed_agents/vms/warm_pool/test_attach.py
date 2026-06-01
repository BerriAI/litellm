"""Unit tests for `litellm/managed_agents/vms/warm_pool/attach.py`.

Cover the race-safe attach path:
* the CAS via `update_many(state='warm')` blocks double-attach
* concurrent attempts on the same pool both succeed when there are enough
  warm VMs to go around
* WarmPoolEmptyError is raised when no warm VMs are available
* a transport failure releases the row to `terminating`
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest


os.environ.setdefault("LITELLM_SALT_KEY", "sk-test-salt-key-do-not-use-in-prod")


from litellm.managed_agents.vms.base import AwsCreds, Ec2Config
from litellm.managed_agents.vms.team_config import TeamVMConfig
from litellm.managed_agents.vms.warm_pool.attach import (
    WarmPoolEmptyError,
    attach_warm_vm,
)


# ---------------------------------------------------------------------------
# Tiny stand-in for the prisma client
# ---------------------------------------------------------------------------


class _Row:
    def __init__(self, **fields: Any) -> None:
        for k, v in fields.items():
            setattr(self, k, v)


class _AgentVMTable:
    def __init__(self) -> None:
        self.rows: List[_Row] = []
        self._lock = asyncio.Lock()

    async def find_many(
        self,
        where: Optional[Dict[str, Any]] = None,
        order: Any = None,
        take: Optional[int] = None,
    ) -> List[_Row]:
        out = []
        for r in self.rows:
            ok = all(getattr(r, k, None) == v for k, v in (where or {}).items())
            if ok:
                out.append(r)
        if order:
            entries = order if isinstance(order, list) else [order]
            for entry in reversed(entries):
                for k, direction in entry.items():
                    out.sort(
                        key=lambda r: getattr(r, k, None) or 0,
                        reverse=(direction == "desc"),
                    )
        if take is not None:
            out = out[:take]
        return out

    async def find_unique(self, where: Dict[str, Any]) -> Optional[_Row]:
        for r in self.rows:
            if getattr(r, "id", None) == where.get("id"):
                return r
        return None

    async def update_many(self, where: Dict[str, Any], data: Dict[str, Any]):
        # Use a lock so concurrent CAS calls in tests are deterministic.
        async with self._lock:
            count = 0
            for r in self.rows:
                if all(getattr(r, k, None) == v for k, v in where.items()):
                    for k, v in data.items():
                        setattr(r, k, v)
                    count += 1
            return _Row(count=count)


class _Empty:
    async def find_many(self, where=None) -> List[_Row]:
        return []

    async def find_unique(self, where=None) -> Optional[_Row]:
        return None


class _DB:
    def __init__(self) -> None:
        self.litellm_agentvm = _AgentVMTable()
        self.litellm_agentsecret = _Empty()
        self.litellm_agentvmconfig = _Empty()


class _Prisma:
    def __init__(self) -> None:
        self.db = _DB()


@pytest.fixture
def prisma():
    return _Prisma()


@pytest.fixture
def fake_team_creds(monkeypatch):
    """Stub `get_team_vm_config` so we don't try to decrypt anything."""

    async def fake(team_id: str, prisma_client, default_region: str = "us-west-2"):
        return TeamVMConfig(
            aws_creds=AwsCreds(
                access_key_id="AKIATEST",
                secret_access_key="secrettest",
                region="us-west-2",
            ),
            ec2_config=Ec2Config(region="us-west-2"),
        )

    monkeypatch.setattr(
        "litellm.managed_agents.vms.warm_pool.attach.get_team_vm_config",
        fake,
    )


@pytest.fixture
def fake_transport():
    """A transport whose `push` is awaited but does nothing."""
    return AsyncMock()


def _seed_warm(prisma: _Prisma, team_id: str, count: int) -> List[str]:
    """Seed `count` warm rows for `team_id`. Returns the seeded ids."""
    now = datetime.now(timezone.utc)
    ids = []
    for i in range(count):
        # Stagger warmed_at so the oldest is picked first.
        warmed_at = now - timedelta(seconds=count - i)
        row = _Row(
            id=f"i-{team_id}-{i}",
            team_id=team_id,
            pool_id=team_id,
            state="warm",
            warmed_at=warmed_at,
            provider="ec2",
            region="us-west-2",
            metadata={},
        )
        prisma.db.litellm_agentvm.rows.append(row)
        ids.append(row.id)
    return ids


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_attach_empty_pool_raises(prisma, fake_team_creds, fake_transport):
    """No warm rows -> `WarmPoolEmptyError`."""
    with pytest.raises(WarmPoolEmptyError):
        await attach_warm_vm(
            prisma_client=prisma,
            team_id="team-A",
            session_id="sess-1",
            agent_id="agent-1",
            jwt="jwt",
            jwt_expires_at=datetime.now(timezone.utc),
            repos=[],
            env_vars=None,
            transport=fake_transport,
        )


@pytest.mark.asyncio
async def test_attach_picks_warm_and_marks_attached(
    prisma, fake_team_creds, fake_transport
):
    """Happy path: oldest warm row -> hydrating -> attached."""
    ids = _seed_warm(prisma, "team-A", count=2)

    result = await attach_warm_vm(
        prisma_client=prisma,
        team_id="team-A",
        session_id="sess-1",
        agent_id="agent-1",
        jwt="jwt",
        jwt_expires_at=datetime.now(timezone.utc),
        repos=[{"url": "https://github.com/foo/bar"}],
        env_vars={"NODE_ENV": "test"},
        transport=fake_transport,
    )

    # The oldest warm row was claimed.
    assert result.vm_id == ids[0]
    # Final state on the row is `attached`, with our session id.
    row = await prisma.db.litellm_agentvm.find_unique(where={"id": ids[0]})
    assert row.state == "attached"
    assert row.attached_session_id == "sess-1"

    # Transport got the payload exactly once.
    fake_transport.push.assert_awaited_once()


@pytest.mark.asyncio
async def test_attach_transport_failure_releases_row(
    prisma, fake_team_creds, fake_transport
):
    """If transport.push raises, the row goes to `terminating` and the
    exception propagates (caller decides to fall through to cold boot)."""
    _seed_warm(prisma, "team-A", count=1)
    fake_transport.push.side_effect = RuntimeError("ssm boom")

    with pytest.raises(RuntimeError):
        await attach_warm_vm(
            prisma_client=prisma,
            team_id="team-A",
            session_id="sess-1",
            agent_id="agent-1",
            jwt="jwt",
            jwt_expires_at=datetime.now(timezone.utc),
            repos=[],
            env_vars=None,
            transport=fake_transport,
        )

    # Row is now `terminating`, not back to `warm` (we don't recycle).
    row = prisma.db.litellm_agentvm.rows[0]
    assert row.state == "terminating"
    assert row.attached_session_id is None


@pytest.mark.asyncio
async def test_concurrent_attach_no_double_claim(
    prisma, fake_team_creds, fake_transport
):
    """5 concurrent attaches against pool size 3 -> exactly 3 succeed."""
    _seed_warm(prisma, "team-A", count=3)

    async def one_attach(sid: str):
        try:
            r = await attach_warm_vm(
                prisma_client=prisma,
                team_id="team-A",
                session_id=sid,
                agent_id="agent-1",
                jwt="jwt",
                jwt_expires_at=datetime.now(timezone.utc),
                repos=[],
                env_vars=None,
                transport=fake_transport,
            )
            return ("ok", r.vm_id)
        except WarmPoolEmptyError:
            return ("empty", None)

    results = await asyncio.gather(*(one_attach(f"sess-{i}") for i in range(5)))
    successes = [r for r in results if r[0] == "ok"]
    empties = [r for r in results if r[0] == "empty"]

    assert len(successes) == 3
    assert len(empties) == 2

    # No two successes attached the same VM.
    vm_ids = [s[1] for s in successes]
    assert len(set(vm_ids)) == len(vm_ids)


@pytest.mark.asyncio
async def test_attach_skips_other_team_warm_vms(
    prisma, fake_team_creds, fake_transport
):
    """Warm rows for a different team are not visible to this team's attach."""
    _seed_warm(prisma, "team-OTHER", count=3)

    with pytest.raises(WarmPoolEmptyError):
        await attach_warm_vm(
            prisma_client=prisma,
            team_id="team-A",
            session_id="sess-1",
            agent_id="agent-1",
            jwt="jwt",
            jwt_expires_at=datetime.now(timezone.utc),
            repos=[],
            env_vars=None,
            transport=fake_transport,
        )
