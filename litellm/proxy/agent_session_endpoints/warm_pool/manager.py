"""Warm pool maintenance loop (LIT-2890).

Per-team background task that:
  1. Reads `LiteLLM_AgentVMConfig.warm_pool_size` for every team where the
     feature is enabled
  2. Counts how many `LiteLLM_AgentVM` rows are currently `state='warm'`
  3. Spawns the deficit by calling `EC2Provider.provision(mode='warm')`
  4. Reaps `state='warm'` rows older than `max_idle_minutes` (validation #8)

Runs every `WARM_POOL_TICK_SECONDS` seconds. Designed to be started ONCE
on proxy boot via `WarmPoolManager.start()` from `proxy_server.py`.

The hot path (`POST /v2/sessions`) calls `attach.attach_warm_vm` directly
and never blocks on the maintenance loop — the loop only refills the pool
back to `warm_pool_size` after an attach.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from litellm._logging import verbose_proxy_logger
from litellm.proxy.agent_session_endpoints.vm_providers.base import (
    AwsCreds,
    Ec2Config,
    ProvisionContext,
    Repo,
)
from litellm.proxy.agent_session_endpoints.vm_providers.factory import (
    build_vm_provider,
)
from litellm.proxy.agent_session_endpoints.vm_providers.team_config import (
    get_team_vm_config,
)

# Tick rate for the maintenance loop. Per LIT-2890 spec: every 30s.
# Tunable via `WARM_POOL_TICK_SECONDS` env var for tests (set to 1 there).
WARM_POOL_TICK_SECONDS = 30

# How many spawn calls we issue concurrently per team per tick. Cap so a
# single team can't saturate the proxy's thread pool.
MAX_CONCURRENT_SPAWNS_PER_TEAM = 5


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_pool_vm_id() -> str:
    """Internal id used while the EC2 provision is in flight.

    The row is rekeyed to the EC2 instance id when the provider returns;
    callers should use this id as a placeholder only.
    """
    return f"vm_pending_{uuid.uuid4().hex[:12]}"


class WarmPoolManager:
    """Async background task that refills + reaps warm-pool VMs.

    Lifecycle:
      - `start()` schedules the loop on the running event loop
      - `stop()` cancels the loop and awaits its completion
      - `tick()` runs ONE iteration (called by start; useful in tests)

    Idempotent restart: calling `start()` while running is a no-op.
    """

    def __init__(
        self,
        *,
        prisma_getter: Optional[Any] = None,
        provider_settings: Optional[Dict[str, Any]] = None,
        tick_seconds: float = WARM_POOL_TICK_SECONDS,
    ) -> None:
        self._tick_seconds = tick_seconds
        self._provider_settings = provider_settings or {}
        # Allow tests to inject a prisma client without spinning up the proxy.
        self._prisma_getter = prisma_getter
        self._task: Optional[asyncio.Task[None]] = None
        self._stopping = asyncio.Event()

    # ---------- public lifecycle ----------

    def start(self) -> None:
        """Schedule the loop on the running event loop. Idempotent."""
        if self._task is not None and not self._task.done():
            return
        self._stopping.clear()
        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self._run_forever())

    async def stop(self) -> None:
        """Cancel the loop and wait for it to exit."""
        self._stopping.set()
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except (asyncio.CancelledError, Exception):
            pass
        self._task = None

    # ---------- core loop ----------

    async def _run_forever(self) -> None:
        verbose_proxy_logger.info(
            "warm_pool.manager: started (tick=%ss)", self._tick_seconds
        )
        while not self._stopping.is_set():
            try:
                await self.tick()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                verbose_proxy_logger.exception(
                    "warm_pool.manager: tick failed: %s", exc
                )
            try:
                await asyncio.wait_for(
                    self._stopping.wait(), timeout=self._tick_seconds
                )
            except asyncio.TimeoutError:
                pass
        verbose_proxy_logger.info("warm_pool.manager: stopped")

    async def tick(self) -> None:
        """Run one maintenance iteration: refill + reap, per team."""
        prisma_client = self._get_prisma_client()
        if prisma_client is None:
            return

        teams = await self._enabled_teams(prisma_client)
        if not teams:
            return

        # Per-team work: refill deficit, reap stale.
        await asyncio.gather(
            *(self._maintain_team(prisma_client, team) for team in teams),
            return_exceptions=True,
        )

    # ---------- per-team work ----------

    async def _maintain_team(self, prisma_client: Any, vm_config: Any) -> None:
        team_id = vm_config.team_id
        desired = int(getattr(vm_config, "warm_pool_size", 0) or 0)
        max_idle_minutes = int(getattr(vm_config, "max_idle_minutes", 30) or 30)

        # Reap first — frees slots before we refill.
        await self._reap_stale_warm(prisma_client, team_id, max_idle_minutes)

        # Then count what's still warm + provisioning, spawn the deficit.
        existing = await prisma_client.db.litellm_agentvm.find_many(
            where={
                "team_id": team_id,
                "state": {"in": ["provisioning", "warm"]},
            }
        )
        deficit = max(0, desired - len(existing))
        if deficit == 0:
            # Cap: if we have more warm than desired (e.g. user shrunk pool),
            # terminate the oldest excess.
            await self._shrink_to_desired(prisma_client, team_id, existing, desired)
            return

        spawn_n = min(deficit, MAX_CONCURRENT_SPAWNS_PER_TEAM)
        verbose_proxy_logger.info(
            "warm_pool.manager: team=%s desired=%s have=%s spawning=%s",
            team_id,
            desired,
            len(existing),
            spawn_n,
        )
        await asyncio.gather(
            *(self._spawn_one(prisma_client, vm_config) for _ in range(spawn_n)),
            return_exceptions=True,
        )

    async def _spawn_one(self, prisma_client: Any, vm_config: Any) -> None:
        """Provision one warm VM end-to-end: row -> provider call -> rekey."""
        team_id = vm_config.team_id
        placeholder_id = _new_pool_vm_id()
        try:
            await prisma_client.db.litellm_agentvm.create(
                data={
                    "id": placeholder_id,
                    "provider": "ec2",
                    "region": getattr(vm_config, "aws_region", None),
                    "state": "provisioning",
                    "team_id": team_id,
                    "pool_id": team_id,
                    "metadata": {"placeholder": True},
                }
            )
        except Exception as exc:
            verbose_proxy_logger.exception(
                "warm_pool.manager: failed to insert placeholder team=%s: %s",
                team_id,
                exc,
            )
            return

        try:
            team_resolved = await get_team_vm_config(team_id, prisma_client)
            aws_creds = team_resolved.aws_creds
            ec2_config = team_resolved.ec2_config

            provider = build_vm_provider(
                {"vm_provider": "ec2", "ec2": self._provider_settings}
            )
            ctx = ProvisionContext(
                session_id=placeholder_id,
                team_id=team_id,
                agent_id=None,
                repos=[],
                env_vars={},
                secrets={},
                aws_creds=aws_creds,
                ec2_config=ec2_config,
                daemon_jwt=None,
                daemon_base_url=None,
                mode="warm",
            )
            handle = await provider.provision(ctx)
        except Exception as exc:
            verbose_proxy_logger.exception(
                "warm_pool.manager: provision failed team=%s placeholder=%s: %s",
                team_id,
                placeholder_id,
                exc,
            )
            try:
                await prisma_client.db.litellm_agentvm.update(
                    where={"id": placeholder_id},
                    data={
                        "state": "terminated",
                        "terminated_at": _now(),
                        "metadata": {"error": type(exc).__name__},
                    },
                )
            except Exception:
                pass
            return

        # Rekey placeholder -> real EC2 id, mark warm. We delete + create
        # because the id is part of the primary key.
        try:
            await prisma_client.db.litellm_agentvm.delete(where={"id": placeholder_id})
            await prisma_client.db.litellm_agentvm.create(
                data={
                    "id": handle.vm_id,
                    "provider": handle.provider,
                    "region": handle.region,
                    "state": "warm",
                    "team_id": team_id,
                    "pool_id": team_id,
                    "warmed_at": _now(),
                    "metadata": handle.metadata or {},
                }
            )
            verbose_proxy_logger.info(
                "warm_pool.manager: warmed team=%s vm_id=%s", team_id, handle.vm_id
            )
        except Exception as exc:
            verbose_proxy_logger.exception(
                "warm_pool.manager: rekey failed team=%s placeholder=%s real=%s: %s",
                team_id,
                placeholder_id,
                handle.vm_id,
                exc,
            )

    # ---------- reaping ----------

    async def _reap_stale_warm(
        self, prisma_client: Any, team_id: str, max_idle_minutes: int
    ) -> None:
        cutoff = _now() - timedelta(minutes=max_idle_minutes)
        stale = await prisma_client.db.litellm_agentvm.find_many(
            where={
                "team_id": team_id,
                "state": "warm",
                "warmed_at": {"lt": cutoff},
            }
        )
        if not stale:
            return
        verbose_proxy_logger.info(
            "warm_pool.manager: reaping team=%s count=%s cutoff=%s",
            team_id,
            len(stale),
            cutoff.isoformat(),
        )
        for row in stale:
            await self._terminate_vm(prisma_client, row)

    async def _shrink_to_desired(
        self,
        prisma_client: Any,
        team_id: str,
        existing: List[Any],
        desired: int,
    ) -> None:
        warm_only = [r for r in existing if r.state == "warm"]
        if len(warm_only) <= desired:
            return
        warm_only.sort(key=lambda r: r.warmed_at or r.created_at)
        excess = warm_only[: len(warm_only) - desired]
        for row in excess:
            await self._terminate_vm(prisma_client, row)

    async def _terminate_vm(self, prisma_client: Any, row: Any) -> None:
        """Best-effort terminate: mark `terminating`, fire provider, mark `terminated`."""
        try:
            await prisma_client.db.litellm_agentvm.update(
                where={"id": row.id},
                data={"state": "terminating"},
            )
            try:
                team_resolved = await get_team_vm_config(row.team_id, prisma_client)
                aws_creds = team_resolved.aws_creds
            except Exception:
                aws_creds = None
            if aws_creds is not None and row.provider == "ec2":
                provider = build_vm_provider(
                    {"vm_provider": "ec2", "ec2": self._provider_settings}
                )
                from litellm.proxy.agent_session_endpoints.vm_providers.base import (
                    VMHandle,
                )

                handle = VMHandle(
                    vm_id=row.id,
                    provider="ec2",
                    region=row.region,
                    metadata=dict(row.metadata or {}),
                )
                await provider.terminate(handle, aws_creds=aws_creds)
            await prisma_client.db.litellm_agentvm.update(
                where={"id": row.id},
                data={"state": "terminated", "terminated_at": _now()},
            )
        except Exception as exc:
            verbose_proxy_logger.warning(
                "warm_pool.manager: terminate failed vm=%s: %s", row.id, exc
            )

    # ---------- helpers ----------

    async def _enabled_teams(self, prisma_client: Any) -> List[Any]:
        """Return the list of `LiteLLM_AgentVMConfig` rows with warm pool on."""
        return await prisma_client.db.litellm_agentvmconfig.find_many(
            where={
                "warm_pool_enabled": True,
                "warm_pool_size": {"gt": 0},
            }
        )

    def _get_prisma_client(self) -> Any:
        if self._prisma_getter is not None:
            return self._prisma_getter()
        from litellm.proxy.proxy_server import prisma_client

        return prisma_client


# Module-level singleton. `proxy_server.py` calls `get_warm_pool_manager().start()`
# on startup and `.stop()` on shutdown.
_MANAGER_SINGLETON: Optional[WarmPoolManager] = None


def get_warm_pool_manager() -> WarmPoolManager:
    """Return (or create) the process-wide warm-pool manager."""
    global _MANAGER_SINGLETON
    if _MANAGER_SINGLETON is None:
        _MANAGER_SINGLETON = WarmPoolManager()
    return _MANAGER_SINGLETON


def reset_warm_pool_manager() -> None:
    """Test helper: clear the singleton between tests."""
    global _MANAGER_SINGLETON
    _MANAGER_SINGLETON = None


__all__ = [
    "MAX_CONCURRENT_SPAWNS_PER_TEAM",
    "WARM_POOL_TICK_SECONDS",
    "WarmPoolManager",
    "get_warm_pool_manager",
    "reset_warm_pool_manager",
]


# Re-export Repo / AwsCreds so tests can build fixtures without crawling deep
# into the providers package.
__all__ += ["AwsCreds", "Ec2Config", "Repo"]
