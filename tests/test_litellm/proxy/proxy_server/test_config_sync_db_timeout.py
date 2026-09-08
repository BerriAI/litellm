"""Behavior tests for the config-sync DB timeout guard.

``_with_sync_db_timeout`` bounds each DB call made by the periodic
``ProxyConfig.add_deployment`` job with ``PROXY_CONFIG_SYNC_DB_TIMEOUT_SECONDS``.

Why this exists: a DB query that lands on a dead pooled connection neither
errors nor returns. With APScheduler ``max_instances=1`` the periodic config
sync hangs forever, every later run is skipped, and the in-memory model list
freezes -- requests for models added after the hang return 400 "Invalid model
name" until the process restarts. The timeout turns the hang into a normal
exception so the existing skip-and-retry-next-tick handling takes over.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

import litellm.proxy.proxy_server as proxy_server_module
from litellm.proxy.proxy_server import ProxyConfig


def _patch_sync_timeout(monkeypatch: pytest.MonkeyPatch, seconds: float) -> None:
    """Patch the timeout constant on the module that reads it.

    ``proxy_server`` imports the constant by value at module import time, so
    patch the module attribute the function references.
    """
    monkeypatch.setattr(proxy_server_module, "PROXY_CONFIG_SYNC_DB_TIMEOUT_SECONDS", seconds)


def _hang_forever() -> asyncio.Future:
    """An awaitable that never completes: simulates a query stuck on a dead
    pooled connection (no response, no error, no cancellation on its own)."""
    return asyncio.get_running_loop().create_future()


class _FakePrismaClient:
    """Minimal stand-in; every DB step is mocked per-test."""


class TestWithSyncDbTimeout:
    async def test_hanging_db_call_times_out(self, monkeypatch: pytest.MonkeyPatch):
        _patch_sync_timeout(monkeypatch, 0.1)
        with pytest.raises(asyncio.TimeoutError):
            await proxy_server_module._with_sync_db_timeout(_hang_forever(), step="unit-test")

    async def test_fast_db_call_unaffected(self, monkeypatch: pytest.MonkeyPatch):
        _patch_sync_timeout(monkeypatch, 5)

        async def quick() -> str:
            await asyncio.sleep(0)
            return "ok"

        assert await proxy_server_module._with_sync_db_timeout(quick(), step="unit-test") == "ok"

    async def test_timeout_disabled_passthrough(self, monkeypatch: pytest.MonkeyPatch):
        _patch_sync_timeout(monkeypatch, 0)

        async def slow_but_finite() -> str:
            await asyncio.sleep(0.2)
            return "done"

        # With the timeout disabled the call runs to completion.
        start = time.monotonic()
        assert await proxy_server_module._with_sync_db_timeout(slow_but_finite(), step="unit-test") == "done"
        assert time.monotonic() - start >= 0.2


class TestAddDeploymentDoesNotHang:
    async def test_add_deployment_returns_when_model_fetch_hangs(self, monkeypatch: pytest.MonkeyPatch):
        """The incident scenario: the model-table query hangs on a dead
        connection. ``add_deployment`` must return (not hang) so the next
        scheduler tick can proceed."""
        _patch_sync_timeout(monkeypatch, 0.2)
        proxy_config = ProxyConfig()

        async def _hang(self, prisma_client):
            return await _hang_forever()

        monkeypatch.setattr(ProxyConfig, "_get_models_from_db", _hang)
        # Non-DB steps still run normally.
        monkeypatch.setattr(
            proxy_server_module,
            "prefetch_config_params",
            AsyncMock(return_value=None),
        )
        monkeypatch.setattr(proxy_server_module, "get_config_param", AsyncMock(return_value=None))
        monkeypatch.setattr(
            ProxyConfig,
            "_init_non_llm_objects_in_db",
            AsyncMock(return_value=None),
        )
        monkeypatch.setattr(ProxyConfig, "_should_load_db_object", lambda self, object_type: True)

        start = time.monotonic()
        outcome = await asyncio.wait_for(
            proxy_config.add_deployment(
                prisma_client=_FakePrismaClient(),  # type: ignore[arg-type]
                proxy_logging_obj=None,
            ),
            timeout=10,
        )
        elapsed = time.monotonic() - start

        assert outcome is not None
        assert outcome.still_desired is None  # no reconcile ran
        assert elapsed < 5  # returned promptly instead of hanging forever

    async def test_reconcile_lock_released_after_timeout(self, monkeypatch: pytest.MonkeyPatch):
        """After a timed-out run, MODEL_RECONCILE_LOCK must be free so other
        reconcile callers (model writes, clear_cache) are not blocked."""
        _patch_sync_timeout(monkeypatch, 0.2)
        proxy_config = ProxyConfig()

        async def _hang(self, prisma_client):
            return await _hang_forever()

        monkeypatch.setattr(ProxyConfig, "_get_models_from_db", _hang)
        monkeypatch.setattr(
            proxy_server_module,
            "prefetch_config_params",
            AsyncMock(return_value=None),
        )
        monkeypatch.setattr(ProxyConfig, "_should_load_db_object", lambda self, object_type: True)

        await asyncio.wait_for(
            proxy_config.add_deployment(
                prisma_client=_FakePrismaClient(),  # type: ignore[arg-type]
                proxy_logging_obj=None,
            ),
            timeout=10,
        )

        # The lock was held during the run; it must be released now.
        assert not proxy_server_module.MODEL_RECONCILE_LOCK.locked()
        # And a second run can acquire it immediately.
        acquired = await asyncio.wait_for(proxy_server_module.MODEL_RECONCILE_LOCK.acquire(), timeout=1)
        try:
            assert acquired
        finally:
            proxy_server_module.MODEL_RECONCILE_LOCK.release()
