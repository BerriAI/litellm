"""
Leak inducer — hooks into litellm proxy to simulate the production memory leak.

This module is loaded as a custom callback via the proxy config. On startup, it:

1. Sets RLIMIT_AS to cap worker memory (so workers crash quickly for the demo)
2. Creates a mock PrismaClient-like object and sets it as the global prisma_client
   so that spend_log_transactions accumulates but never drains (simulating DB-down)

This reproduces the exact production scenario: spend_log_transactions grows without
bound because the DB is unreachable, eventually causing MemoryError and worker death.
"""

import asyncio
import os
import resource
import sys

from litellm._logging import verbose_proxy_logger


def _apply_memory_cap():
    """Apply RLIMIT_AS to cap worker virtual memory, forcing faster crash."""
    cap_mb = int(os.environ.get("_REPRO_WORKER_MEM_CAP_MB", "0"))
    if cap_mb <= 0:
        return
    cap_bytes = cap_mb * 1024 * 1024
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (cap_bytes, hard))
        print(f"[leak_inducer] RLIMIT_AS set to {cap_mb}MB (pid={os.getpid()})")
    except Exception as e:
        print(f"[leak_inducer] Failed to set RLIMIT_AS: {e}")


class _MockSpendLogTransactions:
    """
    A mock that replaces PrismaClient as the global prisma_client.
    Has the spend_log_transactions list and lock so the spend log append path works.
    All other attribute accesses return a no-op to prevent crashes.
    """

    def __init__(self):
        self.spend_log_transactions = []
        self._spend_log_transactions_lock = asyncio.Lock()
        self._mock_name = "MockPrismaClient"

    def __getattr__(self, name):
        # Return a no-op callable for any method called on this mock
        if name.startswith("_"):
            raise AttributeError(name)

        class _NoOp:
            def __call__(self, *a, **kw):
                return self

            def __await__(self):
                async def _noop():
                    return None
                return _noop().__await__()

            def __getattr__(self, n):
                return _NoOp()

        return _NoOp()

    def __bool__(self):
        return True  # so `if prisma_client is not None` passes


def _inject_mock_prisma_client():
    """Replace the global prisma_client with our mock so spend logs accumulate."""
    try:
        import litellm.proxy.proxy_server as ps

        mock = _MockSpendLogTransactions()
        ps.prisma_client = mock
        print(
            f"[leak_inducer] Injected mock prisma_client (pid={os.getpid()}). "
            f"spend_log_transactions will accumulate but never drain."
        )
    except Exception as e:
        print(f"[leak_inducer] Failed to inject mock prisma_client: {e}")


def _patch_spend_log_flush():
    """
    Patch the update_spend_logs to be a no-op.
    This ensures the spend_log_transactions list is NEVER drained,
    simulating a DB that is permanently unreachable.
    """
    try:
        import litellm.proxy.utils as pu

        original_update_spend_logs = pu.ProxyUpdateSpend.update_spend_logs

        @staticmethod
        async def _noop_update_spend_logs(*args, **kwargs):
            # Don't drain the queue — simulate DB failure
            verbose_proxy_logger.debug(
                "[leak_inducer] update_spend_logs called but suppressed (simulating DB failure)"
            )
            return

        pu.ProxyUpdateSpend.update_spend_logs = _noop_update_spend_logs
        print(f"[leak_inducer] Patched update_spend_logs to no-op (pid={os.getpid()})")
    except Exception as e:
        print(f"[leak_inducer] Failed to patch update_spend_logs: {e}")


# Apply on import (this runs in each worker process)
_apply_memory_cap()

# Delay the prisma_client injection until after the event loop is running
# (the proxy startup sets prisma_client during the lifespan event)
import threading


def _delayed_inject():
    """Wait a bit for the proxy to finish startup, then inject the mock."""
    import time
    time.sleep(5)  # Wait for lifespan startup to complete
    _inject_mock_prisma_client()
    _patch_spend_log_flush()
    print(f"[leak_inducer] Setup complete (pid={os.getpid()})")


_thread = threading.Thread(target=_delayed_inject, daemon=True)
_thread.start()
