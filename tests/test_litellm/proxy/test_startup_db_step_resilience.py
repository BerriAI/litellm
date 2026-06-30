"""
XCT fork: startup DB-load steps must not be able to wedge application startup.

A single hanging step (e.g. a large agents/MCP table or a remote model-cost-map
fetch) used to block the whole uvicorn startup lifespan — the ASGI app never
served /health/liveliness and the container never became healthy, failing every
deploy. ProxyConfig._run_startup_db_step wraps each step with timing + a hard
timeout and swallows hangs/errors so startup always proceeds.
"""

import asyncio
import time

import pytest

from litellm.proxy.proxy_server import ProxyConfig


def _bare_config() -> ProxyConfig:
    # avoid __init__ side effects; the helper uses no instance state
    return ProxyConfig.__new__(ProxyConfig)


@pytest.mark.asyncio
async def test_hanging_step_times_out_and_returns_fast():
    pc = _bare_config()

    async def hangs():
        await asyncio.sleep(30)

    start = time.time()
    # must NOT raise, and must bail at ~timeout rather than waiting 30s
    await pc._run_startup_db_step("hang", hangs, timeout=0.2)
    assert time.time() - start < 2.0


@pytest.mark.asyncio
async def test_failing_step_is_swallowed():
    pc = _bare_config()

    async def boom():
        raise RuntimeError("kaboom")

    # must NOT propagate — startup continues with this object type unloaded
    await pc._run_startup_db_step("boom", boom, timeout=5)


@pytest.mark.asyncio
async def test_successful_step_runs_to_completion():
    pc = _bare_config()
    ran = {}

    async def ok():
        ran["did"] = True

    await pc._run_startup_db_step("ok", ok, timeout=5)
    assert ran.get("did") is True
