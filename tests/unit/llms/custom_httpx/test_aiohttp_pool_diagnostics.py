"""Tests for the aiohttp pool-saturation warning (issue #46864)."""

import asyncio
import logging

import pytest
from aiohttp import ClientSession, ClientTimeout, TCPConnector, web

from litellm.llms.custom_httpx.aiohttp_pool_diagnostics import (
    build_pool_wait_trace_config,
)


async def _start_slow_server():
    async def handler(request):
        await asyncio.sleep(0.3)
        return web.Response(text="ok")

    app = web.Application()
    app.router.add_get("/", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}/"


def test_disabled_when_threshold_not_positive():
    assert build_pool_wait_trace_config(threshold_seconds=0) is None
    assert build_pool_wait_trace_config(threshold_seconds=-1) is None
    assert build_pool_wait_trace_config(threshold_seconds=0.5) is not None


async def _run_concurrent(threshold, limit_per_host):
    runner, url = await _start_slow_server()
    try:
        return await _fetch_twice(url, threshold, limit_per_host)
    finally:
        await runner.cleanup()


async def _fetch_twice(url, threshold, limit_per_host):
    trace_config = build_pool_wait_trace_config(threshold_seconds=threshold)
    async with ClientSession(
        connector=TCPConnector(limit_per_host=limit_per_host),
        trace_configs=[trace_config],
        timeout=ClientTimeout(total=10),
    ) as session:

        async def fetch():
            async with session.get(url) as resp:
                return await resp.text()

        return await asyncio.gather(fetch(), fetch())


@pytest.mark.asyncio
async def test_warns_when_request_waits_for_pool_slot(caplog):
    caplog.set_level(logging.WARNING, logger="LiteLLM")
    results = await _run_concurrent(threshold=0.1, limit_per_host=1)
    assert results == ["ok", "ok"]
    messages = [r.getMessage() for r in caplog.records]
    assert any("connection pool saturated" in m and "host=127.0.0.1" in m for m in messages), messages


@pytest.mark.asyncio
async def test_no_warning_when_pool_has_capacity(caplog):
    caplog.set_level(logging.WARNING, logger="LiteLLM")
    await _run_concurrent(threshold=0.1, limit_per_host=10)
    assert not any("connection pool saturated" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_no_warning_when_wait_below_threshold(caplog):
    caplog.set_level(logging.WARNING, logger="LiteLLM")
    await _run_concurrent(threshold=5.0, limit_per_host=1)
    assert not any("connection pool saturated" in r.getMessage() for r in caplog.records)
