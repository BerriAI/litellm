#!/usr/bin/env python3
"""
In-process memory profiler for LiteLLM proxy.

This script starts the proxy with tracemalloc enabled, sends load,
then takes snapshots to identify the top allocation sites.

Usage:
    unset DATABASE_URL && poetry run python scripts/profile_memory.py
"""

import asyncio
import gc
import os
import sys
import time
import tracemalloc

# Must start tracemalloc BEFORE importing litellm
tracemalloc.start(25)  # 25 frames deep

import aiohttp
import psutil
import uvicorn


LARGE_CONTENT = "Explain the theory of relativity in detail. " * 20

ANTHROPIC_BODY = {
    "model": "claude-3-5-sonnet",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": LARGE_CONTENT}],
}
ANTHROPIC_HEADERS = {
    "Authorization": "Bearer sk-1234",
    "Content-Type": "application/json",
    "anthropic-version": "2023-06-01",
}


async def send_requests(n: int, concurrency: int = 50):
    """Send n requests to the proxy."""
    sem = asyncio.Semaphore(concurrency)
    conn = aiohttp.TCPConnector(limit=concurrency * 2)

    async def send(session):
        async with sem:
            async with session.post(
                "http://localhost:4000/v1/messages",
                json=ANTHROPIC_BODY,
                headers=ANTHROPIC_HEADERS,
            ) as resp:
                await resp.read()

    async with aiohttp.ClientSession(connector=conn) as session:
        tasks = [send(session) for _ in range(n)]
        await asyncio.gather(*tasks, return_exceptions=True)


def get_rss_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def print_top_allocations(snapshot, title, limit=20):
    """Print top memory allocations from tracemalloc snapshot."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

    # Group by file:line
    stats = snapshot.statistics("lineno")
    print(f"\n  Top {limit} allocations by file:line:")
    for i, stat in enumerate(stats[:limit]):
        print(f"    {i+1:3d}. {stat.size / 1024:10.1f} KiB  ({stat.count:6d} blocks)  {stat}")

    # Group by filename
    print(f"\n  Top {limit} allocations by file:")
    stats = snapshot.statistics("filename")
    for i, stat in enumerate(stats[:limit]):
        print(f"    {i+1:3d}. {stat.size / 1024:10.1f} KiB  ({stat.count:6d} blocks)  {stat}")


def print_snapshot_diff(snap1, snap2, title, limit=20):
    """Print the difference between two snapshots."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

    stats = snap2.compare_to(snap1, "lineno")
    print(f"\n  Top {limit} memory GROWTH by file:line:")
    for i, stat in enumerate(stats[:limit]):
        print(f"    {i+1:3d}. {stat.size_diff / 1024:+10.1f} KiB  ({stat.count_diff:+6d} blocks)  {stat}")

    stats = snap2.compare_to(snap1, "filename")
    print(f"\n  Top {limit} memory GROWTH by file:")
    for i, stat in enumerate(stats[:limit]):
        print(f"    {i+1:3d}. {stat.size_diff / 1024:+10.1f} KiB  ({stat.count_diff:+6d} blocks)  {stat}")


async def main():
    print("Starting LiteLLM proxy with tracemalloc profiling...")
    print(f"Initial RSS: {get_rss_mb():.1f} MB")

    # Import and configure litellm
    os.environ.pop("DATABASE_URL", None)

    from litellm.proxy.proxy_server import app

    # Start proxy in background
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=4000,
        log_level="warning",
    )
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())

    # Wait for proxy to be ready
    print("Waiting for proxy startup...")
    for _ in range(60):
        try:
            async with aiohttp.ClientSession() as sess:
                async with sess.get(
                    "http://localhost:4000/health",
                    headers={"Authorization": "Bearer sk-1234"},
                ) as resp:
                    if resp.status == 200:
                        break
        except Exception:
            pass
        await asyncio.sleep(1)
    else:
        print("ERROR: Proxy didn't start")
        return

    print(f"Proxy ready. RSS: {get_rss_mb():.1f} MB")

    # Force GC and take baseline snapshot
    gc.collect()
    snap_baseline = tracemalloc.take_snapshot()
    rss_baseline = get_rss_mb()
    print(f"\n--- Baseline snapshot taken. RSS: {rss_baseline:.1f} MB ---")

    # Phase 1: Send 5000 requests
    print("\nSending 5000 requests...")
    t0 = time.time()
    await send_requests(5000, concurrency=50)
    t1 = time.time()
    print(f"  Done in {t1-t0:.1f}s ({5000/(t1-t0):.0f} rps)")

    gc.collect()
    snap_after_5k = tracemalloc.take_snapshot()
    rss_after_5k = get_rss_mb()
    print(f"  RSS: {rss_after_5k:.1f} MB (+{rss_after_5k - rss_baseline:.1f} MB)")

    # Phase 2: Send 10000 more requests
    print("\nSending 10000 more requests...")
    t0 = time.time()
    await send_requests(10000, concurrency=50)
    t1 = time.time()
    print(f"  Done in {t1-t0:.1f}s ({10000/(t1-t0):.0f} rps)")

    gc.collect()
    snap_after_15k = tracemalloc.take_snapshot()
    rss_after_15k = get_rss_mb()
    print(f"  RSS: {rss_after_15k:.1f} MB (+{rss_after_15k - rss_baseline:.1f} MB)")

    # Print analysis
    print_snapshot_diff(snap_baseline, snap_after_5k, "GROWTH after 5K requests")
    print_snapshot_diff(snap_after_5k, snap_after_15k, "GROWTH from 5K→15K requests (steady state)")
    print_top_allocations(snap_after_15k, "TOP ALLOCATIONS after 15K requests")

    # Shutdown
    server.should_exit = True
    await server_task


if __name__ == "__main__":
    # Need to load the proxy config
    sys.argv = ["litellm", "--config", "scripts/repro_memory_leak_config.yaml"]
    asyncio.run(main())
