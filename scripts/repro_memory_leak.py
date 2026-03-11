#!/usr/bin/env python3
"""
Memory Leak Reproduction Script for LiteLLM Proxy

Reproduces the linear memory growth pattern reported in production:
- Sends sustained traffic to /v1/messages (Anthropic format) and /chat/completions
- Monitors RSS memory, file descriptors, and aiohttp session counts
- Reports metrics every N seconds
- Runs for a configurable duration

Usage:
    # Start proxy first (unset DATABASE_URL to avoid default Neon DB):
    #   unset DATABASE_URL && poetry run litellm --config scripts/repro_memory_leak_config.yaml --port 4000
    #
    # Then run this script:
    poetry run python scripts/repro_memory_leak.py

    # Options:
    poetry run python scripts/repro_memory_leak.py --duration 600 --concurrency 100 --port 4000
"""

import argparse
import asyncio
import json
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import aiohttp
import psutil


# ─── Request templates ────────────────────────────────────────────────

# Use large messages to increase per-request memory pressure
# (mimics production payloads with long conversation histories)
LARGE_CONTENT = "Explain the theory of relativity in detail. " * 20  # ~800 chars

ANTHROPIC_MESSAGES_BODY = {
    "model": "claude-3-5-sonnet",
    "max_tokens": 100,
    "messages": [
        {"role": "user", "content": LARGE_CONTENT},
    ],
}

ANTHROPIC_MESSAGES_BODY_STREAM = {
    "model": "claude-3-5-sonnet",
    "max_tokens": 100,
    "stream": True,
    "messages": [
        {"role": "user", "content": LARGE_CONTENT},
    ],
}

CHAT_COMPLETIONS_BODY = {
    "model": "gpt-4o",
    "max_tokens": 100,
    "messages": [
        {"role": "user", "content": LARGE_CONTENT},
    ],
}

CHAT_COMPLETIONS_BODY_STREAM = {
    "model": "gpt-4o",
    "max_tokens": 100,
    "stream": True,
    "messages": [
        {"role": "user", "content": LARGE_CONTENT},
    ],
}

HEADERS = {
    "Authorization": "Bearer sk-1234",
    "Content-Type": "application/json",
}

ANTHROPIC_HEADERS = {
    "Authorization": "Bearer sk-1234",
    "Content-Type": "application/json",
    "anthropic-version": "2023-06-01",
}


# ─── Metrics collection ──────────────────────────────────────────────

@dataclass
class Snapshot:
    timestamp: float
    elapsed_s: float
    rss_mb: float
    fd_count: int
    requests_sent: int
    requests_failed: int
    rps: float
    rss_children_mb: float = 0.0  # RSS of child processes


@dataclass
class MetricsCollector:
    pid: int
    start_time: float = field(default_factory=time.time)
    snapshots: List[Snapshot] = field(default_factory=list)
    _requests_sent: int = 0
    _requests_failed: int = 0
    _last_snapshot_requests: int = 0
    _last_snapshot_time: float = 0

    def record_request(self, success: bool):
        self._requests_sent += 1
        if not success:
            self._requests_failed += 1

    def take_snapshot(self) -> Snapshot:
        now = time.time()
        elapsed = now - self.start_time

        # RSS memory (main process + children)
        rss_mb = -1
        rss_children_mb = 0.0
        try:
            proc = psutil.Process(self.pid)
            rss_mb = proc.memory_info().rss / (1024 * 1024)
            # Also get children RSS (uvicorn workers)
            for child in proc.children(recursive=True):
                try:
                    rss_children_mb += child.memory_info().rss / (1024 * 1024)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            rss_mb = -1

        # Total RSS = main + children
        total_rss = rss_mb + rss_children_mb

        # File descriptors (main + children)
        fd_count = 0
        try:
            proc = psutil.Process(self.pid)
            pids = [self.pid] + [c.pid for c in proc.children(recursive=True)]
            for pid in pids:
                fd_dir = Path(f"/proc/{pid}/fd")
                if fd_dir.exists():
                    try:
                        fd_count += len(list(fd_dir.iterdir()))
                    except (PermissionError, OSError):
                        pass
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            fd_count = -1

        # RPS since last snapshot
        dt = now - self._last_snapshot_time if self._last_snapshot_time else elapsed
        dreqs = self._requests_sent - self._last_snapshot_requests
        rps = dreqs / dt if dt > 0 else 0

        snap = Snapshot(
            timestamp=now,
            elapsed_s=elapsed,
            rss_mb=total_rss,
            fd_count=fd_count,
            requests_sent=self._requests_sent,
            requests_failed=self._requests_failed,
            rps=rps,
            rss_children_mb=rss_children_mb,
        )
        self.snapshots.append(snap)
        self._last_snapshot_requests = self._requests_sent
        self._last_snapshot_time = now
        return snap

    def print_snapshot(self, snap: Snapshot):
        print(
            f"[{snap.elapsed_s:7.1f}s] "
            f"RSS={snap.rss_mb:8.2f}MB  "
            f"FDs={snap.fd_count:5d}  "
            f"reqs={snap.requests_sent:7d}  "
            f"fail={snap.requests_failed:5d}  "
            f"rps={snap.rps:6.1f}"
        )

    def print_summary(self):
        if len(self.snapshots) < 2:
            print("\nNot enough snapshots for summary")
            return

        first = self.snapshots[0]
        last = self.snapshots[-1]
        duration = last.elapsed_s - first.elapsed_s

        rss_growth = last.rss_mb - first.rss_mb
        rss_rate = rss_growth / (duration / 60) if duration > 0 else 0

        fd_growth = last.fd_count - first.fd_count

        print("\n" + "=" * 70)
        print("MEMORY LEAK REPRODUCTION SUMMARY")
        print("=" * 70)
        print(f"  Duration:         {duration:.0f}s ({duration/60:.1f} min)")
        print(f"  Total requests:   {last.requests_sent}")
        print(f"  Failed requests:  {last.requests_failed}")
        print(f"  Avg RPS:          {last.requests_sent / duration:.1f}" if duration > 0 else "  Avg RPS: N/A")
        print()
        print(f"  RSS start:        {first.rss_mb:.2f} MB")
        print(f"  RSS end:          {last.rss_mb:.2f} MB")
        print(f"  RSS growth:       {rss_growth:+.2f} MB")
        print(f"  RSS growth rate:  {rss_rate:+.2f} MB/min")
        print()
        print(f"  FD start:         {first.fd_count}")
        print(f"  FD end:           {last.fd_count}")
        print(f"  FD growth:        {fd_growth:+d}")
        print()

        if rss_rate > 1.0:
            print("  ⚠️  LEAK DETECTED: RSS growing at >1 MB/min")
        elif rss_rate > 0.5:
            print("  ⚠️  POSSIBLE LEAK: RSS growing at >0.5 MB/min")
        else:
            print("  ✅  No significant memory growth detected")

        if fd_growth > 50:
            print(f"  ⚠️  FD LEAK: {fd_growth} new file descriptors")

        # Print CSV-friendly data for graphing
        print("\n--- CSV DATA (for graphing) ---")
        print("elapsed_s,rss_mb,fd_count,requests_sent,rps")
        for snap in self.snapshots:
            print(f"{snap.elapsed_s:.1f},{snap.rss_mb:.2f},{snap.fd_count},{snap.requests_sent},{snap.rps:.1f}")


# ─── Load generators ─────────────────────────────────────────────────

async def send_anthropic_messages(
    session: aiohttp.ClientSession,
    base_url: str,
    semaphore: asyncio.Semaphore,
    metrics: MetricsCollector,
    stream: bool = False,
):
    """Send a single /v1/messages request."""
    url = f"{base_url}/v1/messages"
    body = ANTHROPIC_MESSAGES_BODY_STREAM if stream else ANTHROPIC_MESSAGES_BODY

    async with semaphore:
        try:
            async with session.post(url, json=body, headers=ANTHROPIC_HEADERS) as resp:
                if stream:
                    # Consume the stream fully
                    async for chunk in resp.content.iter_any():
                        pass
                else:
                    await resp.read()
                metrics.record_request(success=(resp.status == 200))
        except Exception as e:
            metrics.record_request(success=False)


async def send_chat_completions(
    session: aiohttp.ClientSession,
    base_url: str,
    semaphore: asyncio.Semaphore,
    metrics: MetricsCollector,
    stream: bool = False,
):
    """Send a single /chat/completions request."""
    url = f"{base_url}/chat/completions"
    body = CHAT_COMPLETIONS_BODY_STREAM if stream else CHAT_COMPLETIONS_BODY

    async with semaphore:
        try:
            async with session.post(url, json=body, headers=HEADERS) as resp:
                if stream:
                    async for chunk in resp.content.iter_any():
                        pass
                else:
                    await resp.read()
                metrics.record_request(success=(resp.status == 200))
        except Exception as e:
            metrics.record_request(success=False)


async def load_generator(
    base_url: str,
    metrics: MetricsCollector,
    duration_s: int,
    concurrency: int,
    stream_ratio: float = 0.5,
    messages_ratio: float = 0.95,
):
    """
    Generate sustained load against the proxy.

    Args:
        base_url: Proxy base URL
        metrics: MetricsCollector to record results
        duration_s: How long to run in seconds
        concurrency: Max concurrent requests
        stream_ratio: Fraction of requests that are streaming (0.0-1.0)
        messages_ratio: Fraction of requests to /v1/messages vs /chat/completions
    """
    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(
        limit=concurrency * 2,
        limit_per_host=concurrency,
        force_close=False,
    )

    async with aiohttp.ClientSession(connector=connector) as session:
        start = time.time()
        request_id = 0

        while time.time() - start < duration_s:
            # Launch a batch of requests
            batch_size = concurrency
            tasks = []
            for _ in range(batch_size):
                request_id += 1
                is_stream = (request_id % 100) < (stream_ratio * 100)
                is_messages = (request_id % 100) < (messages_ratio * 100)

                if is_messages:
                    tasks.append(
                        send_anthropic_messages(session, base_url, semaphore, metrics, stream=is_stream)
                    )
                else:
                    tasks.append(
                        send_chat_completions(session, base_url, semaphore, metrics, stream=is_stream)
                    )

            await asyncio.gather(*tasks, return_exceptions=True)


# ─── Monitoring ───────────────────────────────────────────────────────

async def poll_debug_memory(base_url: str) -> Optional[dict]:
    """Poll the proxy's /debug/memory/details endpoint for internal stats."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{base_url}/debug/memory/details?top_n=10",
                headers={"Authorization": "Bearer sk-1234"},
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
    except Exception:
        pass
    return None


async def monitor_loop(metrics: MetricsCollector, interval_s: int, duration_s: int, base_url: str = "http://localhost:4000"):
    """Periodically take and print snapshots."""
    start = time.time()
    gc_logged = False
    while time.time() - start < duration_s + 5:
        snap = metrics.take_snapshot()
        metrics.print_snapshot(snap)

        # Every 60s, also poll internal debug memory
        if not gc_logged or int(snap.elapsed_s) % 60 < interval_s:
            debug_data = await poll_debug_memory(base_url)
            if debug_data:
                gc_info = debug_data.get("garbage_collector", {})
                obj_info = debug_data.get("objects", {})
                cache_info = debug_data.get("cache_memory", {})
                proc_info = debug_data.get("process_memory", {})
                print(
                    f"  [internal] gc_gen0={gc_info.get('current_counts', {}).get('generation_0', '?')}  "
                    f"total_objects={obj_info.get('total_tracked', '?')}  "
                    f"process_rss={proc_info.get('ram_usage', {}).get('megabytes', '?')}MB  "
                    f"cache_items: user_keys={cache_info.get('user_api_key_cache', {}).get('num_items', '?')} "
                    f"router={cache_info.get('llm_router_cache', {}).get('num_items', '?')} "
                    f"logging={cache_info.get('proxy_logging_cache', {}).get('num_items', '?')}"
                )
                gc_logged = True

        await asyncio.sleep(interval_s)

    # Final snapshot
    snap = metrics.take_snapshot()
    metrics.print_snapshot(snap)


# ─── Proxy process management ────────────────────────────────────────

def find_proxy_pid(port: int) -> Optional[int]:
    """Find the PID of the LiteLLM proxy listening on the given port."""
    for conn in psutil.net_connections(kind="tcp"):
        if conn.laddr.port == port and conn.status == "LISTEN":
            return conn.pid
    return None


async def wait_for_proxy(base_url: str, timeout: int = 60) -> bool:
    """Wait for the proxy to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{base_url}/health",
                    headers={"Authorization": "Bearer sk-1234"},
                ) as resp:
                    if resp.status == 200:
                        return True
        except Exception:
            pass
        await asyncio.sleep(1)
    return False


# ─── Main ─────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="LiteLLM Memory Leak Reproducer")
    parser.add_argument("--port", type=int, default=4000, help="Proxy port")
    parser.add_argument("--duration", type=int, default=900, help="Test duration in seconds (default: 15 min)")
    parser.add_argument("--concurrency", type=int, default=50, help="Max concurrent requests")
    parser.add_argument("--interval", type=int, default=10, help="Metrics snapshot interval in seconds")
    parser.add_argument("--stream-ratio", type=float, default=0.5, help="Fraction of streaming requests (0.0-1.0)")
    parser.add_argument("--messages-ratio", type=float, default=0.95, help="Fraction /v1/messages vs /chat/completions")
    parser.add_argument("--warmup", type=int, default=30, help="Warmup requests before measuring")
    args = parser.parse_args()

    base_url = f"http://localhost:{args.port}"

    print("=" * 70)
    print("LiteLLM Memory Leak Reproduction")
    print("=" * 70)
    print(f"  Proxy:         {base_url}")
    print(f"  Duration:      {args.duration}s ({args.duration/60:.1f} min)")
    print(f"  Concurrency:   {args.concurrency}")
    print(f"  Stream ratio:  {args.stream_ratio:.0%}")
    print(f"  Messages ratio:{args.messages_ratio:.0%}")
    print(f"  Warmup:        {args.warmup} requests")
    print()

    # Wait for proxy
    print("Waiting for proxy to be ready...")
    if not await wait_for_proxy(base_url, timeout=60):
        print("ERROR: Proxy not available. Start it first:")
        print(f"  poetry run litellm --config scripts/repro_memory_leak_config.yaml --port {args.port}")
        sys.exit(1)
    print("Proxy is ready!")

    # Find proxy PID
    proxy_pid = find_proxy_pid(args.port)
    if proxy_pid is None:
        print(f"WARNING: Could not find proxy PID on port {args.port}. Memory tracking will fail.")
        print("Continuing with PID 0...")
        proxy_pid = 0

    print(f"Proxy PID: {proxy_pid}")

    # Warmup
    if args.warmup > 0:
        print(f"\nWarming up with {args.warmup} requests...")
        warmup_metrics = MetricsCollector(pid=proxy_pid)
        sem = asyncio.Semaphore(args.concurrency)
        connector = aiohttp.TCPConnector(limit=args.concurrency * 2, limit_per_host=args.concurrency)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []
            for i in range(args.warmup):
                if i % 2 == 0:
                    tasks.append(send_anthropic_messages(session, base_url, sem, warmup_metrics))
                else:
                    tasks.append(send_chat_completions(session, base_url, sem, warmup_metrics))
            await asyncio.gather(*tasks, return_exceptions=True)
        print(f"  Warmup done: {warmup_metrics._requests_sent} sent, {warmup_metrics._requests_failed} failed")
        await asyncio.sleep(2)  # Let things settle

    # Main test
    metrics = MetricsCollector(pid=proxy_pid)
    print(f"\nStarting load test for {args.duration}s...")
    print("-" * 70)

    # Take initial snapshot
    snap = metrics.take_snapshot()
    metrics.print_snapshot(snap)

    # Run load generator and monitor concurrently
    await asyncio.gather(
        load_generator(
            base_url=base_url,
            metrics=metrics,
            duration_s=args.duration,
            concurrency=args.concurrency,
            stream_ratio=args.stream_ratio,
            messages_ratio=args.messages_ratio,
        ),
        monitor_loop(
            metrics=metrics,
            interval_s=args.interval,
            duration_s=args.duration,
            base_url=base_url,
        ),
    )

    # Print summary
    metrics.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
