#!/usr/bin/env python3
"""
Load generator for LiteLLM proxy memory leak reproduction.

Sends a high volume of chat completion requests to the proxy and
periodically reports throughput + hits the memory diagnostics endpoint.

Usage:
    python load_generator.py [--url URL] [--concurrency N] [--duration SECS]
"""

import argparse
import asyncio
import json
import sys
import time

import aiohttp


async def send_request(session: aiohttp.ClientSession, url: str, api_key: str) -> bool:
    payload = {
        "model": "fake-model",
        "messages": [{"role": "user", "content": "Say hello"}],
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    try:
        async with session.post(
            f"{url}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            await resp.read()
            return resp.status == 200
    except Exception:
        return False


async def check_memory(session: aiohttp.ClientSession, url: str, api_key: str) -> dict:
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        async with session.get(
            f"{url}/debug/memory/summary",
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            if resp.status == 200:
                return await resp.json()
    except Exception as e:
        return {"error": str(e)}
    return {}


async def worker(
    session: aiohttp.ClientSession,
    url: str,
    api_key: str,
    stats: dict,
    stop_event: asyncio.Event,
):
    while not stop_event.is_set():
        ok = await send_request(session, url, api_key)
        if ok:
            stats["success"] += 1
        else:
            stats["fail"] += 1
        stats["total"] += 1


async def reporter(
    session: aiohttp.ClientSession,
    url: str,
    api_key: str,
    stats: dict,
    stop_event: asyncio.Event,
    interval: float = 5.0,
):
    prev_total = 0
    prev_time = time.monotonic()

    while not stop_event.is_set():
        await asyncio.sleep(interval)
        now = time.monotonic()
        dt = now - prev_time
        delta = stats["total"] - prev_total
        rps = delta / dt if dt > 0 else 0
        prev_total = stats["total"]
        prev_time = now

        mem = await check_memory(session, url, api_key)
        mem_summary = mem.get("memory", {}).get("summary", "N/A")
        queue_info = mem.get("spend_log_queue", {})
        queue_len = queue_info.get("queue_length", "N/A")
        queue_pct = queue_info.get("usage_percent", "N/A")
        status = mem.get("status", "N/A")

        print(
            f"[{time.strftime('%H:%M:%S')}] "
            f"rps={rps:.0f}  total={stats['total']}  "
            f"ok={stats['success']}  fail={stats['fail']}  "
            f"| mem={mem_summary}  status={status}  "
            f"| spend_queue={queue_len} ({queue_pct}%)",
            flush=True,
        )


async def main():
    parser = argparse.ArgumentParser(description="LiteLLM proxy load generator")
    parser.add_argument("--url", default="http://127.0.0.1:4000", help="Proxy URL")
    parser.add_argument("--api-key", default="sk-repro-test-1234", help="API key")
    parser.add_argument("--concurrency", type=int, default=50, help="Concurrent requests")
    parser.add_argument("--duration", type=int, default=120, help="Duration in seconds")
    args = parser.parse_args()

    print(f"Load generator: {args.concurrency} concurrent workers, {args.duration}s duration")
    print(f"Target: {args.url}")
    print()

    # Wait for proxy to be ready
    print("Waiting for proxy to be ready...", end="", flush=True)
    connector = aiohttp.TCPConnector(limit=args.concurrency + 10)
    async with aiohttp.ClientSession(connector=connector) as session:
        for _ in range(60):
            try:
                async with session.get(
                    f"{args.url}/health",
                    timeout=aiohttp.ClientTimeout(total=2),
                ) as resp:
                    if resp.status == 200:
                        print(" ready!")
                        break
            except Exception:
                pass
            await asyncio.sleep(1)
            print(".", end="", flush=True)
        else:
            print("\nProxy not ready after 60s, starting anyway.")

        stats = {"total": 0, "success": 0, "fail": 0}
        stop_event = asyncio.Event()

        tasks = []
        for _ in range(args.concurrency):
            tasks.append(
                asyncio.create_task(worker(session, args.url, args.api_key, stats, stop_event))
            )
        tasks.append(
            asyncio.create_task(reporter(session, args.url, args.api_key, stats, stop_event))
        )

        await asyncio.sleep(args.duration)
        stop_event.set()
        await asyncio.gather(*tasks, return_exceptions=True)

    print(f"\nDone. Total={stats['total']} Success={stats['success']} Fail={stats['fail']}")


if __name__ == "__main__":
    asyncio.run(main())
