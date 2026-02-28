#!/usr/bin/env python3
"""
Direct instrumentation profiling - patches the proxy's hot path functions
to measure where time is spent.
"""
import asyncio
import time
import json
import statistics
import aiohttp


async def benchmark_direct_vs_proxy(num_requests=200):
    """Compare direct mock server performance vs through proxy."""

    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0, keepalive_timeout=120)
    timeout = aiohttp.ClientTimeout(total=120)

    async def do_streaming_request(session, url, label):
        payload = {
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 150,
            "stream": True,
        }
        start = time.monotonic()
        ttfb = None
        n_chunks = 0
        try:
            async with session.post(url, json=payload, headers={"Content-Type": "application/json"}) as resp:
                async for line in resp.content:
                    line_str = line.decode("utf-8", errors="ignore").strip()
                    if line_str.startswith("data: "):
                        if ttfb is None:
                            ttfb = time.monotonic() - start
                        data = line_str[6:]
                        if data == "[DONE]":
                            break
                        n_chunks += 1
            total = time.monotonic() - start
            return {"total": total, "ttfb": ttfb or total, "chunks": n_chunks, "ok": True}
        except Exception as e:
            return {"total": time.monotonic() - start, "ok": False, "error": str(e)}

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Direct to mock server
        print(f"\n=== Direct to mock server ({num_requests} concurrent) ===")
        tasks = [
            asyncio.create_task(do_streaming_request(session, "http://localhost:9999/v1/chat/completions", "direct"))
            for _ in range(num_requests)
        ]
        results_direct = await asyncio.gather(*tasks)

        successes = [r for r in results_direct if r["ok"]]
        if successes:
            lats = sorted(r["total"] for r in successes)
            ttfbs = sorted(r["ttfb"] for r in successes)
            print(f"  Success: {len(successes)}/{num_requests}")
            print(f"  Latency avg={statistics.mean(lats):.3f}s p50={lats[len(lats)//2]:.3f}s p99={lats[int(len(lats)*0.99)]:.3f}s")
            print(f"  TTFB    avg={statistics.mean(ttfbs):.3f}s p50={ttfbs[len(ttfbs)//2]:.3f}s p99={ttfbs[int(len(ttfbs)*0.99)]:.3f}s")

        # Through proxy
        print(f"\n=== Through LiteLLM proxy ({num_requests} concurrent) ===")
        tasks = [
            asyncio.create_task(do_streaming_request(session, "http://localhost:4000/chat/completions", "proxy"))
            for _ in range(num_requests)
        ]
        results_proxy = await asyncio.gather(*tasks)

        successes = [r for r in results_proxy if r["ok"]]
        if successes:
            lats = sorted(r["total"] for r in successes)
            ttfbs = sorted(r["ttfb"] for r in successes)
            print(f"  Success: {len(successes)}/{num_requests}")
            print(f"  Latency avg={statistics.mean(lats):.3f}s p50={lats[len(lats)//2]:.3f}s p99={lats[int(len(lats)*0.99)]:.3f}s")
            print(f"  TTFB    avg={statistics.mean(ttfbs):.3f}s p50={ttfbs[len(ttfbs)//2]:.3f}s p99={ttfbs[int(len(ttfbs)*0.99)]:.3f}s")

            # Calculate overhead
            direct_avg = statistics.mean([r["total"] for r in results_direct if r["ok"]])
            proxy_avg = statistics.mean(lats)
            print(f"\n  Proxy overhead: {proxy_avg - direct_avg:.3f}s avg ({proxy_avg/direct_avg:.1f}x slower)")


if __name__ == "__main__":
    for n in [50, 200, 500]:
        asyncio.run(benchmark_direct_vs_proxy(n))
