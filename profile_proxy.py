#!/usr/bin/env python3
"""
Profile LiteLLM proxy streaming overhead by measuring time in key functions.
"""
import asyncio
import time
import json
import aiohttp


async def profile_single_request():
    """Send a single streaming request and measure chunk timings."""
    url = "http://localhost:4000/chat/completions"
    payload = {
        "model": "gpt-4.1-mini",
        "messages": [{"role": "user", "content": "Hello, write a short paragraph about AI."}],
        "max_tokens": 50,
        "stream": True,
    }

    connector = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=60)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        start = time.monotonic()
        async with session.post(url, json=payload, headers={"Content-Type": "application/json"}) as resp:
            first_byte = time.monotonic()
            print(f"Time to first byte: {first_byte - start:.4f}s")

            chunk_times = []
            prev_time = first_byte
            async for line in resp.content:
                now = time.monotonic()
                line_str = line.decode("utf-8", errors="ignore").strip()
                if line_str.startswith("data: "):
                    data = line_str[6:]
                    if data == "[DONE]":
                        print(f"[DONE] at {now - start:.4f}s")
                        break
                    chunk_times.append(now - prev_time)
                    prev_time = now

            total = time.monotonic() - start
            print(f"Total time: {total:.4f}s")
            print(f"Chunks received: {len(chunk_times)}")
            if chunk_times:
                print(f"Avg inter-chunk: {sum(chunk_times)/len(chunk_times)*1000:.2f}ms")
                print(f"Max inter-chunk: {max(chunk_times)*1000:.2f}ms")
                print(f"Min inter-chunk: {min(chunk_times)*1000:.2f}ms")


async def profile_concurrent(num_requests=100):
    """Send concurrent requests and see how per-chunk latency degrades."""
    url = "http://localhost:4000/chat/completions"
    payload = {
        "model": "gpt-4.1-mini",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 50,
        "stream": True,
    }

    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    timeout = aiohttp.ClientTimeout(total=120)

    results = []

    async def do_request(session):
        start = time.monotonic()
        try:
            async with session.post(url, json=payload, headers={"Content-Type": "application/json"}) as resp:
                ttfb = None
                n_chunks = 0
                async for line in resp.content:
                    line_str = line.decode("utf-8", errors="ignore").strip()
                    if line_str.startswith("data: "):
                        data = line_str[6:]
                        if data == "[DONE]":
                            break
                        if ttfb is None:
                            ttfb = time.monotonic() - start
                        n_chunks += 1
                total = time.monotonic() - start
                return {"total": total, "ttfb": ttfb or total, "chunks": n_chunks}
        except Exception as e:
            return {"total": time.monotonic() - start, "error": str(e)}

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        print(f"\n--- Profiling {num_requests} concurrent streaming requests ---")
        tasks = [asyncio.create_task(do_request(session)) for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)

    successes = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]
    print(f"Successes: {len(successes)}, Errors: {len(errors)}")
    if successes:
        latencies = sorted([r["total"] for r in successes])
        ttfbs = sorted([r["ttfb"] for r in successes])
        print(f"Latency - avg: {sum(latencies)/len(latencies):.3f}s, p50: {latencies[len(latencies)//2]:.3f}s, p99: {latencies[int(len(latencies)*0.99)]:.3f}s, max: {latencies[-1]:.3f}s")
        print(f"TTFB    - avg: {sum(ttfbs)/len(ttfbs):.3f}s, p50: {ttfbs[len(ttfbs)//2]:.3f}s, p99: {ttfbs[int(len(ttfbs)*0.99)]:.3f}s, max: {ttfbs[-1]:.3f}s")
    if errors:
        print(f"Sample errors: {errors[:3]}")


if __name__ == "__main__":
    print("=== Single request profile ===")
    asyncio.run(profile_single_request())

    print("\n=== 10 concurrent ===")
    asyncio.run(profile_concurrent(10))

    print("\n=== 100 concurrent ===")
    asyncio.run(profile_concurrent(100))

    print("\n=== 500 concurrent ===")
    asyncio.run(profile_concurrent(500))
