#!/usr/bin/env python3
"""
Measure /chat/completions latency using a single shared HTTP client.

Uses one httpx.AsyncClient and fires all requests in one asyncio.gather() batch.
Eliminates per-user client creation and connection overhead.
"""

import asyncio
import time

import httpx

BASE_URL = "http://localhost:4000"
API_KEY = "sk-1234"
MODEL = "gpt-o1"
NUM_REQUESTS = 200
TIMEOUT = 30.0
PAYLOAD = {"model": MODEL, "messages": [{"role": "user", "content": "Say hello in one word."}]}
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# Module-level client configured for throughput
client = httpx.AsyncClient(
    limits=httpx.Limits(
        max_connections=200,
        max_keepalive_connections=100,
        keepalive_expiry=30.0,
    ),
    timeout=TIMEOUT,
    http2=True,
)


async def one_request(client: httpx.AsyncClient, req_id: int) -> tuple[int, float, bool]:
    """Single request; returns (req_id, latency_sec, success)."""
    start = time.perf_counter()
    try:
        resp = await client.post(
            f"{BASE_URL}/chat/completions",
            json=PAYLOAD,
            headers=HEADERS,
            timeout=TIMEOUT,
        )
        resp.read()
        ok = resp.is_success
    except Exception:
        ok = False
    lat = time.perf_counter() - start
    return (req_id, lat, ok)


async def main() -> None:
    print(f"=== {NUM_REQUESTS} requests, single shared client, batch fire ===")
    print("Latency = time from request start until last byte of response received")
    print()

    try:
        tasks = [one_request(client, i) for i in range(1, NUM_REQUESTS + 1)]
        results = await asyncio.gather(*tasks)
    finally:
        await client.aclose()

    # Sort by req_id for stable output, print as we go
    for req_id, lat, ok in sorted(results, key=lambda r: r[0]):
        status = "[OK]" if ok else "[FAIL]"
        print(f"  {status} Request {req_id:3d}: {lat:.3f}s", flush=True)

    # Summary
    lats = [r[1] for r in results if r[2]]
    if not lats:
        print(f"\n[OK] 0 succeeded, [FAIL] {NUM_REQUESTS} failed")
        return

    n = len(lats)
    avg = sum(lats) / n
    sorted_lat = sorted(lats)
    p95 = sorted_lat[int((n - 1) * 0.95)]
    p99 = sorted_lat[int((n - 1) * 0.99)]

    print(f"\n[OK] {n} succeeded, [FAIL] {NUM_REQUESTS - n} failed")
    print("\nLatency (successful):")
    print(f"  min:  {min(lats):.3f}s")
    print(f"  avg:  {avg:.3f}s")
    print(f"  max:  {max(lats):.3f}s")
    print(f"  p95:  {p95:.3f}s")
    print(f"  p99:  {p99:.3f}s")
    print("\nRequests above threshold (successful only):")
    for t in range(1, 11):
        above = sum(1 for x in lats if x > t)
        print(f"  Above {t}s: {above}/{n} ({100.0 * above / n:.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())
