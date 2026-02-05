#!/usr/bin/env python3
"""Measure /chat/completions latency against the LiteLLM proxy."""

""

import asyncio
import time

import httpx

"""
What makes this file be capable of reproducing the latency issue when using prometheus callbacks:
1. One HTTP client per user, and it sends requests as fast as possible.
2. Don't send too many requests at once but don't be too slow either.
3. Passing a user_id to call makes the issue more reproducible.
4. Sending the requests in waves makes the issue more reproducible.
"""

# -----------------------------------------------------------------------------
# Constants (edit as needed)
# -----------------------------------------------------------------------------
BASE_URL = "http://localhost:4000"
API_KEY = "sk-1234"
MODEL = "gpt-o1"  # must match a model_name in your proxy config (e.g. test_config_123.yaml)
NUM_REQUESTS = 5000
NUM_CONCURRENT = 42  # Concurrent users; each has one connection, reuses it for their requests
MESSAGES = [{"role": "user", "content": "Say hello in one word."}]
TIMEOUT = 30000.0
# -----------------------------------------------------------------------------

_results: list[tuple[float, bool]] = []  # shared; asyncio is single-threaded so no lock needed


async def run_user(
    user_id: int,
    num_requests: int,
    base_url: str,
    path: str,
    payload: dict,
    headers: dict,
) -> None:
    """One user: one client, fires requests as fast as possible."""
    async with httpx.AsyncClient(base_url=base_url, timeout=TIMEOUT) as client:
        for req_num in range(1, num_requests + 1):
            start = time.perf_counter()
            try:
                resp = await client.post(path, json=payload, headers=headers)
                resp.read()  # consume full body - timer stops after last byte received
                ok = resp.is_success
            except Exception:
                ok = False
            lat = time.perf_counter() - start
            _results.append((lat, ok))
            status = "[OK]" if ok else "[FAIL]"
            print(f"  {status} User {user_id} request {req_num:3d}: {lat:.3f}s", flush=True)


async def main() -> None:
    base_url = BASE_URL.rstrip("/")
    path = "/chat/completions"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    base_per_user = NUM_REQUESTS // NUM_CONCURRENT
    remainder = NUM_REQUESTS - base_per_user * NUM_CONCURRENT

    print(f"=== {NUM_REQUESTS} requests, {NUM_CONCURRENT} users, fire-as-fast-as-possible ===", flush=True)
    print("Latency = time from request start until last byte of response received", flush=True)

    _results.clear()
    tasks = []
    for user_id in range(1, NUM_CONCURRENT + 1):
        count = base_per_user + (1 if user_id <= remainder else 0)
        if count > 0:
            payload = {"model": MODEL, "messages": MESSAGES, "user": f"user_{user_id}"}
            tasks.append(asyncio.create_task(run_user(user_id, count, base_url, path, payload, headers)))
    await asyncio.gather(*tasks)

    all_results = _results
    if all_results:
        total = len(all_results)
        success_count = sum(1 for _, ok in all_results if ok)
        failed_count = total - success_count
        all_latencies = [lat for lat, ok in all_results if ok]
        if all_latencies:
            n = len(all_latencies)
            avg = sum(all_latencies) / n
            sorted_lat = sorted(all_latencies)
            p95 = sorted_lat[int((n - 1) * 0.95)]
            p99 = sorted_lat[int((n - 1) * 0.99)]
            print(f"\n[OK] {success_count} succeeded, [FAIL] {failed_count} failed")
            print(f"\nLatency (successful):")
            print(f"  min:  {min(all_latencies):.3f}s")
            print(f"  avg:  {avg:.3f}s")
            print(f"  max:  {max(all_latencies):.3f}s")
            print(f"  p95:  {p95:.3f}s")
            print(f"  p99:  {p99:.3f}s")
            print("\nRequests above threshold (successful only):")
            for threshold in range(1, 11):
                above = sum(1 for t in all_latencies if t > threshold)
                print(f"  Above {threshold}s: {above}/{len(all_latencies)} ({100.0 * above / len(all_latencies):.1f}%)")
        else:
            print(f"\n[OK] {success_count} succeeded, [FAIL] {failed_count} failed")


if __name__ == "__main__":
    asyncio.run(main())
