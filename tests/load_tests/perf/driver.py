"""Load driver for the LiteLLM perf harness.

Each request gets a unique user-message content (counter + random suffix) so no
two requests are identical — no cache hits, distinct JSON parsing, distinct
hash/logging keys. Reports p50/p90/p99 + RPS.

Usage:
  uv run python driver.py single
  uv run python driver.py batch [n=100]
  uv run python driver.py rps [concurrency=50] [duration_seconds=30]
"""
import asyncio
import json
import os
import random
import statistics
import string
import sys
import time

import httpx

URL = os.environ.get(
    "LITELLM_PERF_URL", "http://localhost:4001/v1/chat/completions"
)
TIMEOUT = httpx.Timeout(30.0, connect=5.0)


def make_body(i: int) -> bytes:
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=12))
    payload = {
        "model": "fake-gpt",
        "messages": [{"role": "user", "content": f"req-{i}-{suffix}"}],
    }
    return json.dumps(payload, separators=(",", ":")).encode()


async def one_request(client: httpx.AsyncClient, i: int) -> tuple[float, int]:
    body = make_body(i)
    t0 = time.perf_counter()
    r = await client.post(
        URL, content=body, headers={"Content-Type": "application/json"}
    )
    t1 = time.perf_counter()
    return t1 - t0, r.status_code


async def run_single() -> tuple[list[float], list[int], float]:
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        t0 = time.perf_counter()
        lat, status = await one_request(client, 0)
        return [lat], [status], time.perf_counter() - t0


async def run_batch(n: int) -> tuple[list[float], list[int], float]:
    latencies: list[float] = []
    statuses: list[int] = []
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        t0 = time.perf_counter()
        for i in range(n):
            lat, status = await one_request(client, i)
            latencies.append(lat)
            statuses.append(status)
        return latencies, statuses, time.perf_counter() - t0


async def run_rps(
    concurrency: int, duration: float
) -> tuple[list[float], list[int], float]:
    latencies: list[float] = []
    statuses: list[int] = []
    counter = 0
    stop_at = time.perf_counter() + duration
    lock = asyncio.Lock()

    async def worker(client: httpx.AsyncClient) -> None:
        nonlocal counter
        while time.perf_counter() < stop_at:
            async with lock:
                i = counter
                counter += 1
            lat, status = await one_request(client, i)
            latencies.append(lat)
            statuses.append(status)

    limits = httpx.Limits(
        max_connections=concurrency, max_keepalive_connections=concurrency
    )
    async with httpx.AsyncClient(timeout=TIMEOUT, limits=limits) as client:
        t0 = time.perf_counter()
        await asyncio.gather(*[worker(client) for _ in range(concurrency)])
        return latencies, statuses, time.perf_counter() - t0


def pctl(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    if not xs:
        return 0.0
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def report(latencies: list[float], statuses: list[int], t_total: float) -> None:
    n = len(latencies)
    ok = sum(1 for s in statuses if 200 <= s < 300)
    lat_ms = [x * 1000 for x in latencies]
    print(f"requests:     {n}")
    print(f"ok (2xx):     {ok}")
    print(f"non-2xx:      {n - ok}")
    print(f"duration:     {t_total:.3f}s")
    print(f"rps:          {n / t_total:.1f}" if t_total > 0 else "rps:          n/a")
    if lat_ms:
        print(f"latency min:  {min(lat_ms):.2f} ms")
        print(f"latency p50:  {pctl(lat_ms, 0.50):.2f} ms")
        print(f"latency p90:  {pctl(lat_ms, 0.90):.2f} ms")
        print(f"latency p99:  {pctl(lat_ms, 0.99):.2f} ms")
        print(f"latency max:  {max(lat_ms):.2f} ms")
        print(f"latency mean: {statistics.mean(lat_ms):.2f} ms")
    if n != ok:
        print("\nnon-2xx statuses seen:")
        from collections import Counter

        for code, count in Counter(statuses).most_common():
            if not (200 <= code < 300):
                print(f"  {code}: {count}")


def main() -> None:
    argv = sys.argv[1:]
    if not argv:
        print(
            f"usage: {sys.argv[0]} {{single|batch [n]|rps [concurrency] [duration]}}",
            file=sys.stderr,
        )
        sys.exit(1)
    mode = argv[0]
    if mode == "single":
        latencies, statuses, t_total = asyncio.run(run_single())
    elif mode == "batch":
        n = int(argv[1]) if len(argv) > 1 else 100
        latencies, statuses, t_total = asyncio.run(run_batch(n))
    elif mode == "rps":
        concurrency = int(argv[1]) if len(argv) > 1 else 50
        duration = float(argv[2]) if len(argv) > 2 else 30.0
        latencies, statuses, t_total = asyncio.run(run_rps(concurrency, duration))
    else:
        print(f"error: unknown mode '{mode}'", file=sys.stderr)
        sys.exit(1)
    report(latencies, statuses, t_total)


if __name__ == "__main__":
    main()
