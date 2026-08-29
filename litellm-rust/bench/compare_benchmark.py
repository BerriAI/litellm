"""Comparison benchmark: Rust gateway vs Python proxy.

Runs the same load test against both implementations and compares
throughput, latency, and error rates.
"""

import asyncio
import json
import os
import subprocess
import sys
import time
import statistics

import aiohttp

MOCK_PORT = 11434
RUST_PORT = 4001
PYTHON_PORT = 4002
MASTER_KEY = "bench-master-key"
RUST_BINARY = os.path.join(os.path.dirname(__file__), "..", "target", "release", "litellm-ai-gateway.exe")
PYTHON_CONFIG = os.path.join(os.path.dirname(__file__), "python_proxy_config.yaml")
RUST_CONFIG = os.path.join(os.path.dirname(__file__), "..", "bench_config.yaml")

REQUEST_BODY = {
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello, benchmark test."}],
    "max_tokens": 50,
}

CONCURRENCY_LEVELS = [1, 5, 10, 25, 50, 100]
REQUESTS_PER_LEVEL = 500


async def send_request(session, url, headers, body):
    start = time.perf_counter()
    try:
        async with session.post(url, json=body, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as resp:
            await resp.read()
            elapsed = (time.perf_counter() - start) * 1000
            return resp.status, elapsed
    except Exception:
        elapsed = (time.perf_counter() - start) * 1000
        return 0, elapsed


async def run_bench(concurrency, num_requests, url, headers, body):
    connector = aiohttp.TCPConnector(limit=concurrency + 10)
    async with aiohttp.ClientSession(connector=connector) as session:
        sem = asyncio.Semaphore(concurrency)
        latencies = []
        success = 0
        fail = 0

        async def bounded():
            nonlocal success, fail
            async with sem:
                status, lat = await send_request(session, url, headers, body)
                if status == 200:
                    success += 1
                    latencies.append(lat)
                else:
                    fail += 1

        start = time.perf_counter()
        await asyncio.gather(*[asyncio.create_task(bounded()) for _ in range(num_requests)])
        duration = time.perf_counter() - start

        sorted_lat = sorted(latencies) if latencies else [0]
        p = lambda pct: sorted_lat[min(int(len(sorted_lat) * pct / 100), len(sorted_lat) - 1)]

        return {
            "concurrency": concurrency,
            "rps": success / duration if duration > 0 else 0,
            "mean": statistics.mean(latencies) if latencies else 0,
            "p50": p(50),
            "p95": p(95),
            "p99": p(99),
            "max": max(latencies) if latencies else 0,
            "success": success,
            "fail": fail,
            "duration": duration,
        }


async def wait_for_port(port, timeout=30):
    url = f"http://127.0.0.1:{port}/health/liveness"
    deadline = time.time() + timeout
    async with aiohttp.ClientSession() as session:
        while time.time() < deadline:
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as resp:
                    if resp.status == 200:
                        return True
            except Exception:
                pass
            await asyncio.sleep(0.3)
    return False


def start_mock():
    return subprocess.Popen(
        [sys.executable, os.path.join(os.path.dirname(__file__), "mock_upstream.py")],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def start_rust():
    env = os.environ.copy()
    env["LITELLM_MASTER_KEY"] = MASTER_KEY
    env["LITELLM_YAML_CONFIG"] = RUST_CONFIG
    env["RUST_LOG"] = "error"
    return subprocess.Popen([RUST_BINARY], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def start_python():
    env = os.environ.copy()
    env["LITELLM_MASTER_KEY"] = MASTER_KEY
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.Popen(
        ["litellm",
         "--config", PYTHON_CONFIG,
         "--port", str(PYTHON_PORT),
         "--num_workers", "4"],
        cwd=os.path.join(os.path.dirname(__file__), "..", ".."),
        env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


async def bench_server(name, port, concurrency_levels, requests_per_level):
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {MASTER_KEY}", "Content-Type": "application/json"}

    # Warmup
    await run_bench(10, 50, url, headers, REQUEST_BODY)

    results = []
    for conc in concurrency_levels:
        r = await run_bench(conc, requests_per_level, url, headers, REQUEST_BODY)
        results.append(r)
    return results


async def main():
    print("=" * 90)
    print("  LiteLLM: Rust Gateway vs Python Proxy - Comparison Benchmark")
    print("=" * 90)

    mock = start_mock()
    await asyncio.sleep(1)

    all_results = {}

    # --- Rust ---
    print("\n[1/2] Starting Rust gateway on port", RUST_PORT)
    rust = start_rust()
    if not await wait_for_port(RUST_PORT, timeout=30):
        print("ERROR: Rust gateway did not start")
        rust.kill()
        mock.kill()
        return

    rust_size = os.path.getsize(RUST_BINARY) / (1024 * 1024)
    print(f"  Binary size: {rust_size:.1f} MB")

    t0 = time.perf_counter()
    print("  Benchmarking Rust gateway...")
    all_results["rust"] = await bench_server("Rust", RUST_PORT, CONCURRENCY_LEVELS, REQUESTS_PER_LEVEL)
    rust_bench_time = time.perf_counter() - t0
    print(f"  Rust benchmark completed in {rust_bench_time:.1f}s")

    rust.terminate()
    try:
        rust.wait(timeout=5)
    except subprocess.TimeoutExpired:
        rust.kill()

    await asyncio.sleep(1)

    # --- Python ---
    print("\n[2/2] Starting Python proxy on port", PYTHON_PORT)
    python = start_python()
    if not await wait_for_port(PYTHON_PORT, timeout=120):
        print("ERROR: Python proxy did not start")
        python.kill()
        mock.kill()
        return

    t0 = time.perf_counter()
    print("  Benchmarking Python proxy...")
    all_results["python"] = await bench_server("Python", PYTHON_PORT, CONCURRENCY_LEVELS, REQUESTS_PER_LEVEL)
    python_bench_time = time.perf_counter() - t0
    print(f"  Python benchmark completed in {python_bench_time:.1f}s")

    python.terminate()
    try:
        python.wait(timeout=5)
    except subprocess.TimeoutExpired:
        python.kill()
    mock.terminate()
    try:
        mock.wait(timeout=5)
    except subprocess.TimeoutExpired:
        mock.kill()

    # --- Comparison ---
    print("\n" + "=" * 90)
    print("  RESULTS COMPARISON")
    print("=" * 90)

    print(f"\n{'Conc':>5} | {'Rust RPS':>10} {'p50':>8} {'p95':>8} {'p99':>8} | {'Python RPS':>10} {'p50':>8} {'p95':>8} {'p99':>8} | {'RPS Speedup':>12}")
    print("-" * 105)

    for i, conc in enumerate(CONCURRENCY_LEVELS):
        r = all_results["rust"][i]
        p = all_results["python"][i]
        speedup = r["rps"] / p["rps"] if p["rps"] > 0 else float("inf")
        print(
            f"{conc:>5} | "
            f"{r['rps']:>10.1f} {r['p50']:>7.2f}ms {r['p95']:>7.2f}ms {r['p99']:>7.2f}ms | "
            f"{p['rps']:>10.1f} {p['p50']:>7.2f}ms {p['p95']:>7.2f}ms {p['p99']:>7.2f}ms | "
            f"{speedup:>10.1f}x"
        )

    rust_peak = max(r["rps"] for r in all_results["rust"])
    python_peak = max(p["rps"] for p in all_results["python"])
    peak_speedup = rust_peak / python_peak if python_peak > 0 else float("inf")

    print(f"\n  Rust peak RPS:   {rust_peak:,.0f}")
    print(f"  Python peak RPS: {python_peak:,.0f}")
    print(f"  Peak speedup:    {peak_speedup:.1f}x")
    print(f"  Rust binary:     {rust_size:.1f} MB")
    print(f"  Rust bench time: {rust_bench_time:.1f}s")
    print(f"  Python bench time: {python_bench_time:.1f}s")

    print("\n" + "=" * 90)


if __name__ == "__main__":
    asyncio.run(main())
